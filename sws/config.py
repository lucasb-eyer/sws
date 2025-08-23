from collections.abc import Mapping
from .simpleeval import EvalWithCompoundTypes


class FinalizeError(Exception):
    pass


class CycleError(FinalizeError):
    pass


class Computed:
    def __init__(self, fn):
        self.fn = fn

    def __repr__(self) -> str:
        name = getattr(self.fn, "__name__", None) or "<lambda>"
        return f"Computed({name})"


def lazy(fn):
    return Computed(fn)


class _BaseView:
    """Internal mixin for shared view behavior over a flat store with prefix."""

    def _full(self, key):
        return f"{self._prefix}{key}" if self._prefix else str(key)

    def _iter_child_segments(self):
        p = self._prefix
        plen = len(p)
        seen = set()
        for k in self._store:
            if not k.startswith(p):
                continue
            rest = k[plen:]
            if not rest:
                continue
            seg = rest.split(".", 1)[0]
            if seg not in seen:
                seen.add(seg)
                yield seg

    def __contains__(self, key):
        full = self._full(key)
        return (full in self._store) or any(k.startswith(full + ".") for k in self._store)

    def __len__(self):
        return sum(1 for _ in self._iter_child_segments())

    def to_flat_dict(self):
        d = {}
        p = self._prefix
        plen = len(p)
        for k, v in self._store.items():
            if k.startswith(p):
                d[k[plen:]] = v
        return d

    def to_dict(self):
        out = {}
        for k, v in self.to_flat_dict().items():
            cur = out
            parts = k.split(".")
            for part in parts[:-1]:
                if part not in cur or not isinstance(cur[part], dict):
                    cur[part] = {}
                cur = cur[part]
            cur[parts[-1]] = v
        return out

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _flatten(base, value):
    # Yield (full_key, leaf_value) pairs, flattening nested mappings
    if isinstance(value, Mapping):
        for k, v in dict(value).items():
            fk = f"{base}.{k}" if base else k
            yield from _flatten(fk, v)
    else:
        yield base, value


class Config(_BaseView):
    """Flattened, dotted-key config with prefix views.

    - Stores all values in a single flat dict of full dotted keys.
    - Accessing a group returns a prefixed view sharing the same store.
    - Disallows shadowing: a name cannot be both a leaf and a group.
    """

    def __init__(self, *, _store=None, _prefix="", **kw):
        object.__setattr__(self, "_store", _store or {})
        object.__setattr__(self, "_prefix", _prefix or "")

        # initialize from kwargs
        for k, v in kw.items():
            self._assign(self._full(k), v)

    def _assign_leaf(self, full, value):
        # Forbid assigning a leaf where a group exists
        if any(k.startswith(full + ".") for k in self._store):
            raise ValueError(f"Cannot set leaf '{full}': group with same name exists")
        self._store[full] = value

    def _assign(self, full, value):
        # Assign value at 'full'; if it's a mapping, flatten under that prefix.
        if isinstance(value, Mapping):
            # Forbid creating a group where a leaf exists
            if full in self._store:
                raise ValueError(f"Cannot set group '{full}': leaf with same name exists")
            for fk, v in _flatten(full, value):
                self._assign_leaf(fk, v)
        else:
            self._assign_leaf(full, value)

    # Attribute access
    def __getattr__(self, name):
        if name.startswith("_"):  # TODO: why?
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        if name.startswith("_"):
            return object.__setattr__(self, name, value)
        self[name] = value

    # MutableMapping core
    def __getitem__(self, key):
        full = self._full(key)
        if full in self._store:
            raise TypeError("Config is write-only; use lazy(...) or finalize() to read")
        prefix = full + "."
        if any(k.startswith(prefix) for k in self._store):
            return Config(_store=self._store, _prefix=full + ".")
        raise KeyError(key)

    def __setitem__(self, key, value):
        full = self._full(key)
        self._assign(full, value)

    def __delitem__(self, key):
        full = self._full(key)
        if full in self._store:
            del self._store[full]
            return
        prefix = full + "."
        to_del = [k for k in self._store if k.startswith(prefix)]
        if not to_del:
            raise KeyError(key)
        for k in to_del:
            del self._store[k]

    def __iter__(self):
        return self._iter_child_segments()

    # Representations and conversions
    def __repr__(self):
        return f"Config(_store={self.to_dict()!r})"

    # Allow safe value lookup bypassing read restriction for convenience
    def get(self, key, default=None):
        full = self._full(key)
        if full in self._store:
            return self._store[full]
        # If it's a group or missing, fall back to default
        return default

    # to_dict and to_flat_dict inherited from _BaseView

    # Finalization to an immutable, resolved config
    def finalize(self, argv=None):
        store = dict(self._store)  # snapshot we can mutate with overrides

        memo = {}
        visiting = {}
        stack = []

        def _deep_resolve_in_containers(value, parent_prefix):
            # Resolve any Computed that may appear inside container values
            if isinstance(value, Computed):
                view = _ResolvedView(store, _resolve, parent_prefix)
                return _deep_resolve_in_containers(value.fn(view), parent_prefix)
            if isinstance(value, (list, tuple)):
                return tuple(_deep_resolve_in_containers(v, parent_prefix) for v in value)
            if isinstance(value, set):
                return frozenset(_deep_resolve_in_containers(v, parent_prefix) for v in value)
            # Dict leaves should not occur (assigning Mapping flattens)
            if isinstance(value, Mapping):
                raise FinalizeError("Mapping encountered at leaf during finalize")
            return value

        def _resolve(full):
            if full in memo:
                return memo[full]
            if visiting.get(full):
                cycle = " -> ".join(stack + [full])
                raise CycleError(f"Cycle detected: {cycle}")
            if full not in store:
                # Not a leaf; allow resolution only for existing groups via views
                raise KeyError(full)
            visiting[full] = 1
            stack.append(full)
            try:
                val = store[full]
                if isinstance(val, Computed):
                    parent = full.rsplit(".", 1)[0] + "." if "." in full else ""
                    view = _ResolvedView(store, _resolve, parent)
                    try:
                        out = val.fn(view)
                    except CycleError:
                        # Preserve cycle errors without wrapping so callers can assert on type
                        raise
                    except AttributeError as e:
                        # Treat missing attribute access the same as missing key for clarity
                        raise FinalizeError(
                            f"Missing key {e.args[0]!r} while resolving '{full}'"
                        ) from e
                    except KeyError as e:
                        raise FinalizeError(
                            f"Missing key {e.args[0]!r} while resolving '{full}'"
                        ) from e
                    except Exception as e:  # pragma: no cover - general safety
                        raise FinalizeError(f"Error while resolving '{full}': {e}") from e
                    out = _deep_resolve_in_containers(out, parent)
                else:
                    out = _deep_resolve_in_containers(val, full.rsplit(".", 1)[0] + "." if "." in full else "")
                memo[full] = out
                return out
            finally:
                stack.pop()
                visiting.pop(full, None)

        # Apply overrides (argv) if provided
        if argv is not None:
            # Helpers to parse argv tokens (supports key value and key=value)
            def _iter_key_vals(args):
                args = list(args)
                i = 0
                n = len(args)
                while i < n:
                    token = args[i]
                    if "=" in token:
                        k, v = token.split("=", 1)
                        yield k, v
                        i += 1
                    else:
                        if i + 1 >= n:
                            break
                        yield token, args[i + 1]
                        i += 2

            cfg = Config(_store=store)
            for key, val in _iter_key_vals(argv):
                if key not in cfg:
                    raise AttributeError(key)
                # Disallow assigning to a group without reading via cfg[key]
                if any(k.startswith(key + ".") for k in store):
                    raise AttributeError(f"{key!r} is a group; set leaves like {key}.<name>")
                # Evaluate value with access to current resolved view via 'c'
                # Ensure fresh resolution per assignment (clear memo/cache)
                memo.clear(); visiting.clear(); stack.clear()
                try:
                    evaluated = EvalWithCompoundTypes(names={"c": _ResolvedView(store, _resolve, "")}).eval(val)
                except Exception:
                    evaluated = val
                cfg[key] = evaluated

        resolved = {}
        # Only resolve leaves (keys present in the flat store)
        for k in list(store.keys()):
            resolved[k] = _resolve(k)

        return FinalConfig(_store=resolved, _prefix=self._prefix)


class _ResolvedView:
    """Read-only resolver view with prefix semantics like Config.

    Used inside Computed lambdas. Attribute/item access triggers resolution
    of leaves or returns a nested _ResolvedView for groups. Provides `root`
    to jump to the top-level.
    """

    def __init__(self, store, resolve, prefix=""):
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_resolve", resolve)
        object.__setattr__(self, "_prefix", prefix)

    @property
    def root(self):
        return _ResolvedView(self._store, self._resolve, "")

    def _full(self, key):
        return f"{self._prefix}{key}" if self._prefix else str(key)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __getitem__(self, key):
        full = self._full(key)
        if full in self._store:
            return self._resolve(full)
        prefix = full + "."
        if any(k.startswith(prefix) for k in self._store):
            return _ResolvedView(self._store, self._resolve, prefix)
        raise KeyError(key)


class FinalConfig(_BaseView):
    """Final, read-only config with flat store and prefix views."""

    def __init__(self, _store, _prefix=""):
        object.__setattr__(self, "_store", _store)
        object.__setattr__(self, "_prefix", _prefix if _prefix else "")

    # _full, __contains__, to_dict, to_flat_dict, __len__ from _BaseView

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        raise TypeError("FinalConfig is immutable")

    # Mapping interface (read-only)
    def __getitem__(self, key):
        full = self._full(key)
        if full in self._store:
            return self._store[full]
        prefix = full + "."
        if any(k.startswith(prefix) for k in self._store):
            return FinalConfig(self._store, prefix)
        raise KeyError(key)

    def __setitem__(self, key, value):
        raise TypeError("FinalConfig is immutable")

    def __delitem__(self, key):
        raise TypeError("FinalConfig is immutable")

    def __iter__(self):
        return self._iter_child_segments()

    def __repr__(self):
        return f"FinalConfig({self.to_dict()!r})"

    # to_dict and to_flat_dict inherited from _BaseView
