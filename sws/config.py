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

    # Attribute access: Redirect all attribute getting and setting to item
    # getting and setting, except for _attrs, keep those as normal.
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)  # Avoid making views for inexistent.
        return self[name]

    def __setattr__(self, name, value):
        if name.startswith("_"):
            return object.__setattr__(self, name, value)
        self[name] = value

    # MutableMapping core
    def __getitem__(self, key):
        full = self._full(key)
        if full in self._store:
            raise TypeError("Config is write-only; use lazy(...) or finalize() to read")
        return Config(_store=self._store, _prefix=full + ".")

    def __setitem__(self, key, value):
        self._assign(self._full(key), value)

    def __delitem__(self, key):
        full = self._full(key)
        to_del = [k for k in list(self._store) if k == full or k.startswith(full + ".")]
        if not to_del:
            raise KeyError(key)
        for k in to_del:
            del self._store[k]

    # Finalization to an immutable, resolved config
    def finalize(self, argv=None):
        """Resolve a write-only builder into an immutable, fully-evaluated config.

        High-level algorithm:
        - Work on a snapshot of the current flat store (so overrides don't mutate
          the original builder) and resolve only leaf keys that exist in the flat
          store. All nested mappings were already flattened on assignment.
        - Provide a resolver function `_resolve(full)` that lazily resolves a leaf
          with memoization and cycle detection. If a value is `Computed`, call it
          with a read-only `_ResolvedView` rooted at the leaf's parent so computed
          functions can reference siblings (via the view) or the full tree (via
          `view.root`). Any `Computed` results inside containers are recursively
          resolved.
        - Freeze containers: lists/tuples become tuples, sets become frozensets;
          mappings inside containers are rejected because mappings should have been
          flattened on input. This makes the final config structurally immutable.
        - Apply CLI-style overrides (`argv`) left-to-right onto the snapshot. Only
          tokens of the form `key=value` are considered (others are ignored) and
          values are evaluated as a Python expression using a `_ResolvedView` bound
          to name `c` so expressions can reference already-resolved values
          (e.g., `c.model.width`). Between each override, the resolver memo is
          cleared to reflect the new state.
        - Finally, iterate over all leaf keys and resolve them through `_resolve`,
          returning a `FinalConfig` holding the resolved flat dict and preserving
          the current prefix for any sub-views created before finalize.
        """
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

        # Apply overrides if provided: only `key=value` tokens are processed.
        cfg = Config(_store=store)
        for token in list(argv or []):
            if "=" not in token:
                continue
            key, val = token.split("=", 1)
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

        # Only resolve leaves (keys present in the flat store)
        return FinalConfig(_store={k: _resolve(k) for k in store}, _prefix=self._prefix)


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

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        raise TypeError("FinalConfig is immutable")

    def __delitem__(self, key):
        raise TypeError("FinalConfig is immutable")

    def __iter__(self):
        return self._iter_child_segments()

    def __repr__(self):
        return f"FinalConfig({self.to_dict()!r})"

    def __str__(self):
        # Pretty, human-readable flat view: show each full dotted key on a line,
        # but bold only the last segment to visually hint the tree structure.
        def _fmt_val(v):
            if isinstance(v, float):
                return f"{v:.8g}"
            if isinstance(v, tuple):
                inner = ", ".join(_fmt_val(x) for x in v)
                if len(v) == 1:
                    inner += ","
                return f"({inner})"
            if isinstance(v, frozenset):
                items = sorted((_fmt_val(x) for x in v), key=str)
                return "{" + ", ".join(items) + "}"
            return repr(v)

        def _bold(s: str) -> str:
            return "\x1b[1m" + s + "\x1b[0m"

        def _dim(s: str) -> str:
            return "\x1b[2m" + s + "\x1b[0m"

        def _blue(s: str) -> str:
            return "\x1b[34m" + s + "\x1b[0m"

        flat = self.to_flat_dict()
        if not flat:
            return "{}"
        lines = []
        for full_key in sorted(flat):
            parts = full_key.split(".")
            if len(parts) == 1:
                disp = _bold(parts[0])
            else:
                disp = _dim(".".join(parts[:-1]) + ".") + _bold(parts[-1])
            lines.append(f"{disp}: {_blue(_fmt_val(flat[full_key]))}")
        return "\n".join(lines)
