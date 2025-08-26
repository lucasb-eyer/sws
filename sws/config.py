from collections.abc import Mapping
from .simpleeval import EvalWithCompoundTypes


class FinalizeError(Exception):
    pass


class CycleError(FinalizeError):
    pass


class Fn:
    """Wrapper to store a callable as a plain value."""
    def __init__(self, fn):
        self.fn = fn


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

    def __init__(self, **kw):
        self._store = {}
        self._prefix = ""
        self._phase = "building"
        self._cycle = []
        self._assign(None, kw)  # Init from kw's.

    def _with_prefix(self, prefix):
        """A shallow copy (sharing store/cycle) with new prefix."""
        from copy import copy
        new = copy(self)
        new._prefix = prefix
        return new

    def _assign_leaf(self, full, value):
        # Forbid assigning a leaf where a group exists
        if any(k.startswith(full + ".") for k in self._store):
            raise ValueError(f"Cannot set leaf '{full}': group with same name exists")
        self._store[full] = value

    def _assign(self, full, value):
        # Assign value at 'full'; if it's a mapping/Config, flatten under that prefix.
        if isinstance(value, _BaseView):
            value = value.to_dict()
        if isinstance(value, Mapping):
            if full in self._store:  # Forbid creating a group where a leaf exists
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

        if self._phase == "building":
            if full in self._store:
                raise TypeError("Config is write-only; use finalize() to read, "
                                "or assign a callable for lazy evaluation.")
            return self._with_prefix(full + ".")
        elif self._phase == "finalize":
            if any(k.startswith(full + ".") for k in self._store):
                return self._with_prefix(full + ".")
            val = self._store[full]  # This raising KeyError is part of the API.
            if isinstance(val, Fn):
                return val.fn
            if not callable(val):
                return val

            # val is callable, so it was a lazy. Resolve it, but careful of cycles.
            self._cycle.append(full)
            if self._cycle.count(full) > 1:  # Oops, we have a cycle!
                raise CycleError(f"Cycle detected: {' -> '.join(self._cycle)}")
            v = val()
            self._cycle.pop()
            return v
        else:
            assert False, f"Internal bug: {self._phase}"

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
        """Resolve a write-only builder into an immutable, fully-evaluated config."""
        assert not self._prefix, "Call `finalize` on the top-level config."
        self._phase = "finalize"

        # Apply overrides if provided: only `key=value` tokens are processed.
        evaluator = EvalWithCompoundTypes(names={"c": self, "Fn": Fn})
        for token in list(argv or []):
            if "=" not in token:
                continue
            suffix, val = token.split("=", 1)

            # Find the keys which have this suffix. If multiple, provide error.
            suffix = suffix.removeprefix("c.")
            matches = [k for k in self._store if ("." + k).endswith("." + suffix)]
            if not matches:
                raise AttributeError(suffix)
            if len(matches) > 1:  # Ambiguous suffix; help user disambiguate
                raise AttributeError(
                    f"Ambiguous override key {suffix!r}; candidates:\n"
                    + '\n'.join(sorted(matches)))
            key = matches[0]

            try:
                self._store[key] = evaluator.eval(val)
            except Exception:  # Assume it's a non-quoted string
                self._store[key] = val

        # Now go over all items and resolve those that were lazy.
        finalized_store = {k: self[k] for k in self._store}

        self._phase = "building"
        return FinalConfig(
            _store=finalized_store,
            _prefix=self._prefix
        )


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
