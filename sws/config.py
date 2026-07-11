from collections.abc import Mapping
from contextvars import ContextVar
from copy import copy
import difflib
import json
import os

from .simpleeval import EvalWithCompoundTypes


class FinalizeError(Exception):
    pass


class CycleError(FinalizeError):
    pass


class LazySubtreeError(FinalizeError):
    pass


class MissingKeyError(KeyError, AttributeError):
    """A missing config path that should behave like both key and attr lookup."""
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

    def __iter__(self):
        return self._iter_child_segments()

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
            cur[parts[-1]] = _export_value(v)
        return out


def _flatten(base, value):
    # Yield (full_key, leaf_value) pairs, flattening nested mappings
    if isinstance(value, Mapping):
        for k, v in dict(value).items():
            fk = f"{base}.{k}" if base else k
            yield from _flatten(fk, v)
    else:
        yield base, value


def _export_value(value):
    if isinstance(value, _BaseView):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {k: _export_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_export_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_export_value(v) for v in value)
    if isinstance(value, set):
        return {_export_value(v) for v in value}
    if isinstance(value, frozenset):
        return frozenset(_export_value(v) for v in value)
    return value


class Config(_BaseView):
    """Flattened, dotted-key config with prefix views.

    - Stores all values in a single flat dict of full dotted keys.
    - Accessing a group returns a prefixed view sharing the same store.
    """

    @property
    def _store(self):
        context = self._finalize_context.get()
        return self._base_store if context is None else context[0]

    @property
    def _state(self):
        context = self._finalize_context.get()
        return self._base_state if context is None else context[1]

    @property
    def _phase(self):
        return self._state["phase"]

    @property
    def _cycle(self):
        return self._state["cycle"]

    @property
    def _resolved(self):
        return self._state["resolved"]

    def __init__(self, **kw):
        self._finalize_context = ContextVar("sws_finalize_context", default=None)
        self._base_store = {}
        self._base_state = {"phase": "building", "cycle": [], "resolved": {}}
        self._prefix = ""
        self._assign(None, kw)  # Init from kw's.

    def _with_prefix(self, prefix):
        """A shallow copy (sharing store/finalization state) with new prefix."""
        new = copy(self)
        new._prefix = prefix
        return new

    def _assign_leaf(self, full, value):
        # Assigning a leaf replaces any existing subtree and clears conflicting ancestors.
        prefix = full + "."
        to_del = [k for k in list(self._store) if k.startswith(prefix)]
        for k in to_del:
            del self._store[k]
        if "." in full:
            parts = full.split(".")
            for i in range(1, len(parts)):
                anc = ".".join(parts[:i])
                if anc in self._store:
                    del self._store[anc]
        self._store[full] = value

    def _assign(self, full, value):
        # Assign value at 'full'; if it's a mapping/Config, flatten under that prefix.
        if isinstance(value, _BaseView):
            value = value.to_dict()
        if isinstance(value, Mapping):
            if full in self._store:
                del self._store[full]
            for fk, v in _flatten(full, value):
                self._assign_leaf(fk, v)
        else:
            self._assign_leaf(full, value)

    # Attribute access: Redirect all attribute getting and setting to item
    # getting and setting, except for _attrs, keep those as normal.
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)  # Avoid making views for inexistent.
        try:
            return self[name]
        except KeyError as e:
            raise MissingKeyError(name) from e

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

            # val is callable, so it was a lazy. Resolve it once per finalization,
            # but careful of cycles.
            if full in self._resolved:
                return self._resolved[full]

            # Pop in a finally: if val() raises and the error is caught (e.g. by
            # argv override parsing), a stale entry would make a later resolution
            # of this key look like a cycle.
            self._cycle.append(full)
            try:
                if self._cycle.count(full) > 1:  # Oops, we have a cycle!
                    raise CycleError(f"Cycle detected: {' -> '.join(self._cycle)}")
                resolved = val()
            finally:
                self._cycle.pop()
            self._resolved[full] = resolved
            return resolved
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
    def finalize(self, argv=None, return_unused_argv=False):
        """Resolve a write-only builder into an immutable, fully-evaluated config."""
        if self._prefix:
            raise ValueError("Call `finalize` on the top-level config.")

        context = (
            dict(self._base_store),
            {"phase": "finalize", "cycle": [], "resolved": {}},
        )
        token = self._finalize_context.set(context)

        try:
            return self._finalize_current_store(argv, return_unused_argv)
        finally:
            self._finalize_context.reset(token)

    def _finalize_current_store(self, argv=None, return_unused_argv=False):
        # Apply overrides if provided. Support `key=value` for existing keys
        # (with suffix matching over leaves and group roots),
        # `..suffix=value` to set all matching leaves/group-roots by raw dotted-key suffix,
        # and `key:=value` to create-or-set an exact dotted key
        # (no suffix matching, creates if missing).
        evaluator = EvalWithCompoundTypes(names={"c": self}, functions={"Fn": Fn, "range": range})
        def parse_val(val):
            def _lazy():
                try:
                    return evaluator.eval(val)
                except Exception:
                    return val
            _lazy._sws_argv_override = True
            return _lazy

        def _validate_exact_override_key(raw_key):
            key = raw_key.removeprefix("c.")
            if key.startswith(".") or ".." in key or key.endswith("."):
                msg = f"Invalid exact override key {raw_key!r}. "
                msg += "':=' requires an explicit dotted path with non-empty segments and "
                msg += "does not support wildcard prefixes like '..' or '...'. "
                msg += f"Use '=' for wildcard matching, for example {raw_key}=VALUE."
                raise AttributeError(msg)
            return key

        def _lazy_subtree_hint():
            return (
                "Reusable subtrees should be built by writing into a subtree view during "
                "config construction, for example:\n"
                "    def make_foobar(c, cf):\n"
                "        cf.baz = lambda: c.thingy + 2\n"
                "    make_foobar(c, c.foo.bar)\n"
                "The way you are trying to do it is full of footguns, which is against sws design."
            )

        def _lazy_leaf_ancestor(key_suffix):
            for k, v_existing in sorted(self._store.items(), key=lambda kv: -len(kv[0])):
                if isinstance(v_existing, Fn) or not callable(v_existing):
                    continue
                if getattr(v_existing, "_sws_argv_override", False):
                    continue
                if key_suffix.startswith(k + "."):
                    return k
            return None

        def _raise_lazy_leaf_descendant(raw_key, lazy_key):
            raise LazySubtreeError(
                f"Cannot override {raw_key!r}: {lazy_key!r} is a lazy leaf, "
                "so its children are not known config keys. Lazy values are not "
                "supported for declaring new overridable subtrees.\n"
                + _lazy_subtree_hint()
            )

        unused = []
        for token in list(argv or []):
            if "=" not in token:
                unused.append(token)
                continue

            raw_key, v = token.split("=", 1)
            if raw_key.endswith(":"):
                raw_key = raw_key[:-1]
                key = _validate_exact_override_key(raw_key)
                lazy_key = _lazy_leaf_ancestor(key)
                if lazy_key is not None:
                    _raise_lazy_leaf_descendant(raw_key, lazy_key)
                # Use internal assign to respect overwrite rules and allow mappings
                self._assign(key, parse_val(v))
                continue

            # Find the keys or group-roots which have this suffix. If multiple, provide error.
            suffix = raw_key.removeprefix("c.")
            explicit = raw_key.startswith("c.")
            wildcard = suffix.startswith("..")
            if wildcard:
                suffix = suffix[2:]

            group_roots = None

            def _group_roots():
                nonlocal group_roots
                if group_roots is None:
                    roots = set()
                    for k in self._store:
                        parts = k.split(".")
                        for i in range(1, len(parts)):
                            roots.add(".".join(parts[:i]))
                    group_roots = roots
                return group_roots

            def _raise_unknown(unknown_suffix=None):
                unknown_suffix = suffix if unknown_suffix is None else unknown_suffix
                keys = list(self._store)
                roots = _group_roots()
                if roots:
                    keys.extend(sorted(roots))
                # First, try fuzzy match against full dotted keys
                suggestions = difflib.get_close_matches(unknown_suffix, keys)
                # Also try fuzzy match against last segments (common typo case),
                # then expand those to full keys sharing that last segment.
                num_segs = unknown_suffix.count(".") + 1
                seg_candidates = {".".join(k.split(".")[-num_segs:]) for k in keys}
                seg_matches = difflib.get_close_matches(unknown_suffix, seg_candidates)
                for seg in seg_matches:
                    suggestions.extend(k for k in keys if k.endswith("." + seg) or k == seg)
                # Deduplicate while preserving order
                seen = set()
                suggestions = [s for s in suggestions if not (s in seen or seen.add(s))]
                msg = f"Unknown override key {unknown_suffix!r}"
                if suggestions:
                    msg += "; did you mean:\n" + "\n".join(suggestions)
                raise AttributeError(msg)

            def _segment_suffix_match(candidates, key_suffix):
                return [k for k in candidates if ("." + k).endswith("." + key_suffix)]

            def _raw_suffix_match(candidates, key_suffix):
                return [k for k in candidates if ("." + k).endswith(key_suffix)]

            def _prune_descendant_targets(candidates):
                pruned = []
                for candidate in sorted(set(candidates), key=lambda k: (k.count("."), k)):
                    if any(candidate.startswith(parent + ".") for parent in pruned):
                        continue
                    pruned.append(candidate)
                return pruned

            def _raise_ambiguous(candidates):
                msg = f"Ambiguous override key {suffix!r}; candidates:\n" \
                      + '\n'.join(sorted(candidates))
                if (suffix in self._store or suffix in _group_roots()) and not explicit:
                    msg += (f"\nHint: use 'c.{suffix}=VALUE' to target that exact key, "
                            f"'..{suffix}=VALUE' to target all key with that suffix, and "
                            f"hence '...{suffix}=VALUE' to target all keys with that exact name.")
                raise AttributeError(msg)

            if wildcard:
                if suffix in {"", "."}:
                    raise AttributeError("Invalid wildcard override key '..'; "
                                         "expected '..suffix=VALUE'.")

                targets = _prune_descendant_targets(
                    _raw_suffix_match(self._store, suffix)
                    + _raw_suffix_match(_group_roots(), suffix)
                )
                if not targets:
                    unknown = suffix[1:] if suffix.startswith(".") else suffix
                    _raise_unknown(unknown)

                parsed = parse_val(v)
                for target in targets:
                    self._assign(target, parsed)
            else:
                # Exact match when explicitly prefixed with c.
                if explicit:
                    if suffix in self._store or suffix in _group_roots():
                        target = suffix
                    else:
                        lazy_key = _lazy_leaf_ancestor(suffix)
                        if lazy_key is not None:
                            _raise_lazy_leaf_descendant(raw_key, lazy_key)
                        _raise_unknown()
                else:
                    leaf_matches = _segment_suffix_match(self._store, suffix)
                    group_matches = _segment_suffix_match(_group_roots(), suffix)
                    matches = sorted(set(leaf_matches) | set(group_matches))
                    if len(matches) > 1:
                        _raise_ambiguous(matches)
                    if len(matches) == 1:
                        target = matches[0]
                    else:
                        lazy_key = _lazy_leaf_ancestor(suffix)
                        if lazy_key is not None:
                            _raise_lazy_leaf_descendant(raw_key, lazy_key)
                        _raise_unknown()

                self._assign(target, parse_val(v))

        # Now go over all items and resolve those that were lazy.
        resolved_store = {k: self[k] for k in self._store}
        finalized_store = {}

        def _is_argv_override_key(key):
            return getattr(self._store.get(key), "_sws_argv_override", False)

        def _flatten_final_value(source_key, dest_key, value, cycle):
            if _is_argv_override_key(source_key) and isinstance(value, Mapping):
                for child_dest, child_val in _flatten(dest_key, value):
                    _flatten_final_value(source_key, child_dest, child_val, cycle)
                return

            if isinstance(value, Config):
                if value._store is not self._store:
                    raise LazySubtreeError(
                        f"Lazy field {source_key!r} returned a new sws.Config. "
                        "Lazy values may alias an existing subtree, but they cannot "
                        "declare new overridable subtrees.\n"
                        + _lazy_subtree_hint()
                    )

                if source_key in cycle:
                    raise CycleError(
                        "Cycle detected while flattening config views: "
                        + " -> ".join([*cycle, source_key])
                    )

                src_prefix = value._prefix
                plen = len(src_prefix)
                next_cycle = [*cycle, source_key]
                for child_key, child_val in resolved_store.items():
                    if not child_key.startswith(src_prefix):
                        continue
                    if child_key == source_key or child_key.startswith(source_key + "."):
                        continue

                    rel = child_key[plen:]
                    child_dest = f"{dest_key}.{rel}" if rel else dest_key
                    _flatten_final_value(child_key, child_dest, child_val, next_cycle)
                return

            if isinstance(value, FinalConfig):
                for rel, child_val in value.to_flat_dict().items():
                    child_dest = f"{dest_key}.{rel}" if rel else dest_key
                    finalized_store[child_dest] = child_val
                return

            finalized_store[dest_key] = value

        for key, value in resolved_store.items():
            _flatten_final_value(key, key, value, [])

        final = FinalConfig(
            _store=finalized_store,
            _prefix=self._prefix
        )

        if return_unused_argv:
            return final, unused
        else:
            return final


def json_invalid_to_string(obj):
    return f"<non-jsonable object of type {type(obj).__name__}; repr: {repr(obj)}>"


class FinalConfig(_BaseView):
    """Final, read-only config with flat store and prefix views."""

    def __init__(self, _store, _prefix="", _subtree_prefixes=None):
        object.__setattr__(self, "_store", _store)
        object.__setattr__(self, "_prefix", _prefix if _prefix else "")
        object.__setattr__(self, "_subtree_prefixes", _subtree_prefixes)

    # _full, to_dict, to_flat_dict, __len__ from _BaseView

    @staticmethod
    def _build_subtree_prefixes(store):
        # Store dotted prefixes with trailing dots so subtree membership is O(1).
        prefixes = set()
        for key in store:
            idx = key.find(".")
            while idx != -1:
                prefixes.add(key[: idx + 1])
                idx = key.find(".", idx + 1)
        return frozenset(prefixes)

    def _has_subtree(self, full):
        prefixes = self._subtree_prefixes
        if prefixes is None:
            prefixes = self._build_subtree_prefixes(self._store)
            object.__setattr__(self, "_subtree_prefixes", prefixes)
        return (full + ".") in prefixes

    def __contains__(self, key):
        full = self._full(key)
        return (full in self._store) or self._has_subtree(full)

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
        if self._has_subtree(full):
            return FinalConfig(self._store, prefix, self._subtree_prefixes)
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

    def to_json(self, default=json_invalid_to_string, **json_kwargs):
        return json.dumps(self.to_dict(), default=default, **json_kwargs)

    def to_flat_json(self, default=json_invalid_to_string, **json_kwargs):
        return json.dumps(self.to_flat_dict(), default=default, **json_kwargs)


def from_json(data):
    return FinalConfig(_store=dict(_flatten("", json.loads(data))), _prefix="")


def from_flat_json(data):
    return FinalConfig(_store=json.loads(data), _prefix="")
