# Review of `sws` (Claude Fable, 2026-07-06)

**Overall**: The core design is genuinely good — the write-only builder / finalize split, the flat-dict store with prefix views, and lazy lambdas give a small, coherent mental model, and the code is compact and readable. All 82 tests pass on both venvs (3.14.6 with GIL, 3.14.6 free-threaded). But several confirmed bugs, a few of which directly contradict the README's "no footguns" contract. Everything below was verified with runnable repros, not just read off the code.

## Bugs (confirmed by repro)

### 1. [fixed] `finalize(argv)` permanently mutates the builder

Overrides are written into `self._store` (config.py:295, 385, 410), so the builder is polluted forever:

```python
c = Config(lr=0.1)
c.finalize(["lr=10"])
c.finalize().lr   # == 10, not 0.1!
```

`test_simple` only passes because each later call re-overrides the same key. This breaks the natural mental model that finalize is a pure function of (builder, argv) — e.g. any "finalize once for a dry-run, then finalize for real" or sweep-in-process pattern silently reuses stale overrides. Fix: apply overrides to a copy of the store (or snapshot/restore).

### 2. [fixed] No `try/finally` in `finalize` → corrupted state after any failure

`_phase` is only reset at the end (config.py:457) and `_cycle` is never cleared:

- After a failed finalize (unknown key, cycle, a lazy raising), the builder is stuck in phase `"finalize"` — the write-only protection is silently gone, and reads now resolve lazies mid-construction.
- `_cycle` entries aren't popped when a lazy raises (config.py:207-211), so a retry after fixing the config raises a *spurious* `CycleError: a -> b -> a -> a` even though the cycle was fixed.

Both repro'd. Fix: `try/finally` around finalize resetting phase, and make cycle state local to the finalize call (also helps item 9).

### 3. [fixed] `:=` is checked before `=` in tokens (config.py:288)

Any `=`-override whose *value* contains `:=` gets misparsed and silently creates a garbage key:

```python
Config(msg="hello").finalize(['msg=use := for exact'])
# store: {'msg': 'hello', 'msg=use ': ' for exact'}   — silent!
```

Fix: split on `=` first and treat a key ending in `:` as the exact-create form.

### 4. Leaf/group ambiguity is not detected — README contract violation

The README promises "In the case of ambiguity, sws errs on the cautious side and errors out", but leaf matches are consulted first and group roots only if *zero* leaves match (config.py:393-404):

```python
c.a.size = 1      # leaf ...size
c.b.size.w = 2    # group ...size
c.finalize(["size=9"])   # silently sets a.size, no ambiguity error
```

Same for a top-level leaf `size` vs group `model.size`. A user targeting the group silently mutates the leaf. The two namespaces should be merged before the ambiguity check (with an explicit tie-break rule only for a leaf and *its own* group root, which can't coexist anyway).

### 5. README overclaims the "fresh Config" detection

"sws detects this pattern and gives an error message hinting to the blessed way" is only true for the *lazy* variant (`c.x = lambda: make_sub()`). Direct assignment `c.data.tok = make_sub(...)` is flattened via `to_dict()` (config.py:163-164), copying the internal lambdas whose closures point at the foreign, never-finalized Config — at finalize you get a bare `TypeError: Config is write-only; use finalize() to read...` with no key name and no hint. Very confusing, since the user never read anything. Detectable: when `_assign` flattens a foreign `Config` (different `_store`) containing callables, raise `LazySubtreeError` with the same hint.

### 6. `--config C:\path\to\cfg.py` breaks on Windows

`__init__.py:53-54` does `cfg_path.split(":", 1)`, which eats drive letters. Use `rsplit(":", 1)` and only treat the tail as a function name if it's a valid identifier / the head exists as a file.

## Design footguns worth deciding on

### 7. Failed expressions silently become strings

`parse_val` catches *all* exceptions and falls back to the raw string (config.py:240-246). So `lr=1/0` → the string `"1/0"`, and a typo `wd='c.lrr * 0.1'` → the string `"c.lrr * 0.1"`, crashing far away at use time (or worse, not crashing). This exists so `name=bar` works unquoted, but it converts every expression error into silent data corruption — the exact class of footgun the library exists to prevent. Suggestion: only fall back to string when the token *looks like* a bareword (e.g. matches `[A-Za-z_][\w./-]*` or fails to `ast.parse`); if it parses as a real expression but *evaluation* fails, raise. (The `name:=f'hi-{c.xid}'` missing-key-keeps-string test would need rethinking, but that behavior is itself surprising.)

### 8. Assigning a builder view copies lambdas with stale closures

`c.m2 = c.m1` "works" but the copied lambdas still reference `c.m1.*`:

```python
c.m1.width = 128
c.m1.dim = lambda: c.m1.width // 2
c.m2 = c.m1
c.finalize(["m2.width=64"]).m2.dim   # 64 (follows m1's 128//2), not 32
```

Relatedly, `c.a = c.b` where `b` doesn't exist is a **silent no-op** (`test_bug1` enshrines this — arguably it should raise; an empty-view assignment is almost certainly a typo). Since lazy-returned Configs are already rejected, consider rejecting (or deep-retargeting) view assignment too.

### 9. No memoization → exponential lazy re-evaluation

Each access re-runs the lambda, so diamond dependencies blow up: a chain of 18 lazies each reading its predecessor twice ran the lambdas **262,125 times** (repro'd, ~0.7s; 25 deep would be minutes). Also means side-effecting lazies fire many times. Memoizing resolved values within a single finalize (a per-call cache dict) fixes both and also makes cycle-tracking state local (item 2).

### 10. Not thread-safe, and it shows on free-threaded Python

Concurrent `finalize` calls on the same builder share `_state` (`phase`, `cycle`) and mutate `_store`. On the GIL build an 8-thread probe happened to pass; on 3.14t it reliably produces spurious `CycleError: lr -> lr`. Fixing items 1 and 2/9 (finalize operates on copies with call-local state) makes finalize effectively read-only on shared state, which resolves this for the realistic "share a builder, finalize per-worker" case. Worth a sentence in the README either way.

### 11. Error/shape inconsistencies (smaller)

- A lazy that raises gives **no key context** (`TypeError: can't multiply sequence...` — which of your 200 fields?). Wrap the *top-level* resolution loop (config.py:413) with `raise FinalizeError(f"while resolving {key!r}") from e` — only the outer loop, so `getattr(c, "y", default)`'s KeyError semantics inside lambdas stay intact.
- Unknown/ambiguous override keys raise `AttributeError`, which is a strange type for CLI parse errors and easy to catch accidentally. A dedicated `OverrideError(FinalizeError)` would be a kinder public API.
- `Fn` is only unwrapped when it's the direct leaf value: `c.callbacks = [Fn(f), Fn(g)]` finalizes to a list of `Fn` wrapper objects, not functions.
- CLI `model={'width': 64}` on an existing *group* replaces it with a plain-dict **leaf**, so `f.model.width` becomes `AttributeError` while `f.model["width"]` works — different shape than the same assignment in code, which creates a subtree.
- `finalize` on a subview is guarded by `assert` (config.py:230) — vanishes under `python -O`; make it a real exception.
- `c.to_dict()` / `in` work during building — a documented-by-test read loophole in the "write-only" story; fine, but maybe say so in the README.

### 12. `sws.run` notes

- The no-`--config` fallback re-executes the entire caller file via `run_path` (`__init__.py:60-64`). The `__main__` guard prevents recursion, but all top-level side effects run twice. Worth one README sentence.
- `inspect.stack()[1]` breaks if `run` is called through any wrapper function.
- A trailing bare `--config` (no value) is silently treated as an extra token rather than an error.

## Minor / docs

- pyproject classifiers stop at 3.13; it works on 3.14 and 3.14t (verified), so add them (and maybe a free-threading trove classifier once item 10 is addressed).
- README: "to target all key with that suffix" typo (also in the `_raise_ambiguous` hint text, config.py:366); the fact that `c.`-references in override expressions see *final* (post-all-overrides) values is a nice order-independence property that's tested (`test_override_c_reference_tracks_late_assignments`) but never stated in the README — it deserves a sentence, since "current config" reads as left-to-right.
- Vendored simpleeval looks like a clean upstream copy (2013-2024 header), correctly omitted from coverage. Using it on your own argv is not a security concern.

## Priority order

2 (state corruption — small `try/finally` fix), 1 (builder mutation), 3 (`:=` parsing), 4 (ambiguity contract), 7 (silent string fallback — biggest real-world footgun, but a design call), then the rest. Items 1+2+9 together also fix 10 almost for free.
