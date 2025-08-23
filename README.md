# sws

Minimal, predictable, footgun-free configuration for deep learning experiments.

The remainder of this readme follows the CODE THEN EXPLAIN layout.
Install instructions at the end.

## Basics

```python
from sws import Config, lazy

# Create the config and populate the fields with defaults
c = Config()
c.lr = 3e-4
c.wd = c.lr * 0.1  # ERROR, c is write-only. Instead, do:
c.wd = lazy(lambda c: c.lr * 0.1)

# Alternative convenience for short configs:
c = Config(lr=3e-4, wd=lazy(lambda c: c.lr * 0.1))

# Finalizing resolves all fields to plain values, and integrates CLI args:
c = c.finalize(argv=sys.argv[1:])
assert c.lr == 3e-4 and c.wd == 3e-5

train_agi(lr=c.lr, wd=c.wd)
```

Sws clearly separates two phases: config creation, and config use.
At creation time, you build a (possibly nested) `Config` object, with the option to use `lazy(...)` to make some field's value depend on the values of other fields.
Then, you `finalize()` the config, which resolves all fields (including `lazy` ones) into plain concrete values.
If desired, `finalize()` also applies overrides from the commandline.

To avoid subtle bugs common in many config libraries I've used before, at creation time, the `Config` object is *write-only*; this forces use of `lazy` references that get correctly resolved later.
At use-time (after `finalize()`), on the other hand, the `FinalConfig` object is read-only, which again avoids subtle bugs in complex code.

## Nesting

TODO: Continue the doc rewrite from here!

Nested writes and reading after finalize
```python
c = sws.Config()
c.model = {"width": 128}
c.model.depth = 4

f = c.finalize()
assert f.model.width == 128 and f.model.depth == 4
```

CLI-style overrides (left-to-right) and expression access to `c`
```python
import sys

base = sws.Config(lr=1.0, model={"width": 128}, foo=0, bar=0)
f = base.finalize(sys.argv[1:])  # e.g.: lr=3 foo=c.lr bar=c.model.width

# Or pass an explicit list:
f = base.finalize(["lr", "3", "foo", "c.lr", "bar", "c.model.width"])  # foo=1.0, then lr=3, then bar=3
assert f.foo == 1.0 and f.lr == 3 and f.bar == 3
```

Containers and lazy inside containers
```python
c = sws.Config(
    a=2,
    t=(1, lazy(lambda c: c.a + 1)),
    s={1, lazy(lambda c: c.a)},
)
f = c.finalize()
assert f.t == (1, 3) and f.s == frozenset({1, 2})
```

Cycles and errors
```python
import pytest

c = sws.Config(a=lazy(lambda c: c.b), b=lazy(lambda c: c.a))
wic pytest.raises(sws.CycleError):
    c.finalize()
```

## Install
```bash
pip install sws-config
```

## Test
```bash
python -m pytest
```

## Authors / License
- Authors: see `pyproject.toml`.
- License: MIT © Lucas Beyer. See LICENSE for details.
