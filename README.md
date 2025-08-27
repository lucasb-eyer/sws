# sws

[![Tests](https://github.com/lucasb-eyer/sws/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/lucasb-eyer/sws/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/lucasb-eyer/sws/branch/master/graph/badge.svg)](https://codecov.io/gh/lucasb-eyer/sws)

Minimal, predictable, footgun-free configuration for deep learning experiments.
The most similar existing ones are
[OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html) and
[ConfigDict](https://github.com/google/ml_collections) -
if you are happy with them, you probably don't need this.
If you want some lore, have a look at the end.

The remainder of this readme follows the CODE THEN EXPLAIN layout.
The `example/` folder contains a nearly real-world example of structuring a project.
Install instructions at the end.

## Basics

```python
from sws import Config

# Create the config and populate the fields with defaults
c = Config()
c.lr = 3e-4

# Alternative shorthand handy for very small configs:
c = Config(lr=3e-4)

# How to make a field depend on others?
c.wd = c.lr * 0.1  # ERROR: c is write-only.
# Instead, use a lambda to make the value "lazy"
c.wd = lambda: c.lr * 0.1

# Finalizing resolves all fields to plain values, and integrates CLI args:
c = c.finalize(argv=sys.argv[1:])
assert c.lr == 3e-4 and c.wd == 3e-5

train_agi(lr=c.lr, wd=c.wd)
```

`sws` clearly separates two phases: config creation, and config use.
At creation time, you build a (possibly nested) `Config` object.
To avoid subtle bugs common in many config libraries I've used before, at
creation time, the `Config` object is *write-only*; you cannot read its values.
Once you finished building it up, a call to `c.finalize()` turns it into a
read-only `FinalConfig` object that contains "final" values for all fields.

This *finalization* step can also integrate overrides from, for example,
commandline arguments; more on that a little later.

If you want to make one field's value depend on another field's value, you can
do so by wrapping the value in a `lambda`, which computes the derived value.
This lambda will be called during finalization, where concrete config values
can be accessed. In this way, in the example above, the `wd` setting will use
the correct value of `c.lr` even when it is overriden by commandline arguments
during `finalize`. This works transitively, just as you'd expect it to.

Since callable values receive this special treatment, if you want to actually
set a config field's value to an actual function, that needs to be wrapped by
`sws.Fn`:

```python
from sws import Fn

# If you want to store a callable as a value (not execute it at finalize), wrap it:
c.log_fn = Fn(lambda s: print(s))
c = c.finalize()

# Five moments later...
c.log_fn("After finalization, the config field is just this plain function")
```

## Nesting

Of course any respectable config library allows nested structures:

```python
from sws import Config

# Create the config and populate the fields with defaults
c = Config()
c.lr = 3e-4
c.model.depth = 4  # No need to create parents first.

# In a nested field, lazy and `c` work just as you'd expect them to:
c.model.width = lambda: c.model.depth * 64
c.model.emb_lr = lambda: c.lr * 10 / c.model.width

c = c.finalize()

# Pass model settings as kwargs, for example:
m = MyAGIModel(**c.model.to_dict())
train_agi(m, c.lr)
```

The reason we need `to_dict()` above is that `FinalConfig` implements as few methods as possible,
to leave as many names as possible free to be used for configs. For instance, `keys`, `values`, and
`items` are not implemented so that you can use them as config names.
This also means, that it doesn't implement the `Mapping` protocol and can't be `**`'ed.
So, just call `to_dict`, it's fine.

You don't really need to know this, but internally, the full config is stored as a flat dict
(`"model.emb_lr"` is a key), and subfields are just prefix-views into that dict.

## Commandline overrides

The `finalize()` method allows you to pass a list of `argv` strings to it that serve as overrides:

```python
from sws import Config

c = Config(lr=1.0, model={"width": 128, "depth": 4})
c = c.finalize(["c.model.width=512", "c.model.depth=2+2"])

# However, we're lazy. The shortest unique segment suffix works:
c = c.finalize(["width=512", "depth=2+2"])

# In real life, you'd probably pass sys.argv[1:] instead.
```

Only the syntax `a=b` is supported, any argument without `=` is ignored.
This is to reduce ambiguity and allow catching typos.

The values of the overrides are parsed as Python expressions using the `simpleeval`
library. This makes a lot of Python code just work, for example you can write
`model.vocab=[ord(c) for c in "hello"]` and it'll work. You can also access the
current config using the name `c`, so something like `'c.model.width=3 * c.model.depth'`
works. Note that I quoted the whole thing, for two reasons: (1) to stop my shell
from interpreting `*` as wildcard, and (2) because I used spaces.

For convenience, the keyname can be shortened to the shortest unique suffix
across the _whole_ config (i.e. all nesting levels).
For example, `model.head.lr` can be shortened to `head.lr` or `lr` if unambiguous.

## `sws.run` and suggested code structure

The `train.py` file could look something like this:

```python
import sws

# ...lots of code...

def train(c):
    # Do some AGI things, but be careful please.
    # `c` is a FinalConfig here, i.e. it's been finalized.

if __name__ == "__main__":
    sws.run(train)
```

This seemingly innocuous code does a lot, thanks to judiciously chosen default arguments.
The full call would be `sws.run(train, argv=sys.argv[1:], config_flag="--config", default_func="get_config")`.

First, it looks for a commandline argument `--config filename.py` (or `--config=filename.py`).

It then loads said file, and runs the `get_config` function defined therein,
which should return a fully populated `sws.Config` object. Note that it's plain
python code, so it may import things, have a lot of logic, feel free to do as much
or as little as you want.

Finally, it `finalize`s the config with the remaining commandline arguments,
and calls the specified function (in this example, `train`) with the `FinalConfig`.

Here's what a config file might look like, let's call it `vit_i1k.py`:

```python
from sws import Config

def get_config():
    c = Config()
    c.lr = 3e-4
    c.wd = lambda: c.lr * 0.1
    c.model.name = "vit"
    c.model.depth = 8
    c.model.width = 512
    c.model.patch_size = (16, 16)
    c.dataset = "imagenet_2012"
    c.batch = 4096
    return c
```

Then, you would run training as `python -m train --config vit_i1k.py batch=1024`.
In a real codebase, you'd have quite a few config files, maybe in some structured
`config/` folder with sub-folders per project, user, topic, ...

There's three more things `sws.run` does for convenience:
- If no `--config` is passed, it looks for the `get_config` function in the file
  which called it. This is very convenient for quick small scripts.
- If you use `run(fn, forward_extras=True)`, then all unused commandline arguments,
  i.e. all those without a `=`, are passed in a list as the second argument to `fn`.
  This can be used to do further custom processing unrelated to `sws`.
- For extra flexibility, you can actually specify which function should be called.
  The syntax is `--config file.py:function_name`, it's just that the function name
  defaults to `get_config`. This way, you can have multiple slight variants in the
  same file, for example.

See the `example/` folder of this repo for a semi-realistic example, including
a sweep to run sweeps.

## Examples

Quick runnable example (mirrors `example/`):

```python
from sws import Config, run

def get_config():
    c = Config()
    c.lr = 0.001
    c.wd = lambda: c.lr * 0.1
    c.model.depth = 4
    c.model.width = 256
    c.model.heads = lambda: 4 if c.model.width > 128 else 1
    return c

def main(c):
    print("Training with config:\n" + str(c))
    # Your training code here...

if __name__ == "__main__":
    run(main)
    # Or, explicitly: main(get_config().finalize(sys.argv[1:]))
```

Run a different config file and override values from CLI:

```bash
python -m example.main --config example/super_agi.py model.depth=32 lr=3e-4
```

See `example/sweep.fish` for a trivial sweep over a few values.

## Some more misc notes

- The `FinalConfig` has a nice pretty printer when cast to string or printed.
- When a dict is assigned to a `Config` field, it's turned into a `Config`.
- You cannot set a group to a value or vice-versa, i.e. no `c.model = "vit"`
  followed by `c.model.depth = 4` or vice-versa.
- Cycles in computed callables are detected and raise an exception at `finalize`.

# Installing
```bash
pip install sws-config
```

# Testing
```bash
python -m pytest
```

# TODOs

- When passing commandline args, using lazy/lambda makes no more sense.
  So we should lift the requirement for `Fn`-wrapping of callables here.
  `'log_fn=Fn(lambda s: print(f"Log: {s}"))'`.
- finalization no more converts collections into tuples, sets into frozensets,
  and expands dicts. If a lazy field returns a dict, it's just a dict.
  Consider if this is good or bad, though it does sound obscure.

Probably overkill:
- Auto-generate a commandline --help?
- Auto-generate a terminal UI to browse/change config values on `finalize()` could be fun.
- Rewrite the finalization algorithm. It currently is obviously vibe-coded shit;
  it's correct, but way over-complicated and could be much simpler.

# Lore

You obviously wonder "Why yet another config library, ffs?!" - and you're right.
There are many, but there's none that fully pleases me. So [I gave in](https://x.com/giffmana/status/1953200176526471637).

I've heavily used, and hence been influenced by, many config systems in the past.
Most notably [`ml_collections.ConfigDict`](https://github.com/google/ml_collections)
and [`chz`](https://github.com/openai/chz), both of which I generally liked,
but both had quite some pitfalls after serious use, which I try to avoid here.
Notable examples which I used but _did not_ like are [`gin`](https://github.com/google/gin-config),
[yaml](https://en.wikipedia.org/wiki/YAML) / [Hydra](https://hydra.cc/docs/intro/),
[`kauldron.konfig`](https://kauldron.readthedocs.io/en/latest/konfig_philosophy.html);
they are too heavy, unpythonic, and magic; there be footguns.
[fiddle](https://github.com/google/fiddle) requires your config to import everything,
which I don't like.
I refuse to built around types in Python, like pydantic, tyro, dataclasses, ..., so not even linking them.
Finally, I haven't used, but thoroughly read [Pydra](https://github.com/jordan-benjamin/pydra)
and [Cue](https://cuelang.org/docs/tour/), which together inspired the two-step
approach with finalization.

Why is it called `sws`? It's a nod to OpenAI's `chz` config library,
and the author being a very fond resident of Switzerland.
