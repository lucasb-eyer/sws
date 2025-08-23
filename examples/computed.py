"""Examples for computed fields and finalize.

Run: python examples/computed.py
"""
import os
import sys

# Allow running from repository without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sws import Config, lazy


def ex_basic():
    cfg = Config(lr=0.001, wd=lazy(lambda c: c.lr * 0.5))
    f = cfg.finalize()
    print("basic:", dict(f.to_flat_dict()))


def ex_override_parse():
    base = Config(lr=0.001, wd=lazy(lambda c: c.lr * 0.5))
    f = base.finalize(["lr", "10"])  # override takes precedence
    print("override-parse:", dict(f.to_flat_dict()))


def ex_local_and_root():
    cfg = Config(
        a={
            "x": 2,
            "y": lazy(lambda c: c.x * 3),  # sibling via local view
        },
        b={
            "z": lazy(lambda c: c.root.a.x + 1),  # cross-branch via root
        },
    )
    f = cfg.finalize()
    print("local-root:", dict(f.to_flat_dict()))


def ex_containers():
    cfg = Config(vals=[1, lazy(lambda c: c.a.x)])
    cfg.a = {"x": 7}
    f = cfg.finalize()
    print("containers:", dict(f.to_flat_dict()))


def ex_cycle():
    cfg = Config(a=lazy(lambda c: c.b), b=lazy(lambda c: c.a))
    try:
        cfg.finalize()
    except Exception as e:
        print("cycle:", str(e))


if __name__ == "__main__":
    ex_basic()
    ex_override_parse()
    ex_local_and_root()
    ex_containers()
    ex_cycle()
