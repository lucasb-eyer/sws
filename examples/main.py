"""Minimal example using the sws package."""

import os
import sys

# Allow running from repository without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sws import Config, lazy


def get_config():
    return Config(
        lr=0.0003,
        model={"width": 128},
        wd=lazy(lambda c: c.lr * 0.1),
    )


if __name__ == "__main__":
    base = get_config()
    # Allow overriding from CLI, e.g.: lr=0.01 model.width=64
    f = base.finalize(sys.argv[1:])
    print("final:", f.to_dict())
