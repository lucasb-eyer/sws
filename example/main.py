"""Minimal example using the sws package.

To run this, either install sws, or do:

    python -m example.main
"""

from sws import Config, run


def get_config():
    c = Config()
    c.lr = 0.001
    # Lazies are zero-arg callables that can reference `c` safely.
    c.wd = lambda: c.lr * 0.1
    c.model.depth = 4
    c.model.width = 256
    c.model.heads = lambda: 4 if c.model.width > 128 else 1
    return c


def make_model(depth, width, heads, **secret):
    print(f"ModelToAGI(d={depth}, wh={width // heads}, nh={heads})")
    if secret:
        print(f"/!\\ The model is using secret sauce {secret}")


def main(c):
    print("Running AGI training with the full config:")
    print(c)
    make_model(**c.model.to_dict())


if __name__ == "__main__":
    run(main)

    # Alternate, a bit more explicit and a bit less flexible:
    # import sys
    # main(get_config().finalize(sys.argv[1:]))
    # NOTE: This doesn't allow to change config file with --config.
