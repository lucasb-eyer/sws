import sys
from sws import run


def get_config():
    from sws import Config
    c = Config(lr=0.1)
    return c


def main(cfg, rest):
    # Print values in a simple, parseable way
    print(f"lr={cfg.lr}")
    print("unused=" + ",".join(rest))


if __name__ == "__main__":
    run(main, forward_extras=True)
