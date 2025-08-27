"""Run with this config by doing:

    python -m example.main --config example/super_agi.py
"""
from sws import Config


def get_config():
    c = Config()
    c.lr = 0.0003
    c.wd = lambda: c.lr * 0.1
    c.model.depth = 64
    c.model.heads = 96
    # Reference nested fields via `c.model.*` inside lazies
    c.model.width = lambda: c.model.heads * 128
    c.model.init_seed = 42
    return c
