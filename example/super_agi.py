"""Run with this config by doing python -m example.main --config super_agi.py"""
from sws import Config, lazy


def get_config():
    c = Config()
    c.lr = 0.0003
    c.wd = lazy(lambda c: c.lr * 0.1)
    c.model.depth = 64
    c.model.heads = 96
    c.model.width = lazy(lambda c: c.heads * 128)
    c.model.init_seed = 42
    return c
