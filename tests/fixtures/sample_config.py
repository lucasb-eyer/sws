from sws import Config


def get_config():
    c = Config()
    c.lr = 0.1
    c.model.width = 128
    c.model.depth = 4
    c.wd = lambda: c.lr * 0.1
    return c
