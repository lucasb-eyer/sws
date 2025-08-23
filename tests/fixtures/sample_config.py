from sws import Config, lazy


def get_config():
    c = Config()
    c.lr = 0.1
    c.model.width = 128
    c.model.depth = 4
    c.wd = lazy(lambda c: c.lr * 0.1)
    return c

