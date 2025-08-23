import sws


def test_version_is_string():
    assert isinstance(sws.__version__, str)
    assert sws.__version__
