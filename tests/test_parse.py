import pytest
import sws


def test_overrides_simple():
    c = sws.Config(
        an_int=1,
        a_string="hi",
        a_float=0.3,
    )
    f1 = c.finalize(["an_int", "3"])
    assert f1.an_int == 3
    assert f1.a_string == "hi"
    assert f1.a_float == 0.3

    f2 = c.finalize(["a_float", "0.1", "a_string", "lol hi", "an_int", "3"])
    assert f2.an_int == 3
    assert f2.a_string == "lol hi"
    assert f2.a_float == 0.1

    f3 = c.finalize(["a_float=0.1", "a_string=lol hi", "an_int", "3"])
    assert f3.an_int == 3
    assert f3.a_string == "lol hi"
    assert f3.a_float == 0.1


def test_overrides_notypes():
    c = sws.Config(
        an_int=1,
        a_string="hi",
        a_float=0.3,
    )
    f1 = c.finalize(["an_int", "hello shapeshifter"])
    assert f1.an_int == "hello shapeshifter"


def test_overrides_nested():
    c = sws.Config(lr=0.1, model=dict(width=128, depth=4))
    f1 = c.finalize(["lr", "0.001", "model.width", "64", "model.depth", "8"])
    assert f1.lr == 0.001
    assert f1.model.width == 64
    assert f1.model.depth == 8


def test_overrides_inexistent():
    c = sws.Config(lr=0.1, model=dict(width=128, depth=4))
    with pytest.raises(AttributeError):
        c.finalize(["lol", "0.001"])
    with pytest.raises(AttributeError):
        c.finalize(["model.expand", "2"])


def test_overrides_expressions():
    c = sws.Config(
        an_int=1,
        a_string="hi",
        a_float=0.3,
    )
    f1 = c.finalize(["an_int", "3 * 2", "a_string=','.join('abc')"])
    assert f1.an_int == 6
    assert f1.a_string == "a,b,c"


def test_overrides_expressions_with_c_view():
    base = sws.Config(lr=1.0, model={"width": 128}, foo=0, bar=0)
    # Reference current config state via c
    f1 = base.finalize(["foo", "c.lr", "bar", "c.model.width"])
    assert f1.foo == 1.0 and f1.bar == 128
    # Order matters: left-to-right application of overrides
    f2 = base.finalize(["foo", "c.lr", "lr", "3", "bar", "c.lr"])
    assert f2.foo == 1.0 and f2.bar == 3
