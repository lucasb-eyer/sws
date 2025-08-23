import pytest
import sws
from sws import Config, lazy


def test_simple():
    c = Config(lr=0.1, wd=lazy(lambda c: c.lr * 0.5))
    f = c.finalize()
    assert f.wd == pytest.approx(0.05)

    f2 = c.finalize(["lr=10"])  # override lr
    assert f2.lr == 10
    assert f2.wd == 5

    f3 = c.finalize(["lr=10", "wd=0.2"])  # explicit wd suppresses computed
    assert f3.lr == 10
    assert f3.wd == pytest.approx(0.2)


def test_computed_nested_and_root():
    c = Config(
        model={"lr": 1e-3, "wd": lazy(lambda c: c.lr * 0.5)},
        optimizer={"wd": lazy(lambda c: c.root.model.lr * 0.1)},
    )
    f = c.finalize()
    assert f.model.wd == pytest.approx(5e-4)
    assert f.optimizer.wd == pytest.approx(1e-4)


def test_computed_containers_and_freeze():
    c = Config(lr=3, aug=[1, lazy(lambda c: c.lr * 2)])
    f = c.finalize()
    assert isinstance(f.aug, tuple)
    assert f.aug == (1, 6)
    with pytest.raises(TypeError):
        f["new"] = 1
    with pytest.raises(TypeError):
        del f["lr"]
    # Subview is also frozen
    fm = sws.FinalConfig(f.to_flat_dict())
    with pytest.raises(TypeError):
        fm["lr"] = 1


def test_cycle_detection():
    c = Config(
        a=lazy(lambda c: c.b),
        b=lazy(lambda c: c.a),
    )
    with pytest.raises(sws.CycleError):
        c.finalize()


# --- Override parsing and behavior tests ---

def test_overrides_notypes():
    c = Config(
        an_int=1,
        a_string="hi",
        a_float=0.3,
    )
    f1 = c.finalize(["an_int='hello shapeshifter'"])
    assert f1.an_int == "hello shapeshifter"


def test_overrides_nested():
    c = Config(lr=0.1, model=dict(width=128, depth=4))
    f1 = c.finalize(["lr=0.001", "model.width=64", "model.depth=8"])
    assert f1.lr == 0.001
    assert f1.model.width == 64
    assert f1.model.depth == 8


def test_overrides_inexistent():
    c = Config(lr=0.1, model=dict(width=128, depth=4))
    with pytest.raises(AttributeError):
        c.finalize(["lol=0.001"])
    with pytest.raises(AttributeError):
        c.finalize(["model.expand=2"])


def test_overrides_expressions():
    c = Config(
        an_int=1,
        a_string="hi",
        a_float=0.3,
    )
    f1 = c.finalize(["an_int=3 * 2", "a_string=','.join('abc')"])
    assert f1.an_int == 6
    assert f1.a_string == "a,b,c"


def test_overrides_expressions_with_c_view():
    base = Config(lr=1.0, model={"width": 128}, foo=0, bar=0)
    # Reference current config state via c
    f1 = base.finalize(["foo=c.lr", "bar=c.model.width"])
    assert f1.foo == 1.0 and f1.bar == 128
    # Order matters: left-to-right application of overrides
    f2 = base.finalize(["foo=c.lr", "lr=3", "bar=c.lr"])
    assert f2.foo == 1.0 and f2.bar == 3

