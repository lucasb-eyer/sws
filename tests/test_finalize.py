import pytest
import sws


def test_computed_basic_and_override():
    c = sws.Config(lr=0.1, wd=sws.lazy(lambda c: c.lr * 0.5))
    f = c.finalize()
    assert pytest.approx(f.wd) == 0.05

    f2 = c.finalize(["lr", "10"])  # override lr
    assert f2.lr == 10
    assert f2.wd == 5

    f3 = c.finalize(["lr", "10", "wd", "0.2"])  # explicit wd suppresses computed
    assert f3.lr == 10
    assert pytest.approx(f3.wd) == 0.2


def test_computed_nested_and_root():
    c = sws.Config(
        model={"lr": 1e-3, "wd": sws.lazy(lambda c: c.lr * 0.5)},
        optimizer={"wd": sws.lazy(lambda c: c.root.model.lr * 0.1)},
    )
    f = c.finalize()
    assert pytest.approx(f.model.wd) == 5e-4
    assert pytest.approx(f.optimizer.wd) == 1e-4


def test_computed_containers_and_freeze():
    c = sws.Config(lr=3, aug=[1, sws.lazy(lambda c: c.lr * 2)])
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
    c = sws.Config(
        a=sws.lazy(lambda c: c.b),
        b=sws.lazy(lambda c: c.a),
    )
    with pytest.raises(sws.CycleError):
        c.finalize()
