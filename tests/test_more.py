import pytest
import sws


def test_finalize_tuple_and_set():
    c = sws.Config(a=2)
    c.t = lambda: (1, c.a + 1)
    c.s = lambda: {1, c.a}
    f = c.finalize()
    assert isinstance(f.t, tuple)
    assert f.t == (1, 3)
    assert isinstance(f.s, set)
    assert f.s == {1, 2}


def test_frozen_prefix_view_and_contains():
    c = sws.Config(model={"width": 128, "depth": 4})
    f = c.finalize()
    mv = f["model"]
    assert isinstance(mv, sws.FinalConfig)
    assert "width" in mv and "model.width" not in mv
    assert mv.to_flat_dict() == {"width": 128, "depth": 4}
    with pytest.raises(TypeError):
        mv["width"] = 64


def test_overrides_from_dict_like_base():
    base = sws.Config(a=1, b=2)
    f1 = base.finalize(["a=3"])  # overrides during finalize
    assert f1.a == 3 and f1.b == 2

    dbase = {"x": 1, "y": {"z": 2}}
    f2 = sws.Config(**dbase).finalize(["y.z=5"])  # build Config from dict
    assert f2.x == 1 and f2.y.z == 5


def test_delete_via_subview_and_contains_on_view():
    c = sws.Config(model={"width": 128, "depth": 4})
    mv = c["model"]
    assert "width" in mv and "depth" in mv
    del mv["width"]
    assert "width" not in mv and "model.width" not in c
    del c["model"]
    assert "model" not in c and "depth" not in mv


def test_finalize_missing_key():
    c = sws.Config()
    c.x = lambda: c.y
    with pytest.raises(KeyError):
        c.finalize()


def test_overrides_boolean_and_list_eval():
    base = sws.Config(flag=False, lst=[1])
    f = base.finalize(["flag=True", "lst=[1,2,3]"])
    assert f.flag is True and f.lst == [1, 2, 3]


def test_bug1():
    c = sws.Config()
    c.a = c.b
    assert c.finalize().to_dict() == {}


def test_bug2():
    c = sws.Config()
    c.a.b = 3
    assert c.finalize().to_dict() == dict(a=dict(b=3))


def test_fn_wraps_callable_as_value():
    c = sws.Config()

    def greet(name):
        return f"hi {name}"

    # Store callable as a plain value via Fn wrapper
    c.greet = sws.Fn(greet)
    f = c.finalize()

    assert callable(f.greet)
    assert f.greet("bob") == "hi bob"


def test_bare_callable_requires_fn_or_zero_arg():
    c = sws.Config()

    def needs_arg(x):
        return x

    # Assigning a bare callable will be treated as lazy and invoked at finalize,
    # which fails if it requires arguments.
    c.bad = needs_arg
    with pytest.raises(TypeError):
        c.finalize()


def test_fn_prevents_eager_execution_of_zero_arg_callable():
    c = sws.Config()

    # Without Fn this would execute at finalize and raise; with Fn it should not execute.
    c.zero = sws.Fn(lambda: 1 / 0)
    f = c.finalize()

    assert callable(f.zero)
    with pytest.raises(ZeroDivisionError):
        f.zero()


def test_iterate_finalconfig():
    # Empty config iterates to nothing
    empty = sws.Config().finalize()
    assert list(empty) == []

    # Non-empty: top-level iterates over top-level child segments
    c = sws.Config(lr=0.1, model=dict(width=128, depth=4))
    assert sorted(c.finalize()) == ["lr", "model"]

    # Can also iterate on child
    assert sorted(c.finalize()["model"]) == ["depth", "width"]
