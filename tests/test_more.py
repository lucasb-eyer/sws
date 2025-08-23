import pytest
import sws


def test_finalize_tuple_and_set_freezing():
    c = sws.Config(
        a=2,
        t=(1, sws.lazy(lambda c: c.a + 1)),
        s={1, sws.lazy(lambda c: c.a)},
    )
    f = c.finalize()
    assert isinstance(f.t, tuple)
    assert f.t == (1, 3)
    assert isinstance(f.s, frozenset)
    assert f.s == frozenset({1, 2})


def test_finalize_mapping_in_container_raises():
    c = sws.Config(
        a=1,
        bad=[sws.lazy(lambda c: {"k": c.a})],
    )
    with pytest.raises(sws.FinalizeError):
        c.finalize()


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
    f1 = base.finalize(["a", "3"])  # overrides during finalize
    assert f1.a == 3 and f1.b == 2

    dbase = {"x": 1, "y": {"z": 2}}
    f2 = sws.Config(**dbase).finalize(["y.z", "5"])  # build Config from dict
    assert f2.x == 1 and f2.y.z == 5


def test_delete_via_subview_and_contains_on_view():
    c = sws.Config(model={"width": 128, "depth": 4})
    mv = c["model"]
    assert "width" in mv and "depth" in mv
    del mv["width"]
    assert "width" not in mv and "model.width" not in c
    del c["model"]
    assert "model" not in c and "depth" not in mv


def test_finalize_missing_key_error_message():
    c = sws.Config(x=sws.lazy(lambda c: c.y))
    with pytest.raises(sws.FinalizeError) as ei:
        c.finalize()
    # Message should indicate missing key path and field name
    msg = str(ei.value)
    assert "Missing key" in msg and "'y'" in msg and "'x'" in msg


def test_overrides_boolean_and_list_eval():
    base = sws.Config(flag=False, lst=[1])
    f = base.finalize(["flag", "True", "lst", "[1,2,3]"])
    assert f.flag is True and f.lst == (1, 2, 3)
