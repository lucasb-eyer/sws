from concurrent.futures import ThreadPoolExecutor
import threading

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


def test_finalize_overrides_do_not_mutate_builder():
    c = sws.Config(lr=0.1)
    c.wd = lambda: c.lr * 0.5

    overridden = c.finalize(["lr=10"])
    original = c.finalize()

    assert overridden.lr == 10
    assert overridden.wd == 5
    assert original.lr == 0.1
    assert original.wd == 0.05


def test_finalized_containers_do_not_mutate_builder_or_later_finalizations():
    original = [{"values": [1]}, {1}]
    c = sws.Config(xs=original)

    first = c.finalize()
    first.xs[0]["values"].append(2)
    first.xs[1].add(2)

    second = c.finalize()
    assert second.xs == [{"values": [1]}, {1}]
    assert original == [{"values": [1]}, {1}]

    original.append({"later": True})
    assert first.xs == [{"values": [1, 2]}, {1, 2}]
    assert second.xs == [{"values": [1]}, {1}]


def test_finalization_preserves_leaf_identity_and_container_aliases():
    marker = object()
    shared = [marker]
    c = sws.Config(first=shared, second=shared)

    final = c.finalize()

    assert final.first is final.second
    assert final.first is not shared
    assert final.first[0] is marker


def test_failed_finalize_restores_builder_phase_and_store():
    c = sws.Config(lr=0.1)

    with pytest.raises(sws.OverrideError):
        c.finalize(["unknown=1"])

    with pytest.raises(TypeError):
        _ = c.lr
    assert c.finalize().lr == 0.1


def test_failed_cycle_finalize_does_not_poison_retry():
    c = sws.Config()
    c.a = lambda: c.b
    c.b = lambda: c.a

    with pytest.raises(sws.CycleError):
        c.finalize()

    c.b = 1
    assert c.finalize().a == 1


def test_lazy_failure_identifies_the_actual_failing_field():
    c = sws.Config()
    c.result = lambda: c.model.dim * 2
    c.model.dim = lambda: "wide" + 1

    with pytest.raises(
        sws.FinalizeError, match="Failed to resolve lazy field 'model.dim'"
    ) as exc_info:
        c.finalize()

    assert isinstance(exc_info.value.__cause__, TypeError)
    assert "str" in str(exc_info.value.__cause__)


def test_lazy_values_are_memoized_once_per_finalize():
    calls = 0
    c = sws.Config()

    def value():
        nonlocal calls
        calls += 1
        return calls

    # Define the dependent field first so resolving it has to resolve value.
    c.total = lambda: c.value + c.value
    c.value = value

    first = c.finalize()
    assert first.total == 2
    assert first.value == 1
    assert calls == 1

    second = c.finalize()
    assert second.total == 4
    assert second.value == 2
    assert calls == 2


def test_lazy_memoization_prevents_exponential_dependency_resolution():
    calls = 0
    c = sws.Config(x0=1)

    def make_double(previous):
        def double():
            nonlocal calls
            calls += 1
            return c[previous] + c[previous]
        return double

    for i in range(1, 19):
        c[f"x{i}"] = make_double(f"x{i - 1}")

    final = c.finalize()
    assert final.x18 == 2 ** 18
    assert calls == 18


def test_concurrent_finalizations_use_isolated_store_and_resolution_state():
    barrier = threading.Barrier(2)
    overlap = threading.Event()
    overlap.set()

    c = sws.Config()

    def slow_value():
        if overlap.is_set():
            barrier.wait(timeout=2)
        return c.value

    # Put the synchronizing lazy first so both finalizations overlap before
    # resolving their differently overridden values.
    c.slow = slow_value
    c.value = 0

    def finalize(value):
        final = c.finalize([f"value={value}"])
        return final.value, final.slow

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(finalize, [1, 2]))

    assert results == [(1, 1), (2, 2)]

    overlap.clear()
    original = c.finalize()
    assert (original.value, original.slow) == (0, 0)


def test_captured_subtree_view_sees_temporary_overrides():
    c = sws.Config()
    model = c.model
    model.width = 128
    c.half_width = lambda: model.width // 2

    overridden = c.finalize(["model.width=64"])
    original = c.finalize()

    assert overridden.half_width == 32
    assert original.half_width == 64


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


def test_finalize_missing_key_supports_getattr_default():
    c = sws.Config()
    c.x = lambda: getattr(c, "y", None)
    assert c.finalize().x is None


def test_finalize_missing_key_supports_hasattr():
    c = sws.Config()
    c.x = lambda: hasattr(c, "y")
    assert c.finalize().x is False


def test_finalize_config_alias_flattening_for_optional_subtree():
    c = sws.Config()
    c.eval.tokenizer = lambda: getattr(c, "tokenizer", None)

    f = c.finalize()
    assert f.eval.tokenizer is None
    assert f.to_flat_dict() == {"eval.tokenizer": None}

    f = c.finalize(["tokenizer.first_N:=128"])
    assert f.eval.tokenizer.first_N == 128
    assert f.to_flat_dict() == {
        "tokenizer.first_N": 128,
        "eval.tokenizer.first_N": 128,
    }


def test_finalize_self_alias_subtree_stays_finite():
    c = sws.Config()
    c.a.b = 1
    c.a.alias = lambda: c.a
    f = c.finalize()

    assert f.a.alias.b == 1
    assert f.to_flat_dict() == {
        "a.b": 1,
        "a.alias.b": 1,
    }


def test_finalize_config_alias_cycles_are_detected():
    c = sws.Config()
    c.x = lambda: c
    c.y = lambda: c.x

    with pytest.raises(sws.CycleError):
        c.finalize()


def test_raising_lazy_does_not_fake_cycle():
    # A lazy that raises used to leave a stale entry in the cycle-tracking
    # state when the error was swallowed (as argv override parsing does),
    # so a later, perfectly ordinary resolution of the same key raised a
    # false CycleError instead of surfacing the real error.
    c = sws.Config()
    c.a = 1
    c.z = lambda: c.a * 2
    c.bad = lambda: 1 / 0

    with pytest.raises(sws.FinalizeError, match="lazy field 'bad'") as exc_info:
        c.finalize(["a=c.bad"])
    assert isinstance(exc_info.value.__cause__, ZeroDivisionError)


def test_overrides_boolean_and_list_eval():
    base = sws.Config(flag=False, lst=[1])
    f = base.finalize(["flag=True", "lst=[1,2,3]"])
    assert f.flag is True and f.lst == [1, 2, 3]


def test_assigning_empty_config_view_is_rejected():
    c = sws.Config()
    with pytest.raises(ValueError, match="Cannot assign empty Config view"):
        c.a = c.b


def test_assigning_eager_config_view_copies_its_values():
    c = sws.Config()
    c.m1.width = 100
    c.m1.depth = 4
    c.m2 = c.m1

    final = c.finalize(["m2.width=60"])
    assert final.m1.to_dict() == {"width": 100, "depth": 4}
    assert final.m2.to_dict() == {"width": 60, "depth": 4}


def test_assigning_lazy_config_view_is_rejected_before_stale_closure_copy():
    c = sws.Config()
    c.m1.width = 100
    c.m1.dim = lambda: c.m1.width // 2

    with pytest.raises(sws.LazySubtreeError) as exc_info:
        c.m2 = c.m1

    message = str(exc_info.value)
    assert "Cannot assign Config view to 'm2'" in message
    assert "'dim'" in message
    assert "preserve closures" in message
    assert c.finalize().to_dict() == {"m1": {"width": 100, "dim": 50}}


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


def test_fn_unwrapped_inside_containers():
    def f():
        return 1

    def g():
        return 2

    c = sws.Config()
    c.callbacks = [sws.Fn(f), (sws.Fn(g),)]
    c.by_name = lambda: {"f": sws.Fn(f)}  # a dict returned by a lazy stays a leaf
    final = c.finalize()

    assert final.callbacks[0] is f
    assert final.callbacks[1][0] is g
    assert final.by_name["f"] is f


def test_bare_callable_requires_fn_or_zero_arg():
    c = sws.Config()

    def needs_arg(x):
        return x

    # Assigning a bare callable will be treated as lazy and invoked at finalize,
    # which fails if it requires arguments.
    c.bad = needs_arg
    with pytest.raises(sws.FinalizeError, match="lazy field 'bad'") as exc_info:
        c.finalize()
    assert isinstance(exc_info.value.__cause__, TypeError)


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


def test_iterate_config_view():
    c = sws.Config()
    c.foo = dict(a=3, b=4)

    assert sorted(c) == ["foo"]
    assert sorted(c.foo) == ["a", "b"]
