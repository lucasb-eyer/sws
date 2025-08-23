import sys
import os
import sws


def _call_run(argv, main, forward_extras=True):
    old = sys.argv
    try:
        sys.argv = [old[0]] + argv
        return sws.run(main, forward_extras=forward_extras)
    finally:
        sys.argv = old


def test_run_with_config_and_overrides():
    cfg_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_config.py")

    def main(final, unused):
        # Return for assertions
        return final, unused

    final, unused = _call_run([
        "--config", cfg_path,
        "lr=0.2",
        "model.width=256",
        "--extra",
        "pos",
    ], main)

    assert final.lr == 0.2
    assert final.model.width == 256
    # Unused keeps order for non key=value tokens
    assert unused == ["--extra", "pos"]


def test_run_with_config_colon_func_and_group_override_ignored():
    cfg_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_config.py:get_config")

    def main(final, unused):
        return final, unused

    import pytest
    with pytest.raises(AttributeError):
        _call_run([
            f"--config={cfg_path}",
            "model=bad",  # invalid: group assignment should raise
            "model.depth=6",
        ], main)

def test_run_unknown_key_raises():
    cfg_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_config.py")

    def main(final, unused):
        return final, unused

    import pytest
    with pytest.raises(AttributeError):
        _call_run([
            "--config", cfg_path,
            "foo=unknown",  # not a known key; should raise
        ], main)


def test_run_without_config_uses_empty_builder():
    def main(final, unused):
        return final, unused

    final, unused = _call_run(["--other", "x"], main)
    # In library-call context (not as __main__), default attempts caller file,
    # which has no get_config, so falls back to empty config.
    assert isinstance(final, sws.FinalConfig)
    assert len(list(final.to_flat_dict().keys())) == 0
    assert unused == ["--other", "x"]


def test_run_default_loads_caller_script(tmp_path):
    import subprocess, sys, os
    script = os.path.join(os.path.dirname(__file__), "fixtures", "runner_default.py")
    root = os.path.dirname(os.path.dirname(__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # Pass an override and an unused positional
    res = subprocess.run([sys.executable, script, "lr=0.5", "pos"], capture_output=True, text=True, env=env)
    assert res.returncode == 0
    out = res.stdout.strip().splitlines()
    assert "lr=0.5" in out[0]
    assert "unused=pos" in out[1]
