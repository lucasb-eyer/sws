from importlib.metadata import PackageNotFoundError, version as _version
from .config import Config, Fn, FinalConfig, FinalizeError, CycleError

def run(main, *, argv=None, config_flag="--config", default_func="get_config", forward_extras=False):
    """Entry-point helper to load a config file, finalize it, and call `main`.

    Behavior:
    - Looks for a `--config` flag in argv (supports `--config path.py[:func]` and
      `--config=path.py[:func]`). If present, loads the Python file and calls the
      specified function (default: `get_config`) to obtain a builder `Config`.
    - Partitions remaining argv into config overrides and unused tokens:
      all `key=value` tokens are passed to `finalize` (which will raise on
      unknown keys or invalid group assignments). If `forward_extras` is True,
      non `key=value` tokens are forwarded to `main` as a second positional
      argument `unused` with preserved order; otherwise they are ignored here.
    - Calls `main(finalized_config, unused_tokens)` when `forward_extras` is True,
      else calls `main(finalized_config)`. Returns the result of `main`.
    """
    import sys as _sys
    import runpy as _runpy
    import os as _os

    args = list(_sys.argv[1:] if argv is None else argv)

    # Extract --config value if present; consume it from args
    cfg_path = None
    func_name = default_func
    consumed = set()
    for i, tok in enumerate(args):
        if tok == config_flag:
            if i + 1 < len(args):
                cfg_path = args[i + 1]
                consumed.update({i, i + 1})
            break
        if tok.startswith(config_flag + "="):
            cfg_path = tok.split("=", 1)[1]
            consumed.add(i)
            break

    # Remaining tokens after consuming --config
    args = [tok for i, tok in enumerate(args) if i not in consumed]

    if cfg_path:
        if ":" in cfg_path:
            cfg_path, func_name = cfg_path.split(":", 1)
        # Resolve path relative to CWD
        cfg_path = _os.path.abspath(cfg_path)
        factory = _runpy.run_path(cfg_path).get(func_name)
    else:
        # Try convenient default: load from the caller's file (typically __main__)
        import inspect as _inspect
        caller_file = _inspect.stack()[1].filename
        if caller_file and _os.path.exists(caller_file):
            cfg_path = caller_file
            factory = _runpy.run_path(cfg_path).get(func_name, lambda: Config())

    if cfg_path:  # Execute file, retrieve factory, get Config.
        if not callable(factory):
            raise AttributeError(f"Function {func_name!r} not found in {cfg_path}")
        builder = factory()
        if not isinstance(builder, Config):
            raise TypeError("Config factory must return a sws.Config")
    else:
        builder = Config()

    final = builder.finalize([arg for arg in args if "=" in arg])
    if forward_extras:
        return main(final, [arg for arg in args if "=" not in arg])
    else:
        return main(final)

try:
    # Distribution name on PyPI is 'sws-config'; import name remains 'sws'
    __version__ = _version("sws-config")
except PackageNotFoundError:  # pragma: no cover - during local, non-installed runs
    __version__ = "0.0.0"
