from importlib.metadata import PackageNotFoundError, version as _version
from .config import Config, lazy, Computed, FinalConfig, FinalizeError, CycleError

try:
    # Distribution name on PyPI is 'sws-config'; import name remains 'sws'
    __version__ = _version("sws-config")
except PackageNotFoundError:  # pragma: no cover - during local, non-installed runs
    __version__ = "0.0.0"
