"""Print the versions of PyPU and its dependencies, for bug reports."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

# Distributions reported by show_versions: core dependencies first, then optional ones.
_DEPENDENCIES = ("numpy", "scipy", "scikit-learn", "lightgbm", "xgboost", "torch", "pyod")


def show_versions() -> None:
    """Print the versions of PyPU, Python and the main dependencies.

    Include this output when reporting a bug.
    """
    import pypu

    print(f"pypu: {pypu.__version__}")
    print(f"python: {sys.version.split()[0]} ({platform.platform()})")
    for dist in _DEPENDENCIES:
        try:
            dist_version = version(dist)
        except PackageNotFoundError:
            dist_version = "not installed"
        print(f"{dist}: {dist_version}")
