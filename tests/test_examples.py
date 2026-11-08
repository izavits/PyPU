"""Run every script in examples/, so that the examples keep working as PyPU changes.

They need matplotlib (the ``examples`` dependency group), and are skipped without it. CI
runs them in a job of their own.
"""

import runpy
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXAMPLES = Path(__file__).parents[1] / "examples"
SCRIPTS = sorted(EXAMPLES.rglob("plot_*.py"))
# Optional modules an example cannot run without (plot_compare_methods.py and
# plot_prior_estimation.py skip the models that need them instead).
REQUIRES = {
    "plot_almpu.py": ["torch", "jenkspy"],
    "plot_lbe.py": ["torch"],
    "plot_pue.py": ["torch"],
    "plot_upu_boost.py": ["xgboost"],
    "plot_nnpu_boost.py": ["xgboost"],
    "plot_imbalanced_nnpu_boost.py": ["xgboost"],
}


def test_every_script_is_listed():
    # examples/README.rst lists every script, so that none is left out of the index.
    readme = (EXAMPLES / "README.rst").read_text()
    assert SCRIPTS
    assert [s.name for s in SCRIPTS if s.name not in readme] == []


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_example_runs(script, monkeypatch):
    for module in REQUIRES.get(script.name, []):
        pytest.importorskip(module)
    monkeypatch.chdir(script.parent)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        plt.close("all")
