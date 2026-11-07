import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

import pypu
import pypu.models


def test_version_is_set():
    assert isinstance(pypu.__version__, str)
    assert pypu.__version__


def test_models_are_listed_and_importable():
    assert "NaivePU" in pypu.models.__all__
    for name in pypu.models.__all__:
        assert getattr(pypu.models, name).__name__ == name
        assert name in dir(pypu.models)


def test_model_registry_and_all_agree():
    source = Path(pypu.models.__file__).read_text()
    imported = {
        (match.group(1), name.strip())
        for match in re.finditer(r"from (pypu\.models\.\w+) import ([\w, ]+)", source)
        for name in match.group(2).split(",")
    }
    assert sorted(pypu.models.__all__) == sorted(pypu.models._MODELS)
    for name, module in pypu.models._MODELS.items():
        assert (module, name) in imported, f"{name} missing from TYPE_CHECKING"


def test_unknown_model_raises_attribute_error():
    with pytest.raises(AttributeError, match="NoSuchModel"):
        pypu.models.NoSuchModel  # noqa: B018


def test_models_are_imported_lazily():
    # A fresh interpreter: importing pypu.models must not import any model module.
    code = "import sys, pypu.models; print(any(m.startswith('pypu.models.') for m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("module_name", ["pypu.datasets", "pypu.metrics", "pypu.prior"])
def test_public_functions_are_documented(module_name):
    module = importlib.import_module(module_name)
    for name in module.__all__:
        obj = getattr(module, name)
        assert obj.__module__.startswith(f"{module_name}.")
        if name not in ("PUSplit", "BasePriorEstimator"):
            assert "Examples\n" in obj.__doc__, f"{name} has no Examples section"


def test_show_versions(capsys):
    pypu.show_versions()
    out = capsys.readouterr().out
    assert f"pypu: {pypu.__version__}" in out
    assert "scikit-learn:" in out
