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
    assert sorted(pypu.models.__all__) == sorted(pypu.models._MODELS)
    for name, module in pypu.models._MODELS.items():
        assert f"from {module} import {name}" in source, f"{name} missing from TYPE_CHECKING"


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


def test_datasets_public_functions_are_documented():
    import pypu.datasets

    for name in pypu.datasets.__all__:
        obj = getattr(pypu.datasets, name)
        assert obj.__module__.startswith("pypu.datasets.")
        if name != "PUSplit":
            assert "Examples\n" in obj.__doc__, f"{name} has no Examples section"


def test_show_versions(capsys):
    pypu.show_versions()
    out = capsys.readouterr().out
    assert f"pypu: {pypu.__version__}" in out
    assert "scikit-learn:" in out
