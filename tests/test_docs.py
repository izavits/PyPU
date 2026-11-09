"""Check that the API reference of the documentation site lists the whole public API.

docs/api/index.rst names each object in an autosummary list, under a ``currentmodule``.
These tests read the file as text, so they need none of the documentation tools.
"""

import importlib
import re
from pathlib import Path

API_INDEX = Path(__file__).parents[1] / "docs" / "api" / "index.rst"
SUBPACKAGES = ["models", "datasets", "metrics", "prior", "utils"]


def documented_names():
    """Return the qualified names in the autosummary lists of the API index."""
    names, module, in_autosummary = set(), None, False
    for line in API_INDEX.read_text(encoding="utf-8").splitlines():
        if match := re.match(r"\.\. currentmodule:: (\S+)", line):
            module = match[1]
        elif line.startswith(".. autosummary::"):
            in_autosummary = True
        elif in_autosummary and line and not line.startswith(" "):
            in_autosummary = False
        elif in_autosummary and (match := re.fullmatch(r" {3}(\w+)", line)):
            names.add(f"{module}.{match[1]}")
    return names


def public_names():
    names = {"pypu.base.BasePUClassifier"}
    for subpackage in SUBPACKAGES:
        module = importlib.import_module(f"pypu.{subpackage}")
        names.update(f"pypu.{subpackage}.{name}" for name in module.__all__)
    return names


def test_api_reference_lists_every_public_name():
    assert sorted(public_names() - documented_names()) == []


def test_api_reference_lists_only_public_names():
    assert sorted(documented_names() - public_names()) == []
