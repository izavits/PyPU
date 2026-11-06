"""PU learning models.

Every model follows the interface of :class:`pypu.base.BasePUClassifier`::

    from pypu.models import NaivePU

    clf = NaivePU().fit(X_train, s_train)  # s: 1 = labeled positive, 0 = unlabeled
    proba = clf.predict_proba(X_test)[:, 1]

Models are imported lazily: importing :mod:`pypu.models` does not import the optional
dependencies (such as PyTorch) of models that are not used.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# Public model name -> module that defines it. Register every new model here, in __all__ and
# in the TYPE_CHECKING block below (tests check that the three agree).
_MODELS: dict[str, str] = {
    "IFPU": "pypu.models.ifpu",
    "NaivePU": "pypu.models.naive",
}

__all__ = ["IFPU", "NaivePU"]

if TYPE_CHECKING:
    from pypu.models.ifpu import IFPU
    from pypu.models.naive import NaivePU


def __getattr__(name: str):
    try:
        module_name = _MODELS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    model = getattr(import_module(module_name), name)
    globals()[name] = model  # later lookups skip __getattr__
    return model


def __dir__():
    return sorted(set(globals()) | set(_MODELS))
