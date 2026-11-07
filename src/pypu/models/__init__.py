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
    "ALMPU": "pypu.models.almpu",
    "IFPNU": "pypu.models.ifpnu",
    "IFPU": "pypu.models.ifpu",
    "ElkanNoto": "pypu.models.elkan_noto",
    "ImbalancedNNPUBoost": "pypu.models.risk_boost",
    "LBE": "pypu.models.lbe",
    "NNPUBoost": "pypu.models.risk_boost",
    "NaivePU": "pypu.models.naive",
    "PUBagging": "pypu.models.bagging",
    "PUHellingerForest": "pypu.models.hellinger",
    "PUHellingerTree": "pypu.models.hellinger",
    "PUe": "pypu.models.pue",
    "SAREM": "pypu.models.sarem",
    "UPUBoost": "pypu.models.risk_boost",
}

__all__ = [
    "ALMPU",
    "IFPNU",
    "IFPU",
    "LBE",
    "SAREM",
    "ElkanNoto",
    "ImbalancedNNPUBoost",
    "NNPUBoost",
    "NaivePU",
    "PUBagging",
    "PUHellingerForest",
    "PUHellingerTree",
    "PUe",
    "UPUBoost",
]

if TYPE_CHECKING:
    from pypu.models.almpu import ALMPU
    from pypu.models.bagging import PUBagging
    from pypu.models.elkan_noto import ElkanNoto
    from pypu.models.hellinger import PUHellingerForest, PUHellingerTree
    from pypu.models.ifpnu import IFPNU
    from pypu.models.ifpu import IFPU
    from pypu.models.lbe import LBE
    from pypu.models.naive import NaivePU
    from pypu.models.pue import PUe
    from pypu.models.risk_boost import ImbalancedNNPUBoost, NNPUBoost, UPUBoost
    from pypu.models.sarem import SAREM


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
