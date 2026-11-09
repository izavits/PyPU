"""PyPU: positive-unlabeled (PU) learning in Python.

PU learning trains a binary classifier from positive examples and unlabeled examples
(a mix of positives and negatives), without any example labeled negative. PyPU collects
PU learning methods behind one scikit-learn style interface: every model is fitted with
``fit(X, s)``, where ``s[i] == 1`` marks a labeled positive and ``s[i] == 0`` an unlabeled
sample.

Subpackages
-----------
models
    The PU classifiers, e.g. ``from pypu.models import NaivePU``.
datasets
    PU data from fully labeled datasets, under several labeling mechanisms.
metrics
    Evaluation with the true labels, and estimates from PU labels alone.
prior
    Class-prior estimation from PU data.
utils
    Input validation and helpers shared by the models.
"""

from pypu.utils import show_versions

try:
    from pypu._version import __version__
except ImportError:  # pragma: no cover - only when running from a source tree without installing
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "show_versions"]
