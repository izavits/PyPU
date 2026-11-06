"""Validation and sampling helpers shared by the PU data generators."""

from __future__ import annotations

import math
import numbers

import numpy as np

from pypu.utils.validation import check_true_labels

__all__ = [
    "check_label_frequency",
    "check_positive_int",
    "check_true_labels",
    "labels_from_indices",
    "n_labeled_positives",
    "weighted_choice",
]


def check_label_frequency(label_frequency, *, allow_one: bool = True) -> float:
    r"""Validate a label frequency :math:`c = P(s = 1 \mid y = 1)`."""
    interval = "(0, 1]" if allow_one else "(0, 1)"
    if isinstance(label_frequency, bool) or not isinstance(label_frequency, numbers.Real):
        raise TypeError(f"label_frequency must be a float in {interval}, got {label_frequency!r}.")
    c = float(label_frequency)
    if not (0.0 < c < 1.0 or (allow_one and c == 1.0)):
        raise ValueError(f"label_frequency must be in {interval}, got {c}.")
    return c


def check_positive_int(value, name: str) -> int:
    """Validate a parameter that must be an integer >= 1."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
    return int(value)


def n_labeled_positives(label_frequency: float, n_positive: int) -> int:
    """Return how many positives to label: ``label_frequency * n_positive``, rounded half up."""
    n_labeled = math.floor(label_frequency * n_positive + 0.5)
    if n_labeled == 0:
        raise ValueError(
            f"label_frequency={label_frequency} labels none of the {n_positive} positives; "
            "use a larger label_frequency."
        )
    return n_labeled


def labels_from_indices(n_samples: int, labeled) -> np.ndarray:
    """PU labels with ones at the ``labeled`` indices."""
    s = np.zeros(n_samples, dtype=np.int64)
    s[labeled] = 1
    return s


def weighted_choice(rng: np.random.RandomState, weights, size: int) -> np.ndarray:
    """Draw ``size`` distinct indices with probability proportional to ``weights``.

    Indices with zero weight are drawn, uniformly at random, only after every index with a
    positive weight has been drawn.
    """
    weights = np.asarray(weights, dtype=np.float64)
    candidates = np.flatnonzero(weights > 0)
    n_first = min(size, candidates.size)
    if n_first == 0:
        first = np.empty(0, dtype=np.intp)
    else:
        p = weights[candidates] / weights[candidates].sum()
        first = rng.choice(candidates, n_first, replace=False, p=p)
    if n_first == size:
        return first
    rest = rng.choice(np.flatnonzero(weights <= 0), size - n_first, replace=False)
    return np.concatenate([first, rest])
