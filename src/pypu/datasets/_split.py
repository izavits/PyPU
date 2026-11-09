"""Train/test split with PU labels on the training part."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from pypu.datasets._common import check_true_labels
from pypu.datasets._labeling import (
    label_sar_cluster,
    label_sar_confidence,
    label_sar_knn,
    label_sar_propensity,
    label_scar,
)

_MECHANISMS = {
    "scar": label_scar,
    "sar_knn": label_sar_knn,
    "sar_propensity": label_sar_propensity,
    "sar_cluster": label_sar_cluster,
    "sar_confidence": label_sar_confidence,
}


class PUSplit(NamedTuple):
    """Result of :func:`make_pu_split`.

    A named tuple, so it can be unpacked in field order or read by attribute.
    """

    X_train: Any
    """Training features."""
    X_test: Any
    """Test features."""
    s_train: np.ndarray
    """PU labels of the training samples: 1 for a labeled positive, 0 for unlabeled."""
    y_train: np.ndarray
    """True labels of the training samples, hidden from the model, for evaluation."""
    y_test: np.ndarray
    """True labels of the test samples."""


def make_pu_split(
    X,
    y,
    *,
    label_frequency=0.5,
    mechanism="scar",
    test_size=0.25,
    random_state=None,
    **mechanism_params,
):
    r"""Split a labeled dataset into PU training data and a fully labeled test set.

    The split is stratified on ``y``, so both parts have the class prior of the full data.
    The training labels are then turned into PU labels with the chosen labeling mechanism,
    while the test set keeps its true labels for evaluation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features. Lists, arrays, sparse matrices and DataFrames are accepted, and the parts
        are returned in the same type.
    y : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    label_frequency : float, default=0.5
        Fraction of the training positives to label, :math:`c = P(s = 1 \mid y = 1)`.
    mechanism : {"scar", "sar_knn", "sar_propensity", "sar_cluster", "sar_confidence"}
        Labeling mechanism, ``"scar"`` by default: :func:`label_scar`, :func:`label_sar_knn`,
        :func:`label_sar_propensity`, :func:`label_sar_cluster` or
        :func:`label_sar_confidence`.
    test_size : float or int, default=0.25
        Fraction (float) or number (int) of samples in the test set.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator for both the split and the labels.
    **mechanism_params
        Further keyword arguments of the labeling function, such as ``bias="far"`` for
        ``"sar_knn"``.

    Returns
    -------
    split : PUSplit
        A named tuple ``(X_train, X_test, s_train, y_train, y_test)``.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import NaivePU
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(
    ...     X, y, label_frequency=0.3, mechanism="sar_knn", random_state=0
    ... )
    >>> X_train.shape, X_test.shape
    ((750, 20), (250, 20))
    >>> clf = NaivePU(prior=y_train.mean()).fit(X_train, s_train)
    """
    y = check_true_labels(y)
    if mechanism not in _MECHANISMS:
        raise ValueError(f"mechanism must be one of {list(_MECHANISMS)}, got {mechanism!r}.")
    if "return_propensity" in mechanism_params:
        raise TypeError(
            "make_pu_split does not return propensities; call label_sar_propensity directly."
        )
    rng = check_random_state(random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=rng
    )
    labeling_function = _MECHANISMS[mechanism]
    data = (y_train,) if mechanism == "scar" else (X_train, y_train)
    s_train = labeling_function(
        *data, label_frequency=label_frequency, random_state=rng, **mechanism_params
    )
    return PUSplit(X_train, X_test, s_train, y_train, y_test)
