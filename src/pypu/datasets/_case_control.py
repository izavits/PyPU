"""Case-control PU data: labeled positives and an unlabeled sample drawn separately."""

from __future__ import annotations

import math

import numpy as np
from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils.validation import check_consistent_length

from pypu.datasets._common import (
    check_label_frequency,
    check_true_labels,
    labels_from_indices,
    n_labeled_positives,
)


def make_case_control(X, y, *, label_frequency=0.5, random_state=None):
    r"""Make PU data in the case-control scenario from a labeled dataset.

    The labeling functions (such as :func:`label_scar`) follow the *single-training-set*
    scenario: the labeled positives are taken out of one sample, so the unlabeled set has
    fewer positives than the population. In the *case-control* scenario the labeled
    positives and the unlabeled set are drawn separately, and the unlabeled set is a random
    sample of the whole population, with its class prior [1]_.

    Here every positive is used: a ``label_frequency`` fraction of them, chosen uniformly at
    random, are labeled and the rest are unlabeled. Negatives are subsampled uniformly so
    that the unlabeled set has the same class prior as ``y``. Some negatives are therefore
    dropped, and the returned data are a subset of the rows of ``X``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features. Lists, arrays, sparse matrices and DataFrames are accepted, and the rows are
        returned in the same type.
    y : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    label_frequency : float, default=0.5
        Fraction of the positives to label, in ``(0, 1)``. In the returned data it is also
        the fraction of positives with ``s == 1``.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator, for reproducible data.

    Returns
    -------
    X_cc : array-like of shape (n_kept, n_features)
        The kept rows of ``X``, in their original order.
    s_cc : ndarray of shape (n_kept,), dtype int64
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.
    y_cc : ndarray of shape (n_kept,), dtype int64
        True labels of the kept rows.

    References
    ----------
    .. [1] J. Bekker and J. Davis. "Learning from positive and unlabeled data: a survey."
       Machine Learning, 2020.

    Examples
    --------
    >>> import numpy as np
    >>> from pypu.datasets import make_case_control
    >>> X = np.arange(20).reshape(-1, 1)
    >>> y = np.array([1] * 8 + [0] * 12)
    >>> X_cc, s_cc, y_cc = make_case_control(X, y, label_frequency=0.5, random_state=0)
    >>> int(s_cc.sum()), int((s_cc == 0).sum())  # 4 labeled, 4 + 6 unlabeled
    (4, 10)
    >>> float(y_cc[s_cc == 0].mean())  # the unlabeled set keeps the prior of y
    0.4
    """
    y = check_true_labels(y)
    check_consistent_length(X, y)
    c = check_label_frequency(label_frequency, allow_one=False)
    rng = check_random_state(random_state)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)

    n_labeled = n_labeled_positives(c, positives.size)
    if n_labeled == positives.size:
        raise ValueError(
            f"label_frequency={c} labels all {positives.size} positives, leaving none for the "
            "unlabeled set; use a smaller label_frequency."
        )
    labeled = rng.choice(positives, n_labeled, replace=False)
    unlabeled_positives = np.setdiff1d(positives, labeled)
    # As many negatives as keep the class prior of y in the unlabeled set, rounded half up.
    n_unlabeled_negatives = math.floor(
        unlabeled_positives.size * negatives.size / positives.size + 0.5
    )
    if n_unlabeled_negatives == 0:
        raise ValueError(
            "Too few negatives to keep the class prior in the unlabeled set; use a smaller "
            "label_frequency or a larger dataset."
        )
    unlabeled_negatives = rng.choice(negatives, n_unlabeled_negatives, replace=False)

    rows = np.sort(np.concatenate([labeled, unlabeled_positives, unlabeled_negatives]))
    s = labels_from_indices(y.size, labeled)
    return _safe_indexing(X, rows), s[rows], y[rows]
