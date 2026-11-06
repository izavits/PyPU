"""Labeling mechanisms: turn the true labels of a binary dataset into PU labels.

Each function takes the true labels ``y`` (1 positive, 0 negative) and returns PU labels
``s`` of the same length: ``s = 1`` for the positives chosen to be labeled and ``s = 0`` for
everything else, so the unlabeled set mixes the remaining positives with all negatives. Only
positives are ever labeled.

Apart from :func:`label_sar_propensity`, which draws each label independently from a known
propensity, the functions label exactly ``round(label_frequency * n_positive)`` positives.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_consistent_length

from pypu.datasets._common import (
    check_label_frequency,
    check_positive_int,
    check_true_labels,
    labels_from_indices,
    n_labeled_positives,
    weighted_choice,
)


def _check_X_y(X, y):
    X = check_array(X, dtype=[np.float64, np.float32])
    y = check_true_labels(y)
    check_consistent_length(X, y)
    return X, y


def _check_option(value, name: str, options: tuple[str, ...]) -> str:
    if value not in options:
        raise ValueError(f"{name} must be one of {list(options)}, got {value!r}.")
    return value


def _legacy_hidden(rng, n_positive: int, label_frequency: float, weights=None) -> np.ndarray:
    """Positives hidden by the old ``create_pu_data``, as indices into the positives.

    The old function took the fraction of positives to hide (``ratio``) and drew them with
    ``rng.choice`` in one call; this repeats that call exactly.
    """
    # Recover the float the old code was given: 1 - 0.7 is 0.30000000000000004, not 0.3, and
    # ceil() would turn that into one extra hidden positive.
    ratio = round(1.0 - label_frequency, 12)
    size = math.ceil(ratio * n_positive)
    if weights is not None:
        size = min(size, np.count_nonzero(weights))
    return rng.choice(range(n_positive), size, replace=False, p=weights)


def label_scar(y, *, label_frequency=0.5, legacy=False, random_state=None):
    r"""Label positives uniformly at random (SCAR).

    Under the *selected completely at random* assumption every positive has the same
    probability of being labeled, whatever its features [1]_. This is the assumption behind
    most PU learning methods.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    label_frequency : float, default=0.5
        Fraction of the positives to label, :math:`c = P(s = 1 \mid y = 1)`, in ``(0, 1]``.
    legacy : bool, default=False
        Reproduce the labels of the ``create_pu_data`` function used in the iFPU and iFPNU
        experiments, called with ``ratio=1 - label_frequency``, ``label_assumption="SCAR"``
        and the same ``random_state``.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator, for reproducible labels.

    Returns
    -------
    s : ndarray of shape (n_samples,), dtype int64
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.

    See Also
    --------
    label_sar_knn, label_sar_propensity, label_sar_cluster, label_sar_confidence :
        Labeling mechanisms where the chance of being labeled depends on the features.

    References
    ----------
    .. [1] J. Bekker and J. Davis. "Learning from positive and unlabeled data: a survey."
       Machine Learning, 2020.

    Examples
    --------
    >>> from pypu.datasets import label_scar
    >>> y = [1, 1, 1, 1, 0, 0, 0, 0]
    >>> label_scar(y, label_frequency=0.5, random_state=0).tolist()
    [0, 0, 1, 1, 0, 0, 0, 0]
    """
    y = check_true_labels(y)
    c = check_label_frequency(label_frequency)
    rng = check_random_state(random_state)
    positives = np.flatnonzero(y == 1)
    if legacy:
        labeled = np.delete(positives, _legacy_hidden(rng, positives.size, c))
    else:
        labeled = rng.choice(positives, n_labeled_positives(c, positives.size), replace=False)
    return labels_from_indices(y.size, labeled)


def label_sar_knn(
    X, y, *, label_frequency=0.5, n_neighbors=None, bias="near", legacy=False, random_state=None
):
    r"""Label positives according to their distance from the negatives (SAR).

    For each positive, compute the mean Euclidean distance to its ``n_neighbors`` nearest
    negatives. Positives are then drawn without replacement with probability proportional to
    that distance:

    - ``bias="near"``: the draws pick the positives to *hide*. Positives far from the
      negatives tend to stay unlabeled, so the labeled ones lie mostly near the class
      boundary. This is the mechanism of the original experiments.
    - ``bias="far"``: the draws pick the positives to *label*, so the labeled ones lie mostly
      far from the negatives, where they are easy to recognise.

    Positives that coincide with negatives (distance 0) are drawn only once all others have
    been. Distances are computed on ``X`` as given, so scale the features first if their
    units differ.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features.
    y : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    label_frequency : float, default=0.5
        Fraction of the positives to label, :math:`c = P(s = 1 \mid y = 1)`, in ``(0, 1]``.
    n_neighbors : int, default=None
        Number of nearest negatives to average over. ``None`` uses
        ``floor(sqrt(n_samples))``, capped at the number of negatives.
    bias : {"near", "far"}, default="near"
        Whether the labeled positives lie mostly near the negatives or far from them.
    legacy : bool, default=False
        Reproduce the labels of the ``create_pu_data`` function used in the iFPU and iFPNU
        experiments, called with ``ratio=1 - label_frequency``, ``label_assumption="SAR"``
        and the same ``n_neighbors`` and ``random_state``. That function averaged the
        distance to the first ``n_neighbors`` negatives in row order rather than to the
        nearest ones, and may label more positives than requested; only ``bias="near"`` is
        available.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator, for reproducible labels.

    Returns
    -------
    s : ndarray of shape (n_samples,), dtype int64
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.

    Examples
    --------
    >>> import numpy as np
    >>> from pypu.datasets import label_sar_knn
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    >>> y = [0, 0, 0, 0, 1, 1, 1, 1]
    >>> s = label_sar_knn(X, y, label_frequency=0.5, n_neighbors=1, random_state=0)
    >>> s.tolist()  # here the labels went to the two positives nearest the negatives
    [0, 0, 0, 0, 1, 1, 0, 0]
    """
    X, y = _check_X_y(X, y)
    c = check_label_frequency(label_frequency)
    bias = _check_option(bias, "bias", ("near", "far"))
    rng = check_random_state(random_state)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if n_neighbors is None:
        n_neighbors = min(math.floor(math.sqrt(y.size)), negatives.size)
    elif check_positive_int(n_neighbors, "n_neighbors") > negatives.size:
        raise ValueError(
            f"n_neighbors={n_neighbors} is larger than the number of negatives ({negatives.size})."
        )

    if legacy:
        if bias != "near":
            raise ValueError("legacy=True reproduces the old code, which only had bias='near'.")
        distances = pairwise_distances(X[positives], X[negatives[:n_neighbors]])
        weights = distances.mean(axis=1)
        hidden = _legacy_hidden(rng, positives.size, c, weights / weights.sum())
        return labels_from_indices(y.size, np.delete(positives, hidden))

    distances, _ = (
        NearestNeighbors(n_neighbors=n_neighbors).fit(X[negatives]).kneighbors(X[positives])
    )
    weights = distances.mean(axis=1)
    n_labeled = n_labeled_positives(c, positives.size)
    if bias == "far":
        labeled = positives[weighted_choice(rng, weights, n_labeled)]
    else:
        labeled = np.delete(positives, weighted_choice(rng, weights, positives.size - n_labeled))
    return labels_from_indices(y.size, labeled)


def _check_features(features, n_features: int) -> np.ndarray:
    if features is None:
        return np.arange(n_features)
    columns = np.atleast_1d(np.asarray(features))
    if columns.ndim != 1 or columns.size == 0 or columns.dtype.kind not in "iu":
        raise ValueError(f"features must be a non-empty list of column indices, got {features!r}.")
    if columns.min() < 0 or columns.max() >= n_features:
        raise ValueError(
            f"features must be column indices in [0, {n_features - 1}], got {columns.tolist()}."
        )
    return columns


def _standardize(values, reference):
    """Center and scale ``values`` by the mean and standard deviation of ``reference``."""
    std = reference.std(axis=0)
    return (values - reference.mean(axis=0)) / np.where(std > 0, std, 1.0)


def _solve_intercept(logits, target: float) -> float:
    """Intercept ``a`` such that the mean of ``expit(a + logits)`` equals ``target``."""
    center = logit(target)
    # At `low` every term is below `target`, at `high` every term is above it.
    low = center - logits.max() - 1.0
    high = center - logits.min() + 1.0
    return brentq(lambda a: expit(a + logits).mean() - target, low, high, xtol=1e-12)


def label_sar_propensity(
    X,
    y,
    *,
    label_frequency=0.5,
    features=None,
    coef=None,
    strength=1.0,
    return_propensity=False,
    random_state=None,
):
    r"""Label each positive independently with a known, feature-dependent propensity (SAR).

    The propensity :math:`e(x) = P(s = 1 \mid y = 1, x)` is a logistic function of the chosen
    features [1]_:

    .. math:: e(x) = \sigma(a + \text{strength} \cdot u(x)),

    where :math:`u(x)` is the ``coef``-weighted sum of the selected features, each
    standardized over the positives, and rescaled to unit standard deviation over the
    positives. The intercept :math:`a` is set so that the mean propensity of the positives
    equals ``label_frequency``. Each positive is then labeled with probability
    :math:`e(x)`, so the realised fraction of labeled positives varies around
    ``label_frequency``. Because the true propensity is known, this mechanism suits the
    evaluation of methods that model it, such as SAR-EM.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features.
    y : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    label_frequency : float, default=0.5
        Mean propensity of the positives, :math:`c = P(s = 1 \mid y = 1)`, in ``(0, 1)``.
    features : array-like of int, default=None
        Indices of the columns the propensity depends on. ``None`` uses all columns.
    coef : array-like of shape (n_selected_features,), default=None
        Relative weight of each selected feature; negative weights make larger values less
        likely to be labeled. ``None`` gives every selected feature weight 1.
    strength : float, default=1.0
        How strongly the propensity depends on the features: the change in the log-odds of
        being labeled per standard deviation of :math:`u(x)`. ``0`` gives SCAR labels.
    return_propensity : bool, default=False
        Also return the propensity of every sample.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator, for reproducible labels.

    Returns
    -------
    s : ndarray of shape (n_samples,), dtype int64
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.
    propensity : ndarray of shape (n_samples,)
        :math:`e(x)` for every sample, negatives included. Only returned when
        ``return_propensity=True``.

    References
    ----------
    .. [1] J. Bekker, P. Robberechts and J. Davis. "Beyond the selected completely at random
       assumption for learning from positive and unlabeled data." ECML PKDD 2019.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import label_sar_propensity
    >>> X, y = make_classification(n_samples=1000, random_state=0)
    >>> s, e = label_sar_propensity(
    ...     X, y, label_frequency=0.3, features=[0, 1], return_propensity=True, random_state=0
    ... )
    >>> round(float(e[y == 1].mean()), 6)
    0.3
    """
    X, y = _check_X_y(X, y)
    c = check_label_frequency(label_frequency, allow_one=False)
    columns = _check_features(features, X.shape[1])
    weights = np.ones(columns.size) if coef is None else np.asarray(coef, dtype=np.float64)
    if weights.shape != (columns.size,):
        raise ValueError(
            f"coef must have one weight per selected feature ({columns.size}), got shape "
            f"{weights.shape}."
        )
    if (
        isinstance(strength, bool)
        or not isinstance(strength, numbers.Real)
        or not math.isfinite(strength)
    ):
        raise ValueError(f"strength must be a finite real number, got {strength!r}.")
    rng = check_random_state(random_state)

    is_positive = y == 1
    selected = X[:, columns].astype(np.float64)
    score = _standardize(selected, selected[is_positive]) @ weights
    score = _standardize(score, score[is_positive])
    logits = float(strength) * score
    propensity = expit(_solve_intercept(logits[is_positive], c) + logits)

    positives = np.flatnonzero(is_positive)
    labeled = positives[rng.random_sample(positives.size) < propensity[positives]]
    if labeled.size == 0:
        raise ValueError(
            f"No positive was labeled with label_frequency={c}; use a larger label_frequency "
            "or another random_state."
        )
    s = labels_from_indices(y.size, labeled)
    return (s, propensity) if return_propensity else s


def label_sar_cluster(X, y, *, label_frequency=0.5, n_clusters=4, random_state=None):
    r"""Label positives from only some clusters of the positive class (SAR).

    The positives are split into ``n_clusters`` groups with k-means. Clusters are taken in
    random order until they hold enough positives, and the labeled positives are drawn
    uniformly from those clusters. The remaining clusters get no labels at all, so some kinds
    of positives never appear in the labeled set, as when only one sub-population of the
    positive class gets recorded.

    Distances are computed on ``X`` as given, so scale the features first if their units
    differ.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features.
    y : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    label_frequency : float, default=0.5
        Fraction of the positives to label, :math:`c = P(s = 1 \mid y = 1)`, in ``(0, 1]``.
    n_clusters : int, default=4
        Number of k-means clusters of the positives. ``1`` gives SCAR labels.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator, for reproducible clusters and labels.

    Returns
    -------
    s : ndarray of shape (n_samples,), dtype int64
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.

    Examples
    --------
    >>> import numpy as np
    >>> from pypu.datasets import label_sar_cluster
    >>> X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [10.0], [10.1], [10.2]])
    >>> y = [0, 0, 0, 1, 1, 1, 1, 1]
    >>> s = label_sar_cluster(X, y, label_frequency=0.4, n_clusters=2, random_state=0)
    >>> int(s.sum())
    2
    """
    X, y = _check_X_y(X, y)
    c = check_label_frequency(label_frequency)
    n_clusters = check_positive_int(n_clusters, "n_clusters")
    positives = np.flatnonzero(y == 1)
    if n_clusters > positives.size:
        raise ValueError(
            f"n_clusters={n_clusters} is larger than the number of positives ({positives.size})."
        )
    n_labeled = n_labeled_positives(c, positives.size)
    rng = check_random_state(random_state)

    clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=rng).fit_predict(X[positives])
    order = rng.permutation(n_clusters)
    sizes = np.bincount(clusters, minlength=n_clusters)[order]
    n_used = int(np.searchsorted(np.cumsum(sizes), n_labeled)) + 1
    candidates = positives[np.isin(clusters, order[:n_used])]
    return labels_from_indices(y.size, rng.choice(candidates, n_labeled, replace=False))


def label_sar_confidence(
    X, y, *, label_frequency=0.5, estimator=None, bias="easy", cv=5, random_state=None
):
    r"""Label positives according to how confidently a classifier recognises them (SAR).

    A classifier is trained on the true labels, and out-of-fold predictions give each
    positive a probability :math:`p` of being positive. Positives are then drawn without
    replacement with probability proportional to :math:`p` (``bias="easy"``: the labeled
    positives are mostly the obvious ones) or to :math:`1 - p` (``bias="hard"``: mostly the
    ambiguous ones).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features.
    y : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    label_frequency : float, default=0.5
        Fraction of the positives to label, :math:`c = P(s = 1 \mid y = 1)`, in ``(0, 1]``.
    estimator : classifier, default=None
        A scikit-learn classifier implementing ``predict_proba``. It is cloned before
        fitting. ``None`` uses :class:`~sklearn.linear_model.LogisticRegression`.
    bias : {"easy", "hard"}, default="easy"
        Whether the labeled positives are mostly the confidently or the poorly recognised
        ones.
    cv : int or cross-validation generator, default=5
        Cross-validation splitting for the out-of-fold predictions, as in
        :func:`~sklearn.model_selection.cross_val_predict`.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator for drawing the labels. A random ``estimator`` or
        ``cv`` takes its own ``random_state``.

    Returns
    -------
    s : ndarray of shape (n_samples,), dtype int64
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import label_sar_confidence
    >>> X, y = make_classification(n_samples=500, random_state=0)
    >>> s = label_sar_confidence(X, y, label_frequency=0.2, random_state=0)
    >>> int(s.sum()), int(y.sum())
    (50, 249)
    """
    X, y = _check_X_y(X, y)
    c = check_label_frequency(label_frequency)
    bias = _check_option(bias, "bias", ("easy", "hard"))
    estimator = LogisticRegression() if estimator is None else clone(estimator)
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            f"estimator must implement predict_proba; {type(estimator).__name__} does not."
        )
    rng = check_random_state(random_state)

    # Columns follow the sorted classes [0, 1].
    proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
    positives = np.flatnonzero(y == 1)
    weights = proba[positives] if bias == "easy" else 1.0 - proba[positives]
    labeled = positives[weighted_choice(rng, weights, n_labeled_positives(c, positives.size))]
    return labels_from_indices(y.size, labeled)
