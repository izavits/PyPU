"""Metrics that evaluate scores against the true labels of a test set."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

from pypu.metrics._common import check_scores
from pypu.utils.validation import check_true_labels, threshold_from_prior


def _positives_in_top_k(y_true, y_score, k):
    """Return the expected number of positives among the ``k`` highest scores.

    Scores tied at the cut-off count in proportion, as if the ties were broken at random.
    """
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or not 1 <= k <= y_true.size:
        raise ValueError(f"k must be an integer in [1, {y_true.size}], got {k!r}.")
    cutoff = -np.partition(-y_score, k - 1)[k - 1]  # the k-th highest score
    above = y_score > cutoff
    tied = y_score == cutoff
    n_from_tied = k - np.count_nonzero(above)
    return y_true[above].sum() + n_from_tied * y_true[tied].mean()


def precision_at_k(y_true, y_score, k=None) -> float:
    """Fraction of positives among the ``k`` highest-scored samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    y_score : array-like of shape (n_samples,)
        Scores, higher meaning more likely positive, such as ``predict_proba(X)[:, 1]``.
    k : int, default=None
        Number of top-ranked samples. ``None`` uses the number of positives, which gives the
        R-precision (see :func:`r_precision`).

    Returns
    -------
    precision : float
        Precision among the top ``k``. When samples tied at the cut-off do not all fit in the
        top ``k``, each counts in proportion, as if the ties were broken at random.

    See Also
    --------
    recall_at_k : Fraction of all positives found in the top ``k``.

    Examples
    --------
    >>> from pypu.metrics import precision_at_k
    >>> y_true = [1, 0, 1, 0, 0]
    >>> y_score = [0.9, 0.8, 0.7, 0.2, 0.1]
    >>> precision_at_k(y_true, y_score, k=3)
    0.6666666666666666
    """
    y_true = check_true_labels(y_true, input_name="y_true")
    y_score = check_scores(y_score, y_true)
    if k is None:
        k = int(y_true.sum())
    return float(_positives_in_top_k(y_true, y_score, k) / k)


def recall_at_k(y_true, y_score, k=None) -> float:
    """Fraction of all positives that are among the ``k`` highest-scored samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    y_score : array-like of shape (n_samples,)
        Scores, higher meaning more likely positive, such as ``predict_proba(X)[:, 1]``.
    k : int, default=None
        Number of top-ranked samples. ``None`` uses the number of positives, in which case
        recall equals precision.

    Returns
    -------
    recall : float
        Recall within the top ``k``, with ties at the cut-off counted in proportion as in
        :func:`precision_at_k`.

    Examples
    --------
    >>> from pypu.metrics import recall_at_k
    >>> y_true = [1, 0, 1, 0, 0]
    >>> y_score = [0.9, 0.8, 0.7, 0.2, 0.1]
    >>> recall_at_k(y_true, y_score, k=2)
    0.5
    """
    y_true = check_true_labels(y_true, input_name="y_true")
    y_score = check_scores(y_score, y_true)
    n_positive = int(y_true.sum())
    if k is None:
        k = n_positive
    return float(_positives_in_top_k(y_true, y_score, k) / n_positive)


def r_precision(y_true, y_score) -> float:
    """Precision among the top ``R`` samples, where ``R`` is the number of positives.

    Ranking the test samples by score, R-precision is the fraction of positives among the
    first ``R``. At that cut-off precision equals recall. It needs no threshold, which makes
    it suitable for PU methods whose scores are not calibrated probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    y_score : array-like of shape (n_samples,)
        Scores, higher meaning more likely positive, such as ``predict_proba(X)[:, 1]``.

    Returns
    -------
    r_precision : float
        Equal to ``precision_at_k(y_true, y_score, k=None)``. Without ties at the cut-off it
        matches ``r_precision_at_n`` from the iFPU and iFPNU experiments, which broke ties
        arbitrarily.

    Examples
    --------
    >>> from pypu.metrics import r_precision
    >>> y_true = [1, 0, 1, 0, 0]
    >>> y_score = [0.9, 0.8, 0.7, 0.2, 0.1]
    >>> r_precision(y_true, y_score)  # one of the top two is positive
    0.5
    """
    return precision_at_k(y_true, y_score, k=None)


def f1_at_prior(y_true, y_score, prior, *, average="binary"):
    r"""F1 score when the top ``prior`` fraction of the samples is predicted positive.

    The threshold is the ``1 - prior`` quantile of ``y_score`` itself, so it uses the class
    prior but no labels. This evaluates methods whose scores rank well but are not calibrated
    probabilities, for which a fixed threshold such as 0.5 would be arbitrary.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    y_score : array-like of shape (n_samples,)
        Scores, higher meaning more likely positive, such as ``predict_proba(X)[:, 1]``.
    prior : float
        Class prior :math:`\pi = P(y = 1)`, in ``(0, 1)``: the fraction of samples to
        predict positive.
    average : {"binary", "macro", "weighted", "micro"} or None, default="binary"
        As in :func:`sklearn.metrics.f1_score`: ``"binary"`` is the F1 of the positive class,
        ``"macro"`` the mean of the F1 of both classes, and ``None`` returns
        ``[F1 of the negative class, F1 of the positive class]``.

    Returns
    -------
    f1 : float or ndarray of shape (2,)
        The F1 score, or the per-class scores when ``average=None``. With
        ``average="binary"``, ``"macro"`` and ``None`` it reproduces ``f1_pos``, ``f1_macro``
        and ``[f1_neg, f1_pos]`` from ``f1_at_prior_threshold`` in the iFPU and iFPNU
        experiments.

    Examples
    --------
    >>> from pypu.metrics import f1_at_prior
    >>> y_true = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    >>> f1_at_prior(y_true, y_score, prior=0.2)  # predicts the top 2 positive
    0.5
    """
    y_true = check_true_labels(y_true, input_name="y_true")
    y_score = check_scores(y_score, y_true)
    y_pred = (y_score >= threshold_from_prior(y_score, prior)).astype(np.int64)
    if average == "binary":
        return float(f1_score(y_true, y_pred, zero_division=0))
    f1 = f1_score(y_true, y_pred, labels=[0, 1], average=average, zero_division=0)
    return f1 if average is None else float(f1)


def max_f1_score(y_true, y_score) -> float:
    """Highest F1 score of the positive class over all thresholds of ``y_score``.

    An optimistic measure, since the threshold is chosen with the test labels; it shows how
    good the ranking could be at its best cut-off.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    y_score : array-like of shape (n_samples,)
        Scores, higher meaning more likely positive, such as ``predict_proba(X)[:, 1]``.

    Returns
    -------
    f1 : float
        The maximum F1 over the precision-recall curve, the ``f1`` reported by the iFPU and
        iFPNU experiments.

    Examples
    --------
    >>> from pypu.metrics import max_f1_score
    >>> y_true = [1, 0, 1, 0, 0]
    >>> y_score = [0.9, 0.8, 0.7, 0.2, 0.1]
    >>> max_f1_score(y_true, y_score)  # at threshold 0.7: precision 2/3, recall 1
    0.8
    """
    y_true = check_true_labels(y_true, input_name="y_true")
    y_score = check_scores(y_score, y_true)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    return float(f1.max())
