r"""Metrics estimated from PU labels alone, for model selection without true labels.

They assume the single-training-set scenario with SCAR labels: the samples are drawn from
the population, and the labeled positives are a random subset of the positives among them.
Then the recall on the labeled positives estimates the true recall, and with the class prior
:math:`\pi` the other metrics follow [1]_.

All of them work as scikit-learn scorers, for example in
:class:`~sklearn.model_selection.GridSearchCV`, through
:func:`~sklearn.metrics.make_scorer`.

References
----------
.. [1] S. Jain, M. White and P. Radivojac. "Recovering true classifier performance in
   positive-unlabeled learning." AAAI 2017.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from pypu.metrics._common import check_predictions, check_scores
from pypu.utils.validation import check_prior, check_pu_labels


def _recall_and_positive_rate(s, y_pred):
    s = check_pu_labels(s)
    y_pred = check_predictions(y_pred, s)
    return y_pred[s == 1].mean(), y_pred.mean(), s


def lee_liu_score(s, y_pred) -> float:
    r"""Lee and Liu's criterion :math:`r^2 / P(\hat y = 1)`, which needs no class prior.

    :math:`r` is the recall on the labeled positives and :math:`P(\hat y = 1)` the fraction
    of samples predicted positive. Under SCAR the criterion equals
    :math:`\text{precision} \cdot \text{recall} / \pi`, so for a fixed dataset it ranks
    classifiers like the product of precision and recall [1]_. Higher is better; it is not
    bounded by 1 (a perfect classifier scores :math:`1 / \pi`).

    Parameters
    ----------
    s : array-like of shape (n_samples,)
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.
    y_pred : array-like of shape (n_samples,)
        Predicted labels, 1 for positive and 0 for negative.

    Returns
    -------
    score : float
        The criterion, or 0 if no sample is predicted positive.

    References
    ----------
    .. [1] W. S. Lee and B. Liu. "Learning with positive and unlabeled examples using
       weighted logistic regression." ICML 2003.

    Examples
    --------
    >>> from pypu.metrics import lee_liu_score
    >>> s = [1, 1, 0, 0, 0, 0, 0, 0]
    >>> y_pred = [1, 1, 1, 1, 0, 0, 0, 0]
    >>> lee_liu_score(s, y_pred)  # recall 1, half the samples predicted positive
    2.0
    """
    recall, positive_rate, _ = _recall_and_positive_rate(s, y_pred)
    return float(recall**2 / positive_rate) if positive_rate > 0 else 0.0


def pu_recall_score(s, y_pred) -> float:
    """Recall estimated from PU labels: the recall on the labeled positives.

    Under SCAR the labeled positives are a random sample of the positives, so this is an
    unbiased estimate of the true recall. No class prior is needed.

    Parameters
    ----------
    s : array-like of shape (n_samples,)
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.
    y_pred : array-like of shape (n_samples,)
        Predicted labels, 1 for positive and 0 for negative.

    Returns
    -------
    recall : float
        Fraction of the labeled positives predicted positive.

    Examples
    --------
    >>> from pypu.metrics import pu_recall_score
    >>> pu_recall_score([1, 1, 0, 0], [1, 0, 1, 0])
    0.5
    """
    recall, _, _ = _recall_and_positive_rate(s, y_pred)
    return float(recall)


def pu_precision_score(s, y_pred, prior) -> float:
    r"""Precision estimated from PU labels and the class prior.

    .. math:: \text{precision} = \frac{\pi \, r}{P(\hat y = 1)},

    where :math:`r` is the recall on the labeled positives and :math:`P(\hat y = 1)` the
    fraction of samples predicted positive.

    Parameters
    ----------
    s : array-like of shape (n_samples,)
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.
    y_pred : array-like of shape (n_samples,)
        Predicted labels, 1 for positive and 0 for negative.
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the samples, in ``(0, 1)``.

    Returns
    -------
    precision : float
        The estimate, clipped to ``[0, 1]``; 0 if no sample is predicted positive.

    Examples
    --------
    >>> from pypu.metrics import pu_precision_score
    >>> s = [1, 1, 0, 0, 0, 0, 0, 0]
    >>> y_pred = [1, 1, 1, 1, 0, 0, 0, 0]
    >>> pu_precision_score(s, y_pred, prior=0.25)  # 2 positives expected among 4 predicted
    0.5
    """
    recall, positive_rate, s = _recall_and_positive_rate(s, y_pred)
    prior = check_prior(prior, s)
    if positive_rate == 0:
        return 0.0
    return float(np.clip(prior * recall / positive_rate, 0.0, 1.0))


def pu_f1_score(s, y_pred, prior) -> float:
    r"""F1 score estimated from PU labels and the class prior.

    .. math:: F_1 = \frac{2 \pi r}{\pi + P(\hat y = 1)},

    where :math:`r` is the recall on the labeled positives and :math:`P(\hat y = 1)` the
    fraction of samples predicted positive.

    Parameters
    ----------
    s : array-like of shape (n_samples,)
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.
    y_pred : array-like of shape (n_samples,)
        Predicted labels, 1 for positive and 0 for negative.
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the samples, in ``(0, 1)``.

    Returns
    -------
    f1 : float
        The estimate, clipped to ``[0, 1]``.

    Examples
    --------
    >>> from pypu.metrics import pu_f1_score
    >>> s = [1, 1, 0, 0, 0, 0, 0, 0]
    >>> y_pred = [1, 1, 1, 1, 0, 0, 0, 0]
    >>> round(pu_f1_score(s, y_pred, prior=0.25), 4)
    0.6667
    """
    recall, positive_rate, s = _recall_and_positive_rate(s, y_pred)
    prior = check_prior(prior, s)
    return float(np.clip(2 * prior * recall / (prior + positive_rate), 0.0, 1.0))


def pu_roc_auc_score(s, y_score, prior) -> float:
    r"""ROC AUC estimated from PU labels and the class prior.

    The AUC of separating labeled from unlabeled samples underestimates the true AUC,
    because the unlabeled set contains positives. With :math:`\theta` the fraction of
    positives among the unlabeled samples, which follows from the prior and the counts,

    .. math:: \text{AUC} = \frac{\text{AUC}_{LU} - \theta / 2}{1 - \theta}.

    Parameters
    ----------
    s : array-like of shape (n_samples,)
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample.
    y_score : array-like of shape (n_samples,)
        Scores, higher meaning more likely positive, such as ``predict_proba(X)[:, 1]``.
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the samples, in ``(0, 1)``.

    Returns
    -------
    auc : float
        The estimate, clipped to ``[0, 1]``.

    Examples
    --------
    >>> from sklearn.metrics import roc_auc_score
    >>> from pypu.metrics import pu_roc_auc_score
    >>> y = [1, 1, 0, 1, 1, 0, 0, 0]  # unknown in practice
    >>> s = [1, 0, 0, 0, 1, 0, 0, 0]
    >>> y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1]
    >>> float(roc_auc_score(y, y_score)), float(roc_auc_score(s, y_score))
    (0.875, 0.75)
    >>> round(pu_roc_auc_score(s, y_score, prior=0.5), 4)
    0.875
    """
    s = check_pu_labels(s)
    y_score = check_scores(y_score, s)
    prior = check_prior(prior, s)
    n_labeled = int(s.sum())
    # Positives among the unlabeled: prior * n_samples - n_labeled, which is below
    # n_unlabeled because prior < 1. A prior below the labeled fraction (warned about by
    # check_prior) would make it negative.
    pollution = max((prior * s.size - n_labeled) / (s.size - n_labeled), 0.0)
    auc_lu = roc_auc_score(s, y_score)
    return float(np.clip((auc_lu - pollution / 2) / (1 - pollution), 0.0, 1.0))
