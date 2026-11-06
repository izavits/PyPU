"""Summary statistics of PU data and conversions between them."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.utils.validation import check_consistent_length

from pypu.datasets._common import check_label_frequency, check_true_labels
from pypu.utils.validation import check_prior, check_pu_labels


def pu_summary(y, s) -> dict[str, int | float]:
    r"""Describe PU data whose true labels are known.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative.
    s : array-like of shape (n_samples,)
        PU labels: 1 for a labeled positive, 0 for an unlabeled sample. Every labeled sample
        must be positive.

    Returns
    -------
    summary : dict
        - ``n_samples``, ``n_positive``, ``n_labeled``, ``n_unlabeled``: counts.
        - ``prior``: class prior :math:`\pi = P(y = 1)`.
        - ``label_frequency``: fraction of the positives that are labeled,
          :math:`c = P(s = 1 \mid y = 1)`.
        - ``pollution``: fraction of positives among the unlabeled samples,
          :math:`P(y = 1 \mid s = 0)`.

    Examples
    --------
    >>> from pypu.datasets import pu_summary
    >>> y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    >>> s = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> pu_summary(y, s)  # doctest: +NORMALIZE_WHITESPACE
    {'n_samples': 10, 'n_positive': 4, 'n_labeled': 3, 'n_unlabeled': 7, 'prior': 0.4,
     'label_frequency': 0.75, 'pollution': 0.14285714285714285}
    """
    y = check_true_labels(y)
    s = check_pu_labels(s)
    check_consistent_length(y, s)
    if np.any(s > y):
        raise ValueError(
            f"{int(np.sum(s > y))} labeled samples (s == 1) are negative (y == 0); in PU data "
            "only positives are labeled."
        )
    is_positive = y == 1
    is_unlabeled = s == 0
    return {
        "n_samples": int(y.size),
        "n_positive": int(is_positive.sum()),
        "n_labeled": int(s.sum()),
        "n_unlabeled": int(is_unlabeled.sum()),
        "prior": float(is_positive.mean()),
        "label_frequency": float(s[is_positive].mean()),
        "pollution": float(y[is_unlabeled].mean()),
    }


def pollution_from_label_frequency(label_frequency, prior) -> float:
    r"""Fraction of positives among the unlabeled samples, given the label frequency.

    In the single-training-set scenario, labeling a fraction :math:`c` of the positives
    leaves a fraction

    .. math:: P(y = 1 \mid s = 0) = \frac{\pi (1 - c)}{1 - \pi c}

    of positives in the unlabeled set, where :math:`\pi` is the class prior. This is the
    expected value; with finite data the realised pollution differs slightly because the
    number of labeled positives is rounded.

    Parameters
    ----------
    label_frequency : float
        Fraction of the positives that are labeled, :math:`c`, in ``(0, 1]``.
    prior : float
        Class prior :math:`\pi = P(y = 1)`, in ``(0, 1)``.

    Returns
    -------
    pollution : float
        Fraction of positives among the unlabeled samples.

    See Also
    --------
    label_frequency_from_pollution : The inverse conversion.

    Examples
    --------
    >>> from pypu.datasets import pollution_from_label_frequency
    >>> round(pollution_from_label_frequency(0.5, prior=0.4), 4)
    0.25
    """
    c = check_label_frequency(label_frequency)
    prior = check_prior(prior)
    return prior * (1.0 - c) / (1.0 - prior * c)


def label_frequency_from_pollution(pollution, prior) -> float:
    r"""Label frequency that leaves a given fraction of positives among the unlabeled.

    The inverse of :func:`pollution_from_label_frequency`:

    .. math:: c = \frac{\pi - p}{\pi (1 - p)},

    where :math:`p` is the pollution and :math:`\pi` the class prior. Use it to make PU data
    with a chosen pollution: pass the result as ``label_frequency`` to a labeling function.

    Parameters
    ----------
    pollution : float
        Fraction of positives among the unlabeled samples, :math:`P(y = 1 \mid s = 0)`, in
        ``[0, prior)``.
    prior : float
        Class prior :math:`\pi = P(y = 1)`, in ``(0, 1)``.

    Returns
    -------
    label_frequency : float
        Fraction of the positives to label, :math:`c = P(s = 1 \mid y = 1)`.

    See Also
    --------
    pollution_from_label_frequency : The inverse conversion.

    Examples
    --------
    >>> from pypu.datasets import label_frequency_from_pollution
    >>> round(label_frequency_from_pollution(0.25, prior=0.4), 4)
    0.5
    """
    prior = check_prior(prior)
    if isinstance(pollution, bool) or not isinstance(pollution, numbers.Real):
        raise TypeError(f"pollution must be a float in [0, prior), got {pollution!r}.")
    pollution = float(pollution)
    if not 0.0 <= pollution < prior:
        raise ValueError(
            f"pollution must be in [0, prior) = [0, {prior}), got {pollution}: the unlabeled "
            "set cannot hold a larger share of positives than the whole dataset."
        )
    return (prior - pollution) / (prior * (1.0 - pollution))
