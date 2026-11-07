"""Validation of labels, PU labels and class priors."""

from __future__ import annotations

import numbers
import warnings

import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import column_or_1d

_PU_LABELS_HINT = "PU labels must be 1 (labeled positive) or 0 (unlabeled)"


def check_true_labels(y, *, input_name: str = "y") -> np.ndarray:
    """Validate fully observed binary labels, with both classes present.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True labels: 1 for positive, 0 for negative. Booleans are accepted.
    input_name : str, default="y"
        Name of the argument, used in error messages.

    Returns
    -------
    y : ndarray of shape (n_samples,), dtype int64
        The validated labels.

    Raises
    ------
    ValueError
        If the labels are not 0/1, or if one of the classes is missing.
    """
    y = column_or_1d(y, warn=True)
    if y.size == 0:
        raise ValueError(f"{input_name} is empty.")
    values = np.unique(y)
    if y.dtype.kind not in "biuf" or not np.isin(values, (0, 1)).all():
        raise ValueError(
            f"{input_name} must hold the true binary labels: 1 for positive, 0 for negative, "
            f"got values {values[:10].tolist()}. Map other encodings (such as -1/1) to 0/1 first."
        )
    if values.size < 2:
        missing = "positive" if values[0] == 0 else "negative"
        raise ValueError(f"{input_name} has no {missing} sample; both classes are needed.")
    return y.astype(np.int64)


def check_pu_labels(s) -> np.ndarray:
    """Validate PU labels.

    Parameters
    ----------
    s : array-like of shape (n_samples,)
        PU labels: ``1`` for a labeled positive, ``0`` for an unlabeled sample.
        Booleans are accepted (``True`` means labeled positive).

    Returns
    -------
    s : ndarray of shape (n_samples,), dtype int64
        The validated labels.

    Raises
    ------
    ValueError
        If the labels are not 0/1, or if there is no labeled positive or no unlabeled sample.
    """
    s = column_or_1d(s, warn=True)
    if s.size == 0:
        raise ValueError("PU labels are empty.")
    # Same error wording as scikit-learn for non-binary targets.
    target_type = type_of_target(s, input_name="s", raise_unknown=True)
    if target_type == "continuous":
        raise ValueError(f"Unknown label type: continuous. {_PU_LABELS_HINT}.")

    values = np.unique(s)
    if target_type == "multiclass":
        raise ValueError(
            f"Only binary classification is supported. {_PU_LABELS_HINT}, got values "
            f"{values[:10].tolist()}."
        )
    if s.dtype.kind not in "biuf" or not np.isin(values, (0, 1)).all():
        raise ValueError(
            f"{_PU_LABELS_HINT}, got values {values[:10].tolist()}. If unlabeled samples are "
            "encoded as -1, map them to 0."
        )
    if values.size < 2:
        missing = "labeled positive (s == 1)" if values[0] == 0 else "unlabeled sample (s == 0)"
        raise ValueError(f"PU labels contain only one class: there is no {missing}.")
    return s.astype(np.int64)


def check_prior(prior, s=None) -> float:
    r"""Validate a class prior :math:`\pi = P(y = 1)`.

    Parameters
    ----------
    prior : float
        The class prior, in the open interval ``(0, 1)``.
    s : array-like of shape (n_samples,), default=None
        Validated PU labels. When given, a warning is raised if ``prior`` is lower than the
        fraction of labeled positives, which is impossible when the labeled positives are
        drawn from the same sample as the unlabeled data.

    Returns
    -------
    prior : float
        The validated prior.

    Raises
    ------
    TypeError
        If ``prior`` is not a real number.
    ValueError
        If ``prior`` is not in ``(0, 1)``.
    """
    if isinstance(prior, bool) or not isinstance(prior, numbers.Real):
        raise TypeError(f"prior must be a float in (0, 1), got {prior!r}.")
    prior = float(prior)
    if not 0.0 < prior < 1.0:
        raise ValueError(f"prior must be in the open interval (0, 1), got {prior}.")
    if s is not None:
        labeled_fraction = float(np.mean(s))
        if prior < labeled_fraction:
            warnings.warn(
                f"prior={prior} is lower than the fraction of labeled positives "
                f"({labeled_fraction:.4f}). When the labeled positives come from the same sample "
                "as the unlabeled data the prior cannot be lower than that fraction: check that "
                "you passed the class prior P(y=1) and not the label frequency P(s=1 | y=1).",
                UserWarning,
                stacklevel=2,
            )
    return prior


def threshold_from_prior(scores, prior: float) -> float:
    """Return the threshold that labels the top ``prior`` fraction of ``scores`` positive.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Scores where higher means more likely positive (e.g. positive-class probabilities).
    prior : float
        Fraction of samples to label positive, in ``(0, 1)``.

    Returns
    -------
    threshold : float
        The ``1 - prior`` quantile of ``scores``, so that a ``prior`` fraction of them lies
        above it (up to ties).
    """
    prior = check_prior(prior)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return float(np.quantile(scores, 1.0 - prior))
