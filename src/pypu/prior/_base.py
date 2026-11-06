"""Base class of the class-prior estimators."""

from __future__ import annotations

import numbers
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.validation import check_consistent_length, validate_data

from pypu.utils.validation import check_pu_labels


class BasePriorEstimator(BaseEstimator, metaclass=ABCMeta):
    r"""Base class of the estimators of the class prior from PU data.

    ``fit(X, s)`` estimates three related quantities of the PU data, assuming the
    single-training-set scenario (the labeled positives are a SCAR subset of the positives in
    ``X``):

    - ``prior_``: the class prior :math:`\pi = P(y = 1)` of ``X``;
    - ``label_frequency_``: :math:`c = P(s = 1 \mid y = 1)`;
    - ``pollution_``: the fraction of positives among the unlabeled samples,
      :math:`P(y = 1 \mid s = 0)`.

    They satisfy ``prior_ * label_frequency_ == mean(s)``. In the case-control scenario, where
    the unlabeled set is a separate sample of the population, ``pollution_`` estimates the
    class prior of that population.

    Subclasses implement ``_estimate_pollution(scores, s)``, given out-of-fold scores of a
    classifier trained to separate labeled from unlabeled samples.
    """

    def __init__(self, estimator=None, cv=5, random_state=None):
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, s):
        """Estimate the class prior.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.
        s : array-like of shape (n_samples,)
            PU labels: 1 for a labeled positive, 0 for an unlabeled sample.

        Returns
        -------
        self : object
            The fitted estimator, with ``prior_``, ``label_frequency_`` and ``pollution_``.
        """
        X = validate_data(self, X, reset=True, dtype=[np.float64, np.float32])
        s = check_pu_labels(s)
        check_consistent_length(X, s)
        scores = self._out_of_fold_scores(X, s)
        pollution = float(self._estimate_pollution(scores, s))

        n_labeled = int(s.sum())
        n_unlabeled = s.size - n_labeled
        # At least one unlabeled sample stays negative, so that the prior is below 1.
        highest = (n_unlabeled - 1) / n_unlabeled
        if not pollution <= highest:
            warnings.warn(
                f"The estimated share of positives among the unlabeled samples ({pollution:.4f})"
                " implies a class prior of 1 or more; it is clipped. The classifier probably "
                "cannot tell the labeled positives from the unlabeled samples.",
                UserWarning,
                stacklevel=2,
            )
            pollution = highest
        n_positive = n_labeled + pollution * n_unlabeled
        self.pollution_ = pollution
        self.prior_ = float(n_positive / s.size)
        self.label_frequency_ = float(n_labeled / n_positive)
        return self

    def _out_of_fold_scores(self, X, s):
        """Out-of-fold estimates of P(s = 1 | x) for every sample."""
        estimator = LogisticRegression() if self.estimator is None else clone(self.estimator)
        return out_of_fold_scores(estimator, X, s, cv=self.cv, random_state=self.random_state)

    @abstractmethod
    def _estimate_pollution(self, scores, s) -> float:
        """Estimate the fraction of positives among the unlabeled samples."""


def out_of_fold_scores(estimator, X, s, *, cv, random_state):
    """Return out-of-fold estimates of P(s = 1 | x) for every sample.

    ``estimator`` is cloned for each fold. An int ``cv`` is the number of shuffled, stratified
    folds, seeded by ``random_state``.
    """
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            f"estimator must implement predict_proba; {type(estimator).__name__} does not."
        )
    if isinstance(cv, numbers.Integral) and not isinstance(cv, bool):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    # Columns follow the sorted classes [0, 1].
    return cross_val_predict(estimator, X, s, cv=cv, method="predict_proba")[:, 1]
