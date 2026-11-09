"""Naive baseline that treats every unlabeled sample as negative."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from pypu.base import BasePUClassifier
from pypu.utils.validation import check_prior, threshold_from_prior


class NaivePU(BasePUClassifier):
    r"""Naive PU baseline: a classifier trained with unlabeled samples as negatives.

    The wrapped classifier learns to separate labeled positives (``s == 1``) from unlabeled
    samples (``s == 0``), so it estimates :math:`P(s = 1 \mid x)` rather than
    :math:`P(y = 1 \mid x)`. Under the SCAR assumption (labeled positives are selected
    completely at random) the two differ by the constant factor
    :math:`c = P(s = 1 \mid y = 1)` [1]_: the ranking of samples, and therefore ROC AUC, is
    preserved, but the probabilities are too low and a ``0.5`` threshold under-predicts
    positives. Passing the class ``prior`` sets the threshold so that a ``prior`` fraction of
    the training samples is predicted positive.

    It is the usual lower bound when comparing PU learning methods.

    Parameters
    ----------
    estimator : classifier, default=None
        A scikit-learn classifier implementing ``predict_proba``. It is cloned before
        fitting. ``None`` uses :class:`~sklearn.linear_model.LogisticRegression`.
    prior : float, default=None
        Class prior :math:`\pi = P(y = 1)`. If given, :attr:`threshold_` is set so that a
        ``prior`` fraction of the training samples is predicted positive. Otherwise the
        threshold is ``0.5``.

    Attributes
    ----------
    estimator_ : classifier
        The fitted classifier.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``.

    References
    ----------
    .. [1] C. Elkan and K. Noto. "Learning classifiers from only positive and unlabeled
       data." KDD 2008.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from pypu.models import NaivePU
    >>> X, y = make_classification(n_samples=500, weights=[0.7], random_state=0)
    >>> rng = np.random.default_rng(0)
    >>> s = ((y == 1) & (rng.random(len(y)) < 0.5)).astype(int)  # label half the positives
    >>> clf = NaivePU(prior=y.mean()).fit(X, s)
    >>> clf.predict_proba(X).shape
    (500, 2)
    """

    def __init__(self, estimator=None, prior=None):
        self.estimator = estimator
        self.prior = prior

    def _fit(self, X, s):
        if self.prior is not None:
            check_prior(self.prior, s)
        estimator = LogisticRegression() if self.estimator is None else clone(self.estimator)
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                f"estimator must implement predict_proba; {type(estimator).__name__} does not."
            )
        self.estimator_ = estimator.fit(X, s)

    def _predict_proba(self, X):
        positive_column = np.flatnonzero(self.estimator_.classes_ == 1)[0]
        return self.estimator_.predict_proba(X)[:, positive_column]

    def _compute_threshold(self, X, s):
        if self.prior is None:
            return 0.5
        return threshold_from_prior(self._predict_proba(X), self.prior)
