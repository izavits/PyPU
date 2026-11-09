"""Elkan's rebalancing of a classifier of labeled versus unlabeled samples."""

from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import has_fit_parameter

from pypu.base import BasePUClassifier
from pypu.utils.validation import check_prior


class RebalancedPU(BasePUClassifier):
    r"""A classifier of labeled versus unlabeled samples, rebalanced with the label frequency.

    Under SCAR, a classifier trained to separate labeled from unlabeled samples (a
    non-traditional classifier) classifies like a classifier of the true labels if its
    threshold :math:`\tau` is moved to :math:`c \tau`, with :math:`c = P(s = 1) / \pi` the
    label frequency. Rebalancing [1]_ weights the samples instead, so that the classifier
    trained on the weighted data classifies at the threshold :math:`\tau` itself: labeled
    samples get the weight :math:`w^+ = 1 - c \tau` and unlabeled samples
    :math:`w^- = (1 - \tau) c` [2]_. With :math:`\tau = 0.5` these are :math:`1 - c / 2` and
    :math:`c / 2` (BiasMC); with a class prior of 0.5 as well, they are :math:`P(s = 0)` and
    :math:`P(s = 1)`, the weights of Lee and Liu's weighted logistic regression.

    Rebalancing only targets classification at :math:`\tau`: ``predict`` thresholds at
    ``threshold``, but ``predict_proba`` is not an estimate of :math:`P(y = 1 \mid x)`.

    Parameters
    ----------
    estimator : classifier, default=None
        A scikit-learn classifier implementing ``predict_proba`` and accepting
        ``sample_weight`` in ``fit``. It is cloned before fitting. ``None`` uses
        :class:`~sklearn.linear_model.LogisticRegression`.
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the training data, in ``(0, 1)``, which gives
        the label frequency. Required; the default ``None`` raises an error in ``fit``.
    threshold : float, default=0.5
        The target probability threshold :math:`\tau`, in ``(0, 1)``.

    Attributes
    ----------
    estimator_ : classifier
        The classifier fitted on the weighted samples.
    label_frequency_ : float
        The label frequency :math:`c = P(s = 1) / \pi`.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``: ``threshold``.

    See Also
    --------
    ElkanNoto : Divides the non-traditional classifier's probabilities by the label frequency
        instead, which estimates :math:`P(y = 1 \mid x)`.
    NaivePU : The non-traditional classifier without weights.

    References
    ----------
    .. [1] C. Elkan. "The foundations of cost-sensitive learning." IJCAI 2001.
    .. [2] J. Bekker and J. Davis. "Learning from positive and unlabeled data: a survey."
       Machine Learning, 109, 719-760 (2020).

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import RebalancedPU
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = RebalancedPU(prior=y_train.mean()).fit(X_train, s_train)
    >>> clf.predict(X_test).shape
    (250,)
    """

    def __init__(self, estimator=None, prior=None, threshold=0.5):
        self.estimator = estimator
        self.prior = prior
        self.threshold = threshold

    def _fit(self, X, s):
        if self.prior is None:
            raise ValueError(
                "RebalancedPU needs the class prior: pass prior=P(y=1), the fraction of "
                "positives in the training data."
            )
        prior = check_prior(self.prior, s)
        tau = self.threshold
        if isinstance(tau, bool) or not isinstance(tau, numbers.Real) or not 0 < tau < 1:
            raise ValueError(f"threshold must be a number in (0, 1), got {tau!r}.")
        estimator = LogisticRegression() if self.estimator is None else clone(self.estimator)
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                f"estimator must implement predict_proba; {type(estimator).__name__} does not."
            )
        if not has_fit_parameter(estimator, "sample_weight"):
            raise TypeError(
                f"estimator must accept sample_weight in fit; {type(estimator).__name__} does not."
            )
        c = min(float(np.mean(s)) / prior, 1.0)
        sample_weight = np.where(s == 1, 1 - c * tau, (1 - tau) * c)
        self.label_frequency_ = c
        self.estimator_ = estimator.fit(X, s, sample_weight=sample_weight)

    def _predict_proba(self, X):
        positive_column = np.flatnonzero(self.estimator_.classes_ == 1)[0]
        return self.estimator_.predict_proba(X)[:, positive_column]

    def _compute_threshold(self, X, s):
        return float(self.threshold)
