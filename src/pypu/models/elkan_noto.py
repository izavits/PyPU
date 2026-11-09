"""The classifiers of Elkan and Noto, which correct a labeled-vs-unlabeled classifier."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import has_fit_parameter

from pypu.base import _EPS, BasePUClassifier
from pypu.prior._base import out_of_fold_scores
from pypu.utils.validation import check_prior


class ElkanNoto(BasePUClassifier):
    r"""Elkan and Noto's PU classifier, in its adjusted or its weighted form.

    A classifier :math:`g` trained to separate labeled from unlabeled samples estimates
    :math:`P(s = 1 \mid x)`. When the labeled positives are selected completely at random
    (SCAR), :math:`P(s = 1 \mid x) = c \, P(y = 1 \mid x)` with the label frequency
    :math:`c = P(s = 1 \mid y = 1)` [1]_. The model estimates :math:`c` (or derives it from
    ``prior``) and then estimates :math:`P(y = 1 \mid x)` in one of two ways:

    - ``weighted=False``: :math:`g(x) / c`, clipped to ``[0, 1]``.
    - ``weighted=True``: every unlabeled sample enters the training data twice, as a positive
      with weight :math:`w(x)` and as a negative with weight :math:`1 - w(x)`, where
      :math:`w(x) = P(y = 1 \mid x, s = 0) = \frac{1 - c}{c} \frac{g(x)}{1 - g(x)}`, clipped
      to ``[0, 1]``. Labeled positives get weight 1. A classifier trained on these weighted
      samples estimates :math:`P(y = 1 \mid x)` directly.

    Without ``prior``, :math:`c` is the mean of :math:`g` over the labeled positives (the
    ``"e1"`` estimator of :class:`~pypu.prior.ElkanNotoPrior`). It and the weights
    :math:`w(x)` are computed on out-of-fold scores, so that :math:`g` is never scored on the
    samples it was trained on. Both forms need :math:`g` to give calibrated probabilities.

    Parameters
    ----------
    estimator : classifier, default=None
        A scikit-learn classifier implementing ``predict_proba``, with calibrated
        probabilities. It is cloned before fitting. With ``weighted=True`` its ``fit`` must
        accept ``sample_weight``, and it is also the classifier trained on the weighted
        samples. ``None`` uses :class:`~sklearn.linear_model.LogisticRegression`.
    weighted : bool, default=False
        Whether to train a classifier on weighted unlabeled samples instead of dividing
        :math:`g` by :math:`c`.
    prior : float, default=None
        Class prior :math:`\pi = P(y = 1)` of the training samples. If given, the label
        frequency is :math:`c = P(s = 1) / \pi` instead of an estimate.
    cv : int or cross-validation generator, default=5
        Splitting for the out-of-fold scores, which are needed unless ``prior`` is given and
        ``weighted=False``. An int is the number of shuffled, stratified folds.
    random_state : int, RandomState instance or None, default=None
        Seed for shuffling the folds when ``cv`` is an int.

    Attributes
    ----------
    estimator_ : classifier
        With ``weighted=False``, :math:`g` trained on all the samples; with ``weighted=True``,
        the classifier trained on the weighted samples.
    label_frequency_ : float
        The label frequency :math:`c = P(s = 1 \mid y = 1)`, estimated or derived from
        ``prior``.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``; ``0.5``.

    See Also
    --------
    pypu.prior.ElkanNotoPrior : The estimators of :math:`c` and of the class prior.
    NaivePU : :math:`g` alone, without the correction.

    Notes
    -----
    With ``weighted=False``, samples with :math:`g(x) \geq c` all get probability 1. To keep
    ranking them, ``decision_function`` returns
    :math:`\operatorname{logit} g(x) - \operatorname{logit}(c / 2)`, which is positive exactly
    when the probability is above ``0.5``.

    References
    ----------
    .. [1] C. Elkan and K. Noto. "Learning classifiers from only positive and unlabeled
       data." KDD 2008.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import label_scar
    >>> from pypu.models import ElkanNoto
    >>> X, y = make_classification(n_samples=2000, weights=[0.7], class_sep=2, random_state=0)
    >>> s = label_scar(y, label_frequency=0.4, random_state=0)
    >>> clf = ElkanNoto(random_state=0).fit(X, s)
    >>> abs(clf.label_frequency_ - 0.4) < 0.03  # the true label frequency is 0.4
    True
    >>> float((clf.predict(X) == y).mean()) > 0.9
    True
    """

    def __init__(self, estimator=None, weighted=False, prior=None, cv=5, random_state=None):
        self.estimator = estimator
        self.weighted = weighted
        self.prior = prior
        self.cv = cv
        self.random_state = random_state

    def _fit(self, X, s):
        estimator = LogisticRegression() if self.estimator is None else clone(self.estimator)
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                f"estimator must implement predict_proba; {type(estimator).__name__} does not."
            )
        if self.weighted and not has_fit_parameter(estimator, "sample_weight"):
            raise TypeError(
                "weighted=True needs an estimator whose fit accepts sample_weight; "
                f"{type(estimator).__name__} does not."
            )
        labeled = s == 1
        scores = None
        if self.prior is None or self.weighted:
            scores = out_of_fold_scores(estimator, X, s, cv=self.cv, random_state=self.random_state)
        if self.prior is None:
            c = self._estimate_label_frequency(scores, labeled)
        else:
            c = min(float(labeled.mean()) / check_prior(self.prior, s), 1.0)
        self.label_frequency_ = c

        if not self.weighted:
            self.estimator_ = estimator.fit(X, s)
            return
        weight = _unlabeled_weights(scores[~labeled], c)
        X_unlabeled = X[~labeled]
        n_labeled, n_unlabeled = int(labeled.sum()), X_unlabeled.shape[0]
        X_fit = np.concatenate([X[labeled], X_unlabeled, X_unlabeled])
        y_fit = np.repeat([1, 1, 0], [n_labeled, n_unlabeled, n_unlabeled])
        sample_weight = np.concatenate([np.ones(n_labeled), weight, 1.0 - weight])
        keep = sample_weight > 0
        self.estimator_ = estimator.fit(X_fit[keep], y_fit[keep], sample_weight=sample_weight[keep])

    @staticmethod
    def _estimate_label_frequency(scores, labeled):
        c = float(scores[labeled].mean())
        n_labeled, n_samples = int(labeled.sum()), labeled.size
        # Below P(s = 1), c would imply a class prior above 1. Keep at least one unlabeled
        # sample negative, as the prior estimators do.
        lowest = n_labeled / (n_samples - 1)
        if not c >= lowest:
            warnings.warn(
                f"The estimated label frequency ({c:.4f}) implies a class prior of 1 or more; "
                "it is clipped. The classifier probably cannot tell the labeled positives "
                "from the unlabeled samples.",
                UserWarning,
                stacklevel=4,
            )
            c = lowest
        return c

    def _scores(self, X):
        positive_column = np.flatnonzero(self.estimator_.classes_ == 1)[0]
        return self.estimator_.predict_proba(X)[:, positive_column]

    def _predict_proba(self, X):
        if self.weighted:
            return self._scores(X)
        return np.clip(self._scores(X) / self.label_frequency_, 0.0, 1.0)

    def _decision_function(self, X):
        if self.weighted:
            return super()._decision_function(X)
        # Zero where g(x) / c = 1/2, and unlike the clipped probability it keeps increasing
        # for g(x) > c.
        scores = np.clip(self._scores(X), _EPS, 1.0 - _EPS)
        return logit(scores) - logit(self.label_frequency_ / 2)


def _unlabeled_weights(scores, c):
    """Return P(y = 1 | x, s = 0) for unlabeled samples with scores g(x), clipped to [0, 1]."""
    if c >= 1.0:
        return np.zeros_like(scores)
    with np.errstate(divide="ignore"):
        odds = scores / (1.0 - scores)
    return np.clip((1.0 - c) / c * odds, 0.0, 1.0)
