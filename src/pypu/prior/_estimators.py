"""Class-prior estimators: Elkan-Noto and Best Bin Estimation."""

from __future__ import annotations

import math
import numbers

import numpy as np

from pypu.prior._base import BasePriorEstimator


class ElkanNotoPrior(BasePriorEstimator):
    r"""Estimate the class prior with the method of Elkan and Noto.

    Under SCAR, a classifier :math:`g` that separates labeled from unlabeled samples
    estimates :math:`P(s = 1 \mid x) = c \, P(y = 1 \mid x)`. Where the classes do not
    overlap, :math:`g(x) = c` on the positives, which gives estimators of the label frequency
    :math:`c` [1]_, computed here on out-of-fold scores:

    - ``"e1"``: the mean of :math:`g` over the labeled positives;
    - ``"e2"``: the sum of :math:`g` over the labeled positives divided by its sum over all
      samples;
    - ``"e3"``: the maximum of :math:`g`.

    The prior follows as :math:`\pi = P(s = 1) / c`. When the classes overlap, ``"e1"`` and
    ``"e2"`` underestimate :math:`c` and so overestimate the prior; :class:`BBEPrior` copes
    better with overlap. :math:`g` must give calibrated probabilities.

    Parameters
    ----------
    method : {"e1", "e2", "e3"}, default="e1"
        Which estimator of :math:`c` to use; ``"e1"`` is the one the authors recommend.
    estimator : classifier, default=None
        A scikit-learn classifier implementing ``predict_proba``, trained to separate labeled
        from unlabeled samples. It is cloned for each fold. ``None`` uses
        :class:`~sklearn.linear_model.LogisticRegression`.
    cv : int or cross-validation generator, default=5
        Splitting for the out-of-fold scores. An int is the number of shuffled, stratified
        folds.
    random_state : int, RandomState instance or None, default=None
        Seed for shuffling the folds when ``cv`` is an int.

    Attributes
    ----------
    prior_ : float
        Estimated class prior :math:`\pi = P(y = 1)` of the training samples.
    label_frequency_ : float
        Estimated label frequency :math:`c = P(s = 1 \mid y = 1)`.
    pollution_ : float
        Estimated fraction of positives among the unlabeled samples. In the case-control
        scenario it estimates the class prior of the unlabeled population.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``, when ``X`` has string feature names.

    See Also
    --------
    BBEPrior : Best Bin Estimation, more robust when the classes overlap.

    References
    ----------
    .. [1] C. Elkan and K. Noto. "Learning classifiers from only positive and unlabeled
       data." KDD 2008.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import label_scar
    >>> from pypu.prior import ElkanNotoPrior
    >>> X, y = make_classification(n_samples=2000, weights=[0.7], class_sep=2, random_state=0)
    >>> s = label_scar(y, label_frequency=0.4, random_state=0)
    >>> estimator = ElkanNotoPrior(random_state=0).fit(X, s)
    >>> abs(estimator.prior_ - 0.3) < 0.03  # the true prior is 0.3
    True
    """

    def __init__(self, method="e1", estimator=None, cv=5, random_state=None):
        super().__init__(estimator=estimator, cv=cv, random_state=random_state)
        self.method = method

    def _estimate_pollution(self, scores, s):
        if self.method not in ("e1", "e2", "e3"):
            raise ValueError(f"method must be one of ['e1', 'e2', 'e3'], got {self.method!r}.")
        labeled = s == 1
        if self.method == "e1":
            c = scores[labeled].mean()
        elif self.method == "e2":
            total = scores.sum()
            c = scores[labeled].sum() / total if total > 0 else 0.0
        else:
            c = scores.max()
        if c <= 0:
            return math.inf
        n_labeled = labeled.sum()
        return n_labeled * (1 / c - 1) / (s.size - n_labeled)


class BBEPrior(BasePriorEstimator):
    r"""Estimate the class prior with Best Bin Estimation (BBE).

    A classifier is trained to separate labeled from unlabeled samples, and its out-of-fold
    scores rank all samples. For a cut-off :math:`t`, let :math:`q_p(t)` and :math:`q_u(t)`
    be the fractions of labeled and of unlabeled samples scoring at least :math:`t`. The top
    bin is mostly positive, so :math:`q_u(t) / q_p(t)` estimates the fraction of positives
    among the unlabeled samples. BBE takes the cut-off that minimizes an upper confidence
    bound of that ratio [1]_:

    .. math::

        \hat t = \arg\min_t \frac{q_u(t)}{q_p(t)} + \frac{1 + \gamma}{q_p(t)}
        \left( \sqrt{\frac{\log(4 / \delta)}{2 n_p}} + \sqrt{\frac{\log(4 / \delta)}{2 n_u}}
        \right),

    where :math:`n_p` and :math:`n_u` are the numbers of labeled and unlabeled samples, and
    estimates the fraction of positives among the unlabeled samples as
    :math:`q_u(\hat t) / q_p(\hat t)`. It needs a region of the feature space where (almost)
    all samples are positive, not separable classes, and the scores need not be calibrated
    probabilities.

    The estimate is only as good as the classifier's ranking. When the positives are not
    linearly separable from the rest, a flexible classifier such as
    :class:`~sklearn.ensemble.HistGradientBoostingClassifier` can reduce the bias, at a
    higher cost; with few labeled positives the estimate is noisy either way.

    Parameters
    ----------
    delta : float, default=0.1
        Confidence level :math:`\delta` of the bound, in ``(0, 1)``.
    gamma : float, default=0.01
        Slack :math:`\gamma` of the bound, ``>= 0``.
    estimator : classifier, default=None
        A scikit-learn classifier implementing ``predict_proba``, trained to separate labeled
        from unlabeled samples. It is cloned for each fold. ``None`` uses
        :class:`~sklearn.linear_model.LogisticRegression`; the scores only need to rank the
        samples well.
    cv : int or cross-validation generator, default=5
        Splitting for the out-of-fold scores. An int is the number of shuffled, stratified
        folds.
    random_state : int, RandomState instance or None, default=None
        Seed for shuffling the folds when ``cv`` is an int.

    Attributes
    ----------
    prior_ : float
        Estimated class prior :math:`\pi = P(y = 1)` of the training samples.
    label_frequency_ : float
        Estimated label frequency :math:`c = P(s = 1 \mid y = 1)`.
    pollution_ : float
        Estimated fraction of positives among the unlabeled samples. In the case-control
        scenario it estimates the class prior of the unlabeled population.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``, when ``X`` has string feature names.

    See Also
    --------
    ElkanNotoPrior : The method of Elkan and Noto.

    References
    ----------
    .. [1] S. Garg, Y. Wu, A. J. Smola, S. Balakrishnan and Z. C. Lipton. "Mixture
       proportion estimation and PU learning: a modern approach." NeurIPS 2021.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import label_scar
    >>> from pypu.prior import BBEPrior
    >>> X, y = make_classification(n_samples=2000, weights=[0.7], random_state=0)
    >>> s = label_scar(y, label_frequency=0.4, random_state=0)
    >>> estimator = BBEPrior(random_state=0).fit(X, s)
    >>> abs(estimator.prior_ - 0.3) < 0.03  # the true prior is 0.3
    True
    """

    def __init__(self, delta=0.1, gamma=0.01, estimator=None, cv=5, random_state=None):
        super().__init__(estimator=estimator, cv=cv, random_state=random_state)
        self.delta = delta
        self.gamma = gamma

    def _estimate_pollution(self, scores, s):
        delta, gamma = self.delta, self.gamma
        if isinstance(delta, bool) or not isinstance(delta, numbers.Real) or not 0 < delta < 1:
            raise ValueError(f"delta must be a number in (0, 1), got {delta!r}.")
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, numbers.Real)
            or not 0 <= gamma < math.inf
        ):
            raise ValueError(f"gamma must be a finite number >= 0, got {gamma!r}.")
        labeled = np.sort(scores[s == 1])
        unlabeled = np.sort(scores[s == 0])
        cutoffs = np.unique(labeled)
        # Fractions of each group scoring at least the cut-off; q_p > 0 at every cut-off.
        q_p = 1 - np.searchsorted(labeled, cutoffs, side="left") / labeled.size
        q_u = 1 - np.searchsorted(unlabeled, cutoffs, side="left") / unlabeled.size
        log_term = math.log(4 / delta)
        width = math.sqrt(log_term / (2 * labeled.size)) + math.sqrt(
            log_term / (2 * unlabeled.size)
        )
        best = np.argmin((q_u + (1 + gamma) * width) / q_p)
        return q_u[best] / q_p[best]
