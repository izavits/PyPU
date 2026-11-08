"""Two-step PU methods: find reliable negatives, then train a classifier."""

from __future__ import annotations

import numbers
import warnings

import numpy as np
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.utils.validation import has_fit_parameter

from pypu.base import BasePUClassifier
from pypu.metrics import lee_liu_score


def _positive_proba(estimator, X):
    """Probability of class 1 from a fitted classifier with ``predict_proba``."""
    return estimator.predict_proba(X)[:, np.flatnonzero(estimator.classes_ == 1)[0]]


def _require(estimator, method, name):
    if not hasattr(estimator, method):
        raise TypeError(f"{name} must implement {method}; {type(estimator).__name__} does not.")


class SpyPU(BasePUClassifier):
    r"""The spy technique: reliable negatives below the spies, then a classifier.

    A two-step method [1]_. **Step 1:** a fraction ``spy_ratio`` of the labeled positives,
    the spies, is hidden among the unlabeled samples, and a classifier (``estimator``) is
    trained to separate the remaining labeled samples from the unlabeled ones and the spies.
    The spies behave like the unlabeled positives, so unlabeled samples scored below the
    spies (below the ``spy_quantile`` quantile of their scores; their minimum by default)
    are taken as reliable negatives. **Step 2:** a classifier (``final_estimator``) is
    trained on all the labeled positives against the reliable negatives. With ``n_em_iter >
    0`` it is then refined by expectation-maximization over all the unlabeled samples, whose
    labels become probabilities while the labeled samples stay positive, as in S-EM.

    Parameters
    ----------
    estimator : classifier, default=None
        The classifier of step 1, implementing ``predict_proba``; cloned before fitting.
        ``None`` uses :class:`~sklearn.linear_model.LogisticRegression`.
    final_estimator : classifier, default=None
        The classifier of step 2, implementing ``predict_proba`` (and accepting
        ``sample_weight`` when ``n_em_iter > 0``); cloned before fitting. ``None`` uses
        :class:`~sklearn.naive_bayes.GaussianNB`.
    spy_ratio : float, default=0.1
        Fraction of the labeled positives used as spies, in ``(0, 1)``; at least one spy.
    spy_quantile : float, default=0.0
        Quantile of the spies' scores below which unlabeled samples are reliable negatives,
        in ``[0, 1)``. 0 takes the lowest-scored spy; a small positive value (S-EM's "noise
        level", such as 0.15) tolerates a few outlying spies.
    n_em_iter : int, default=0
        Number of EM iterations refining the final classifier; 0 keeps the classifier of step
        2.
    random_state : int, RandomState instance or None, default=None
        Seed for choosing the spies.

    Attributes
    ----------
    estimator_ : classifier
        The classifier of step 1.
    final_estimator_ : classifier
        The final classifier.
    spies_ : ndarray of int
        Indices of the spies among the training samples.
    spy_threshold_ : float
        The score below which unlabeled samples are reliable negatives.
    reliable_negatives_ : ndarray of bool of shape (n_samples,)
        Which training samples are reliable negatives.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``, ``0.5``.

    See Also
    --------
    PEBL : Spy (or 1-DNF) reliable negatives, then an iterative SVM.
    IFPNU : Reliable negatives from an anomaly detector, then LightGBM.

    Notes
    -----
    The spies must be numerous enough to represent the unlabeled positives: with few labeled
    samples, the threshold is unreliable. When no unlabeled sample scores below the
    threshold, the lowest-scored one is taken as the reliable negative, with a warning.

    References
    ----------
    .. [1] B. Liu, W. S. Lee, P. S. Yu and X. Li. "Partially supervised classification of
       text documents." ICML 2002.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import SpyPU
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = SpyPU(random_state=0).fit(X_train, s_train)
    >>> int(clf.reliable_negatives_.sum()) > 0
    True
    """

    def __init__(
        self,
        estimator=None,
        final_estimator=None,
        spy_ratio=0.1,
        spy_quantile=0.0,
        n_em_iter=0,
        random_state=None,
    ):
        self.estimator = estimator
        self.final_estimator = final_estimator
        self.spy_ratio = spy_ratio
        self.spy_quantile = spy_quantile
        self.n_em_iter = n_em_iter
        self.random_state = random_state

    def _fit(self, X, s):
        _check_spy_settings(self.spy_ratio, self.spy_quantile)
        n_em_iter = self.n_em_iter
        if isinstance(n_em_iter, bool) or not isinstance(n_em_iter, numbers.Integral):
            raise ValueError(f"n_em_iter must be an int >= 0, got {n_em_iter!r}.")
        if n_em_iter < 0:
            raise ValueError(f"n_em_iter must be an int >= 0, got {n_em_iter!r}.")
        estimator = LogisticRegression() if self.estimator is None else clone(self.estimator)
        final = GaussianNB() if self.final_estimator is None else clone(self.final_estimator)
        _require(estimator, "predict_proba", "estimator")
        _require(final, "predict_proba", "final_estimator")
        if n_em_iter and not has_fit_parameter(final, "sample_weight"):
            raise TypeError("n_em_iter > 0 needs a final_estimator accepting sample_weight.")

        # Step 1: spies among the unlabeled samples, and the reliable negatives below them.
        reliable, spies, threshold = _spy_negatives(
            estimator, X, s, self.spy_ratio, self.spy_quantile, self.random_state
        )
        labeled, unlabeled = np.flatnonzero(s == 1), s == 0

        # Step 2: the labeled positives against the reliable negatives, then EM.
        train = (s == 1) | reliable
        final.fit(X[train], s[train])
        X_unlabeled = X[unlabeled]
        X_em = np.concatenate([X[s == 1], X_unlabeled, X_unlabeled])
        y_em = np.concatenate(
            [np.ones(len(labeled)), np.ones(len(X_unlabeled)), np.zeros(len(X_unlabeled))]
        )
        for _ in range(n_em_iter):
            proba = _positive_proba(final, X_unlabeled)
            weight = np.concatenate([np.ones(len(labeled)), proba, 1 - proba])
            final = clone(final).fit(X_em, y_em, sample_weight=weight)

        self.estimator_ = estimator
        self.final_estimator_ = final
        self.spies_ = spies
        self.spy_threshold_ = threshold
        self.reliable_negatives_ = reliable

    def _predict_proba(self, X):
        return _positive_proba(self.final_estimator_, X)


def _check_spy_settings(spy_ratio, spy_quantile):
    if (
        isinstance(spy_ratio, bool)
        or not isinstance(spy_ratio, numbers.Real)
        or not 0 < spy_ratio < 1
    ):
        raise ValueError(f"spy_ratio must be a number in (0, 1), got {spy_ratio!r}.")
    if (
        isinstance(spy_quantile, bool)
        or not isinstance(spy_quantile, numbers.Real)
        or not 0 <= spy_quantile < 1
    ):
        raise ValueError(f"spy_quantile must be a number in [0, 1), got {spy_quantile!r}.")


def _spy_negatives(estimator, X, s, spy_ratio, spy_quantile, random_state):
    """Return the reliable negatives (a mask), the spies (sorted indices) and the threshold.

    ``estimator`` is fitted in place, on the labels with the spies unlabeled.
    """
    labeled = np.flatnonzero(s == 1)
    n_spies = max(1, round(len(labeled) * spy_ratio))
    spies = check_random_state(random_state).choice(labeled, n_spies, replace=False)
    s_spied = s.copy()
    s_spied[spies] = 0
    if not s_spied.any():
        raise ValueError("The spy technique needs labeled samples left after choosing the spies.")
    estimator.fit(X, s_spied)
    scores = _positive_proba(estimator, X)
    threshold = float(np.quantile(scores[spies], spy_quantile))
    unlabeled = s == 0
    reliable = unlabeled & (scores < threshold)
    if not reliable.any():
        warnings.warn(
            "No unlabeled sample scores below the spies; the lowest-scored one is taken as the "
            "reliable negative.",
            UserWarning,
            stacklevel=4,
        )
        reliable[np.flatnonzero(unlabeled)[np.argmin(scores[unlabeled])]] = True
    return reliable, np.sort(spies), threshold


class PEBL(BasePUClassifier):
    r"""Reliable negatives, then an iterative SVM (PEBL, Spy-SVM).

    A two-step method. **Step 1:** reliable negatives, from one of two techniques:

    - ``negatives="spy"`` (the default): the spy technique of :class:`SpyPU`, which works on
      any features; with the iterative SVM this is Spy-SVM [2]_.
    - ``negatives="1dnf"``: 1-DNF, the first step of PEBL [1]_, for binary or count features
      (such as words in documents). A feature is present in a sample when it is not zero; the
      positive features are those present in a larger fraction of the labeled samples than of
      the unlabeled ones, and the reliable negatives are the unlabeled samples with none of
      them. On continuous features every nonzero value counts as present, so 1-DNF finds no
      positive feature and takes every unlabeled sample as a reliable negative (with a
      warning): use spies there.

    **Step 2:** an SVM (``estimator``) is trained on the labeled positives against the reliable
    negatives, the unlabeled samples it classifies as negative join the reliable negatives,
    and so on until no sample joins (or ``max_iter``). **Step 3:** with
    ``selection="lee_liu"`` the SVM of the iteration with the highest Lee and Liu criterion
    on the training data (:func:`~pypu.metrics.lee_liu_score`) is kept, a criterion used by
    Liu et al. [2]_ and by the notebooks of github.com/hkiyomaru/pu-learning; PEBL kept the last
    one (``selection="last"``).

    Parameters
    ----------
    estimator : classifier, default=None
        The classifier of each iteration, implementing ``decision_function`` or
        ``predict_proba``; cloned for each fit. ``None`` uses ``SVC(kernel="linear")``.
    negatives : {"spy", "1dnf"}, default="spy"
        How to find the first reliable negatives.
    spy_estimator : classifier, default=None
        With ``negatives="spy"``, the classifier separating labeled samples from unlabeled
        samples and spies, implementing ``predict_proba``. ``None`` uses
        :class:`~sklearn.linear_model.LogisticRegression`.
    spy_ratio : float, default=0.1
        With ``negatives="spy"``, the fraction of labeled samples used as spies.
    spy_quantile : float, default=0.0
        With ``negatives="spy"``, the quantile of the spies' scores below which unlabeled
        samples are reliable negatives (see :class:`SpyPU`).
    selection : {"lee_liu", "last"}, default="lee_liu"
        Which iteration's classifier to keep.
    max_iter : int, default=100
        Maximum number of SVM iterations.
    random_state : int, RandomState instance or None, default=None
        Seed for choosing the spies.

    Attributes
    ----------
    estimator_ : classifier
        The selected classifier.
    estimators_ : list of classifiers
        The classifier of every iteration.
    selected_iteration_ : int
        Index of the selected classifier in ``estimators_``.
    n_iter_ : int
        Number of iterations run.
    reliable_negatives_ : ndarray of bool of shape (n_samples,)
        Which training samples are reliable negatives after step 1.
    spies_ : ndarray of int
        With ``negatives="spy"``, the indices of the spies.
    positive_features_ : ndarray of bool of shape (n_features_in_,)
        With ``negatives="1dnf"``, which features are positive features.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``, ``0.5``.

    See Also
    --------
    SpyPU : Spy reliable negatives, then a single classifier or EM.

    Notes
    -----
    ``decision_function`` returns the selected SVM's decision value and ``predict_proba`` its
    sigmoid, which is not calibrated; ``predict`` gives the SVM's classification. When step 1
    finds no reliable negative, the most negative-looking unlabeled sample is taken, with a
    warning.

    References
    ----------
    .. [1] H. Yu, J. Han and K. C.-C. Chang. "PEBL: positive example based learning for web
       page classification using SVM." KDD 2002.
    .. [2] B. Liu, Y. Dai, X. Li, W. S. Lee and P. S. Yu. "Building text classifiers using
       positive and unlabeled examples." ICDM 2003.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import PEBL
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = PEBL(random_state=0).fit(X_train, s_train)
    >>> clf.n_iter_ >= 1
    True
    """

    def __init__(
        self,
        estimator=None,
        negatives="spy",
        spy_estimator=None,
        spy_ratio=0.1,
        spy_quantile=0.0,
        selection="lee_liu",
        max_iter=100,
        random_state=None,
    ):
        self.estimator = estimator
        self.negatives = negatives
        self.spy_estimator = spy_estimator
        self.spy_ratio = spy_ratio
        self.spy_quantile = spy_quantile
        self.selection = selection
        self.max_iter = max_iter
        self.random_state = random_state

    def _fit(self, X, s):
        if self.negatives not in ("spy", "1dnf"):
            raise ValueError(f"negatives must be 'spy' or '1dnf', got {self.negatives!r}.")
        if self.selection not in ("lee_liu", "last"):
            raise ValueError(f"selection must be 'lee_liu' or 'last', got {self.selection!r}.")
        max_iter = self.max_iter
        if isinstance(max_iter, bool) or not isinstance(max_iter, numbers.Integral):
            raise ValueError(f"max_iter must be a positive int, got {max_iter!r}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be a positive int, got {max_iter!r}.")
        base = SVC(kernel="linear") if self.estimator is None else self.estimator

        # Step 1: the first reliable negatives.
        if self.negatives == "spy":
            _check_spy_settings(self.spy_ratio, self.spy_quantile)
            spy_estimator = (
                LogisticRegression() if self.spy_estimator is None else clone(self.spy_estimator)
            )
            _require(spy_estimator, "predict_proba", "spy_estimator")
            reliable, self.spies_, _ = _spy_negatives(
                spy_estimator, X, s, self.spy_ratio, self.spy_quantile, self.random_state
            )
        else:
            reliable, self.positive_features_ = _one_dnf_negatives(X, s)

        # Step 2: iterative SVM. Labels: 1 positive, 0 negative, -1 not yet labeled.
        labels = np.where(s == 1, 1, np.where(reliable, 0, -1))
        estimators = []
        for _ in range(max_iter):
            known = labels != -1
            model = clone(base).fit(X[known], labels[known])
            estimators.append(model)
            joining = (labels == -1) & (model.predict(X) == 0)
            if not joining.any():
                break
            labels[joining] = 0

        # Step 3: selection.
        if self.selection == "last":
            selected = len(estimators) - 1
        else:
            criteria = [lee_liu_score(s, model.predict(X)) for model in estimators]
            selected = int(np.argmax(criteria))

        self.estimators_ = estimators
        self.selected_iteration_ = selected
        self.estimator_ = estimators[selected]
        self.n_iter_ = len(estimators)
        self.reliable_negatives_ = reliable

    def _decision_function(self, X):
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X)
        proba = np.clip(_positive_proba(self.estimator_, X), 1e-12, 1 - 1e-12)
        return np.log(proba / (1 - proba))

    def _predict_proba(self, X):
        return expit(self._decision_function(X))


def _one_dnf_negatives(X, s):
    """Return the 1-DNF reliable negatives (a mask) and the positive features (a mask)."""
    present = X != 0
    positive_features = present[s == 1].mean(axis=0) > present[s == 0].mean(axis=0)
    unlabeled = s == 0
    if not positive_features.any():
        warnings.warn(
            "1-DNF found no positive feature, so every unlabeled sample is a reliable negative. "
            "It needs binary or count features; use negatives='spy' otherwise.",
            UserWarning,
            stacklevel=4,
        )
    reliable = unlabeled & ~present[:, positive_features].any(axis=1)
    if not reliable.any():
        warnings.warn(
            "Every unlabeled sample has a positive feature; the one with the fewest is taken "
            "as the reliable negative.",
            UserWarning,
            stacklevel=4,
        )
        counts = present[:, positive_features].sum(axis=1)
        candidates = np.flatnonzero(unlabeled)
        reliable[candidates[np.argmin(counts[candidates])]] = True
    return reliable, positive_features
