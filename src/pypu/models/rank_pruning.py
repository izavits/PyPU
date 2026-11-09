"""Rank Pruning: learning from the confident examples of PU data."""

# Ported from the rankpruning package of Curtis G. Northcutt (MIT License), as copied into the
# PU Hellinger trees code, in its PU learning mode. It keeps the original's estimates, rounding
# and pruning rule, so that the results match it. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import numbers
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.utils.validation import has_fit_parameter

from pypu.base import BasePUClassifier
from pypu.utils.validation import check_prior

# The original never prunes a class below this many samples.
_MIN_NUM_PER_CLASS = 10
# Noise rates are clipped to [0, _MAX_RATE], as in the original.
_MAX_RATE = 0.9999


class RankPruning(BasePUClassifier):
    r"""Rank Pruning: prunes the unlabeled samples that look positive, then refits.

    Rank Pruning [1]_ learns from the confident examples. On PU data, where the labeled
    samples are positive and the unlabeled ones are treated as negatives with label noise,
    it works in three steps:

    1. **Noise rate:** out-of-fold probabilities :math:`g(x) = P(s = 1 \mid x)` of a
       classifier separating labeled from unlabeled samples give, within each fold, a
       confident-counts estimate of :math:`\rho_1 = P(s = 0 \mid y = 1)`: among the samples
       whose :math:`g(x)` is at least the mean :math:`g` of the fold's labeled samples, the
       fraction that is unlabeled. The estimate is the mean over the folds. With ``prior``,
       :math:`\rho_1 = 1 - P(s = 1) / \pi` instead.
    2. **Pruning:** the fraction of positives among the unlabeled samples is
       :math:`\pi_0 = \rho_1 P(y = 1) / P(s = 0)`, with :math:`P(y = 1) = P(s = 1) / (1 -
       \rho_1)`. The :math:`\lfloor \pi_0 n_u \rfloor` unlabeled samples with the highest
       :math:`g(x)` are pruned (keeping at least 10 unlabeled samples).
    3. **Refit:** the classifier is trained on the remaining samples, the labeled ones
       weighted by :math:`1 / (1 - \rho_1)` so that they count for all the positives.

    ``predict_proba`` is the refitted classifier's, an estimate of :math:`P(y = 1 \mid x)`.

    Parameters
    ----------
    estimator : classifier, default=None
        A scikit-learn classifier implementing ``predict_proba`` and accepting
        ``sample_weight`` in ``fit``. It is cloned for each fit. ``None`` uses
        :class:`~sklearn.linear_model.LogisticRegression`, as the original does.
    prior : float, default=None
        Class prior :math:`\pi = P(y = 1)` of the training data, which gives
        :math:`\rho_1`. ``None`` estimates :math:`\rho_1` by confident counts.
    cv : int or cross-validation generator, default=3
        Splitting for the out-of-fold probabilities. An int is the number of shuffled,
        stratified folds, as in the original.
    random_state : int, RandomState instance or None, default=None
        Seed for shuffling the folds when ``cv`` is an int. The original shuffled with
        NumPy's global random state: ``random_state=i`` gives the folds it drew after
        ``np.random.seed(i)``.

    Attributes
    ----------
    estimator_ : classifier
        The classifier refitted on the pruned data.
    label_frequency_ : float
        The label frequency :math:`c = 1 - \rho_1` (estimated, or from ``prior``).
    prior_ : float
        The estimated class prior :math:`P(y = 1) = P(s = 1) / c`.
    pollution_ : float
        The estimated fraction of positives among the unlabeled samples, :math:`\pi_0`.
    pruned_ : ndarray of bool of shape (n_samples,)
        Which training samples were pruned (all of them unlabeled).
    oof_proba_ : ndarray of shape (n_samples,)
        The out-of-fold probabilities :math:`P(s = 1 \mid x)` of the training samples.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``, ``0.5``.

    See Also
    --------
    ElkanNoto : Corrects the labeled-vs-unlabeled classifier by the label frequency instead.
    SpyPU : Prunes the other way: keeps only the unlabeled samples that look negative.

    References
    ----------
    .. [1] C. G. Northcutt, T. Wu and I. L. Chuang. "Learning with confident examples: rank
       pruning for robust classification with noisy labels." UAI 2017.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import RankPruning
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> clf = RankPruning(random_state=0).fit(X_train, s_train)
    >>> int(clf.pruned_.sum()) > 0
    True
    """

    def __init__(self, estimator=None, prior=None, cv=3, random_state=None):
        self.estimator = estimator
        self.prior = prior
        self.cv = cv
        self.random_state = random_state

    def _fit(self, X, s):
        base = LogisticRegression() if self.estimator is None else self.estimator
        if not hasattr(base, "predict_proba"):
            raise TypeError(
                f"estimator must implement predict_proba; {type(base).__name__} does not."
            )
        if not has_fit_parameter(base, "sample_weight"):
            raise TypeError(
                f"estimator must accept sample_weight in fit; {type(base).__name__} does not."
            )
        prior = None if self.prior is None else check_prior(self.prior, s)
        cv = self.cv
        if isinstance(cv, numbers.Integral) and not isinstance(cv, bool):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        cv = check_cv(cv, s, classifier=True)

        # Step 1: out-of-fold probabilities, and the confident-counts noise rate of each fold.
        proba = np.zeros(len(s))
        fold_rates = []
        for train, test in cv.split(X, s):
            model = clone(base).fit(X[train], s[train])
            proba[test] = _positive_proba(model, X[test])
            fold_rates.append(_confident_counts_rho1(s[test], proba[test]))
        if prior is None:
            finite = [rate for rate in fold_rates if np.isfinite(rate)]
            if not finite:
                raise ValueError(
                    "RankPruning could not estimate the noise rate: no fold had a confident "
                    "positive. Pass the class prior instead."
                )
            rho1 = float(np.mean(finite))
        else:
            rho1 = min(max(1.0 - float(np.mean(s)) / prior, 0.0), _MAX_RATE)

        # Step 2: prune the unlabeled samples most likely to be positive.
        labeled_fraction = np.sum(s) / float(len(s))
        prior_estimate = labeled_fraction / float(1 - rho1)
        pollution = min(max(rho1 * prior_estimate / float(1 - labeled_fraction), 0.0), _MAX_RATE)
        pruned = _prune(s, proba, pollution)

        # Step 3: refit, with the labeled samples counting for all the positives.
        keep = ~pruned
        sample_weight = np.where(s[keep] == 1, 1.0 / float(1 - rho1), 1.0)
        self.estimator_ = clone(base).fit(X[keep], s[keep], sample_weight=sample_weight)
        self.label_frequency_ = 1.0 - rho1
        self.prior_ = float(prior_estimate)
        self.pollution_ = float(pollution)
        self.pruned_ = pruned
        self.oof_proba_ = proba

    def _predict_proba(self, X):
        return _positive_proba(self.estimator_, X)


def _positive_proba(estimator, X):
    return estimator.predict_proba(X)[:, np.flatnonzero(estimator.classes_ == 1)[0]]


def _confident_counts_rho1(s, proba):
    """Return the original's confident-counts estimate of P(s = 0 | y = 1), or NaN.

    Samples with a probability at least the labeled samples' mean are confidently positive;
    the estimate is the fraction of them that is unlabeled.
    """
    threshold = np.mean(proba[s == 1])
    confident = proba >= threshold
    n_unlabeled = int(np.sum(confident & (s == 0)))
    n_labeled = int(np.sum(confident & (s == 1)))
    if n_unlabeled + n_labeled == 0:
        return np.nan
    return min(max(n_unlabeled / float(n_unlabeled + n_labeled), 0.0), _MAX_RATE)


def _prune(s, proba, pollution):
    """Mask of the unlabeled samples to prune, by the original's rule (``get_noise_indices``)."""
    unlabeled = s == 0
    n_unlabeled = int(unlabeled.sum())
    k = max(min(int(n_unlabeled * pollution), n_unlabeled - _MIN_NUM_PER_CLASS), 0)
    if not (pollution > 0 and n_unlabeled > _MIN_NUM_PER_CLASS):
        if pollution > 0:
            warnings.warn(
                f"RankPruning prunes nothing: it keeps at least {_MIN_NUM_PER_CLASS} unlabeled "
                "samples.",
                UserWarning,
                stacklevel=4,
            )
        return np.zeros_like(unlabeled)
    # The k-th largest unlabeled probability; samples strictly above it are pruned.
    kth_largest = -np.partition(-proba[unlabeled], k)[k]
    return unlabeled & (proba > kth_largest)
