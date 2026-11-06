"""PU bagging: an ensemble trained on the labeled positives and samples of the unlabeled."""

# The bag sampling follows BaggingPuClassifier by Roy Wright, an adaptation of scikit-learn's
# BaggingClassifier (Gilles Louppe, BSD-3-Clause), random draws included, so that both build
# the same bags. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import numbers

import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter

from pypu.base import BasePUClassifier

_MAX_INT = np.iinfo(np.int32).max


class PUBagging(BasePUClassifier):
    r"""PU bagging: classifiers trained on the labeled positives against unlabeled subsets.

    Each of the ``n_estimators`` classifiers is trained on all the labeled positives, as
    positives, and on a bootstrap sample of ``max_samples`` unlabeled samples, as negatives.
    A small sample rarely contains many of the hidden positives, and averaging over the bags
    smooths out the noise of each one [1]_. The positive-class probability is the mean of the
    classifiers' probabilities, or the fraction of them that predict a positive when they do
    not implement ``predict_proba``.

    The probabilities rank samples but are not calibrated estimates of
    :math:`P(y = 1 \mid x)`. With the default ``max_samples``, every bag has as many unlabeled
    samples as labeled positives.

    Parameters
    ----------
    estimator : classifier, default=None
        The scikit-learn classifier trained on each bag. It is cloned for each bag, and its
        ``random_state`` parameters, if any, are set from ``random_state``. ``None`` uses
        :class:`~sklearn.tree.DecisionTreeClassifier`.
    n_estimators : int, default=100
        Number of bags.
    max_samples : int, float or None, default=None
        Number of unlabeled samples drawn for each bag: an int is a count, a float in
        ``(0, 1]`` a fraction of the unlabeled samples, and ``None`` the number of labeled
        positives (or all unlabeled samples, if there are fewer).
    max_features : int or float, default=1.0
        Number of features drawn for each bag: an int is a count, a float in ``(0, 1]`` a
        fraction of the features.
    bootstrap : bool, default=True
        Whether the unlabeled samples are drawn with replacement.
    bootstrap_features : bool, default=False
        Whether the features are drawn with replacement.
    oob_proba : bool, default=False
        Whether to compute :attr:`oob_proba_`, the probabilities of the training samples from
        the classifiers whose bag left them out.
    n_jobs : int, default=None
        Number of jobs that train the classifiers in parallel. ``None`` means 1 unless in a
        :func:`joblib.parallel_backend` context, and ``-1`` uses all processors. The result
        does not depend on it.
    random_state : int, RandomState instance or None, default=None
        Seed for drawing the bags and for the classifiers' own randomness.

    Attributes
    ----------
    estimators_ : list of classifiers
        The fitted classifiers, one per bag.
    estimators_features_ : list of ndarray
        The features drawn for each bag; ``estimators_[i]`` is applied to
        ``X[:, estimators_features_[i]]``.
    oob_proba_ : ndarray of shape (n_samples, 2)
        Out-of-bag class probabilities of the training samples, averaged over the classifiers
        whose bag did not contain the sample, the transductive scores of [1]_. Rows are NaN
        for samples that were in every bag, such as all labeled positives. Only set when
        ``oob_proba=True``.
    classes_ : ndarray of shape (2,)
        The class labels, always ``[0, 1]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``, when ``X`` has string feature names.
    threshold_ : float
        Positive-class probability threshold used by ``predict``; ``0.5``.

    Notes
    -----
    The bags, and so the predictions, are those of the ``BaggingPuClassifier`` used in
    earlier experiments (adapted from scikit-learn's ``BaggingClassifier`` by Roy Wright)
    with the same parameters and seed. Its out-of-bag estimates differed: when the
    classifier has a ``random_state`` parameter, they were computed from other bags than
    the ones the classifiers were trained on.

    References
    ----------
    .. [1] F. Mordelet and J.-P. Vert. "A bagging SVM to learn from positive and unlabeled
       examples." Pattern Recognition Letters 37, 2014.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import roc_auc_score
    >>> from pypu.datasets import label_scar
    >>> from pypu.models import PUBagging
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> s = label_scar(y, label_frequency=0.3, random_state=0)
    >>> clf = PUBagging(n_estimators=50, oob_proba=True, random_state=0).fit(X, s)
    >>> clf.predict_proba(X).shape
    (1000, 2)
    >>> unlabeled = s == 0  # every unlabeled sample is scored by the bags that left it out
    >>> float(roc_auc_score(y[unlabeled], clf.oob_proba_[unlabeled, 1])) > 0.9
    True
    """

    def __init__(
        self,
        estimator=None,
        n_estimators=100,
        max_samples=None,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_proba=False,
        n_jobs=None,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_proba = oob_proba
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _fit(self, X, s):
        n_estimators = self.n_estimators
        if (
            isinstance(n_estimators, bool)
            or not isinstance(n_estimators, numbers.Integral)
            or n_estimators < 1
        ):
            raise ValueError(f"n_estimators must be a positive int, got {n_estimators!r}.")
        estimator = DecisionTreeClassifier() if self.estimator is None else clone(self.estimator)
        labeled = np.flatnonzero(s == 1)
        unlabeled = np.flatnonzero(s == 0)
        max_samples = _resolve_count(
            self.max_samples,
            unlabeled.size,
            name="max_samples",
            what="unlabeled samples",
            default=min(labeled.size, unlabeled.size),
        )
        max_features = _resolve_count(
            self.max_features, X.shape[1], name="max_features", what="features"
        )

        seeds = check_random_state(self.random_state).randint(_MAX_INT, size=n_estimators)
        use_sample_weight = has_fit_parameter(estimator, "sample_weight")
        members = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_member)(
                estimator,
                X,
                s,
                labeled,
                unlabeled,
                seed,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bool(self.bootstrap),
                bootstrap_features=bool(self.bootstrap_features),
                use_sample_weight=use_sample_weight,
            )
            for seed in seeds
        )
        self.estimators_ = [member[0] for member in members]
        self.estimators_features_ = [member[1] for member in members]
        if self.oob_proba:
            self.oob_proba_ = self._out_of_bag_proba(X, [member[2] for member in members])

    def _member_proba(self, estimator, X):
        """Return the class probabilities, or the class votes, of one classifier."""
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            if proba.shape[1] == 2:
                return proba
            # A classifier that only saw one class (possible with a custom estimator).
            full = np.zeros((X.shape[0], 2))
            full[:, estimator.classes_.astype(np.int64)] = proba
            return full
        votes = np.zeros((X.shape[0], 2))
        votes[np.arange(X.shape[0]), estimator.predict(X).astype(np.int64)] = 1.0
        return votes

    def _predict_proba(self, X):
        total = np.zeros((X.shape[0], 2))
        for estimator, features in zip(self.estimators_, self.estimators_features_, strict=True):
            total += self._member_proba(estimator, X[:, features])
        return total[:, 1] / len(self.estimators_)

    def _out_of_bag_proba(self, X, bags):
        total = np.zeros((X.shape[0], 2))
        for estimator, features, bag in zip(
            self.estimators_, self.estimators_features_, bags, strict=True
        ):
            out_of_bag = np.ones(X.shape[0], dtype=bool)
            out_of_bag[bag] = False
            if out_of_bag.any():
                total[out_of_bag] += self._member_proba(estimator, X[out_of_bag][:, features])
        with np.errstate(invalid="ignore", divide="ignore"):
            return total / total.sum(axis=1, keepdims=True)


def _resolve_count(value, total, *, name, what, default=None):
    """Turn an int count, a float fraction or ``None`` into a count in ``[1, total]``."""
    if value is None and default is not None:
        return default
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        count = int(value)
        if not 0 < count <= total:
            raise ValueError(
                f"{name}={count} must be between 1 and the number of {what} ({total})."
            )
        return count
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1] when it is a float, got {value}.")
        count = int(value * total)
        if count < 1:
            raise ValueError(f"{name}={value} selects no {what}: there are only {total} of them.")
        return count
    allowed = "an int, a float or None" if default is not None else "an int or a float"
    raise ValueError(f"{name} must be {allowed}, got {value!r}.")


def _set_random_states(estimator, random_state):
    """Set every ``random_state`` parameter of ``estimator`` from ``random_state``."""
    to_set = {
        key: random_state.randint(_MAX_INT)
        for key in sorted(estimator.get_params(deep=True))
        if key == "random_state" or key.endswith("__random_state")
    }
    if to_set:
        estimator.set_params(**to_set)


def _draw(random_state, bootstrap, n_population, n_draws):
    if bootstrap:
        return random_state.randint(0, n_population, n_draws)
    return sample_without_replacement(n_population, n_draws, random_state=random_state)


def _fit_member(
    estimator,
    X,
    s,
    labeled,
    unlabeled,
    seed,
    *,
    max_samples,
    max_features,
    bootstrap,
    bootstrap_features,
    use_sample_weight,
):
    """Fit one classifier on its bag; return it, its features and its sample indices."""
    # The order of the random draws matches BaggingPuClassifier.
    random_state = np.random.RandomState(seed)
    estimator = clone(estimator)
    _set_random_states(estimator, random_state)
    features = _draw(random_state, bootstrap_features, X.shape[1], max_features)
    drawn = _draw(random_state, bootstrap, unlabeled.size, max_samples)
    bag = np.concatenate([unlabeled[drawn], labeled])
    if use_sample_weight:
        if bootstrap:
            sample_weight = np.bincount(bag, minlength=X.shape[0]).astype(np.float64)
        else:
            sample_weight = np.zeros(X.shape[0])
            sample_weight[bag] = 1.0
        estimator.fit(X[:, features], s, sample_weight=sample_weight)
    else:
        estimator.fit(X[bag][:, features], s[bag])
    return estimator, features, bag
