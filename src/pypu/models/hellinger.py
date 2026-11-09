"""PU Hellinger decision trees and forests."""

from __future__ import annotations

import math
import numbers

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

from pypu.base import BasePUClassifier
from pypu.models._hellinger_tree import _RAND_R_MAX, Tree, build_tree
from pypu.utils.validation import check_prior

_MAX_INT = np.iinfo(np.int32).max


class PUHellingerTree(BasePUClassifier):
    r"""Decision tree with a PU Hellinger-distance split criterion.

    The tree splits nodes on the Hellinger distance between the positive and the negative
    class, the criterion of Hellinger distance decision trees, which is insensitive to class
    imbalance. In PU data the classes are not observed, so the criterion estimates them: a
    node with :math:`L` labeled samples among :math:`n` is estimated to hold
    :math:`\min(n, L / c)` positives, with the label frequency
    :math:`c = P(s = 1) / \pi` of the training data. Nodes without labeled samples become
    leaves.

    The trees are built by the algorithm of scikit-learn's ``DecisionTreeClassifier`` (best
    splits, depth first), so they are the trees of the original PU Hellinger tree code for
    the same data and ``random_state``.

    Parameters
    ----------
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the training data, in ``(0, 1)``. Required;
        the default ``None`` raises an error in ``fit``.
    max_depth : int or None, default=None
        Maximum depth of the tree; ``None`` grows it until the leaves cannot be split.
    min_samples_split : int or float, default=2
        Minimum number of samples a node needs to be split: an int is a count, a float in
        ``(0, 1]`` a fraction of the training samples.
    min_samples_leaf : int or float, default=1
        Minimum number of samples in a leaf: an int is a count, a float in ``(0, 0.5]`` a
        fraction of the training samples.
    min_impurity_decrease : float, default=0.0
        A node is split only if the Hellinger distance of its best split is at least this.
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        Number of features drawn at random at each split: an int is a count, a float a
        fraction, ``"sqrt"`` and ``"log2"`` functions of the number of features, ``None``
        all features.
    random_state : int, RandomState instance or None, default=None
        Seed for drawing the features at each split. With ``max_features=None`` it still
        decides between equally good splits.

    Attributes
    ----------
    tree_ : Tree
        The fitted tree: ``children_left``, ``children_right``, ``feature``, ``threshold``
        and ``value`` (the weighted numbers of unlabeled and labeled samples in each node).
    max_features_ : int
        Number of features drawn at each split.
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
    PUHellingerForest : An ensemble of these trees on PU bootstrap samples.

    Notes
    -----
    ``predict_proba`` returns the fraction of labeled samples in each leaf, as the original
    experiments used. It ranks samples but is an estimate of :math:`P(s = 1 \mid x)`, not of
    :math:`P(y = 1 \mid x)`; ``predict_pu_proba`` gives the original code's correction.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import roc_auc_score
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import PUHellingerTree
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> tree = PUHellingerTree(prior=y_train.mean(), max_depth=5, random_state=0)
    >>> tree = tree.fit(X_train, s_train)
    >>> float(roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])) > 0.8
    True
    """

    def __init__(
        self,
        prior=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        min_impurity_decrease=0.0,
        random_state=None,
    ):
        self.prior = prior
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    def _fit(self, X, s):
        prior = _check_prior(self, s)
        params = _tree_parameters(self, X.shape)
        self.max_features_ = params["max_features"]
        seed = check_random_state(self.random_state).randint(0, _RAND_R_MAX)
        self.tree_ = build_tree(X.astype(np.float32), s, None, prior=prior, seed=seed, **params)

    def _predict_proba(self, X):
        return _leaf_fraction(self.tree_, X.astype(np.float32))

    def predict_pu_proba(self, X):
        r"""Estimate class probabilities with the original code's PU correction.

        A leaf whose labeled-to-unlabeled ratio equals that of the training data gets the
        prior; leaves with relatively more labeled samples get more, up to 1. That is, the
        positive-class probability is
        :math:`\min(1, \pi \, \frac{L_\text{leaf} / U_\text{leaf}}{L / U})`, with the
        weighted numbers of labeled (:math:`L`) and unlabeled (:math:`U`) samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probability of the negative class (column 0) and of the positive class (column 1).
        """
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        tree = self.tree_
        root = tree.value[0]
        n_unlabeled = root[0]
        n_labeled = np.sum(root) - n_unlabeled
        if n_labeled == 0:
            n_labeled = 1
        counts = tree.value[tree.apply(X.astype(np.float32))]
        total = np.sum(counts, axis=1)
        prior = check_prior(self.prior)
        with np.errstate(divide="ignore", invalid="ignore"):
            positive = counts[:, 1] * prior * n_unlabeled * total / (n_labeled * counts[:, 0])
        positive = np.where(positive == np.inf, total, positive)
        positive = np.where(positive > total, total, positive)
        proba = np.column_stack([total - positive, positive])
        return proba / np.sum(proba, axis=1)[:, np.newaxis]


class PUHellingerForest(BasePUClassifier):
    r"""Random forest of PU Hellinger trees on PU bootstrap samples.

    Each tree is a :class:`PUHellingerTree` grown on all the labeled positives and a bootstrap
    sample of the unlabeled samples, drawing ``max_features`` features at each split. The
    positive-class probability is the mean over the trees of the fraction of labeled samples
    in the leaf a sample falls in.

    The trees, bootstrap samples included, are those of the original
    ``PURandomForestClassifier`` with ``pu_biased_bootstrap=True`` for the same data,
    parameters and ``random_state``.

    Parameters
    ----------
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the training data, in ``(0, 1)``. Required;
        the default ``None`` raises an error in ``fit``.
    max_depth : int or None, default=None
        Maximum depth of the tree; ``None`` grows it until the leaves cannot be split.
    min_samples_split : int or float, default=2
        Minimum number of samples a node needs to be split: an int is a count, a float in
        ``(0, 1]`` a fraction of the training samples.
    min_samples_leaf : int or float, default=1
        Minimum number of samples in a leaf: an int is a count, a float in ``(0, 0.5]`` a
        fraction of the training samples.
    min_impurity_decrease : float, default=0.0
        A node is split only if the Hellinger distance of its best split is at least this.
    n_estimators : int, default=100
        Number of trees.
    max_features : int, float, {"sqrt", "log2"} or None, default="sqrt"
        Number of features drawn at random at each split: an int is a count, a float a
        fraction, ``"sqrt"`` and ``"log2"`` functions of the number of features, ``None``
        all features.
    max_samples : int, float or None, default=None
        Number of unlabeled samples drawn, with replacement, for each tree: an int is a
        count, a float in ``(0, 1]`` a fraction of the unlabeled samples, and ``None`` all
        of them (the setting of the original experiments).
    n_jobs : int, default=None
        Number of jobs that build the trees in parallel. ``None`` means 1 unless in a
        :func:`joblib.parallel_backend` context, and ``-1`` uses all processors. The result
        does not depend on it.
    random_state : int, RandomState instance or None, default=None
        Seed for the bootstrap samples and the features drawn in each tree.

    Attributes
    ----------
    trees_ : list of Tree
        The fitted trees.
    max_features_ : int
        Number of features drawn at each split.
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
    PUHellingerTree : A single tree.

    Notes
    -----
    As for the trees, ``predict_proba`` ranks samples but is not a calibrated estimate of
    :math:`P(y = 1 \mid x)`.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import roc_auc_score
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import PUHellingerForest
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> forest = PUHellingerForest(prior=y_train.mean(), n_estimators=20, random_state=0)
    >>> forest = forest.fit(X_train, s_train)
    >>> float(roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])) > 0.9
    True
    """

    def __init__(
        self,
        prior=None,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        min_impurity_decrease=0.0,
        max_samples=None,
        n_jobs=None,
        random_state=None,
    ):
        self.prior = prior
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _fit(self, X, s):
        prior = _check_prior(self, s)
        n_estimators = self.n_estimators
        if (
            isinstance(n_estimators, bool)
            or not isinstance(n_estimators, numbers.Integral)
            or n_estimators < 1
        ):
            raise ValueError(f"n_estimators must be a positive int, got {n_estimators!r}.")
        params = _tree_parameters(self, X.shape)
        self.max_features_ = params["max_features"]
        labeled, unlabeled = np.flatnonzero(s == 1), np.flatnonzero(s == 0)
        n_draws = _bootstrap_size(self.max_samples, unlabeled.size)

        # As scikit-learn's forests: one seed per tree, drawn in turn.
        random_state = check_random_state(self.random_state)
        seeds = [random_state.randint(_MAX_INT) for _ in range(n_estimators)]
        X32 = X.astype(np.float32)
        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_build_bootstrap_tree)(
                X32, s, labeled, unlabeled, seed, n_draws=n_draws, prior=prior, params=params
            )
            for seed in seeds
        )

    def _predict_proba(self, X):
        X32 = X.astype(np.float32)
        total = np.zeros(X.shape[0])
        for tree in self.trees_:
            total += _leaf_fraction(tree, X32)
        return total / len(self.trees_)


def _build_bootstrap_tree(X, s, labeled, unlabeled, seed, *, n_draws, prior, params):
    """Build one tree of the forest on a PU bootstrap sample, seeded as scikit-learn does."""
    drawn = np.random.RandomState(seed).randint(0, unlabeled.size, n_draws)
    bag = np.concatenate([unlabeled[drawn], labeled])
    sample_weight = np.bincount(bag, minlength=s.size).astype(np.float64)
    tree_seed = np.random.RandomState(seed).randint(0, _RAND_R_MAX)
    return build_tree(X, s, sample_weight, prior=prior, seed=tree_seed, **params)


def _leaf_fraction(tree: Tree, X):
    """Return the weighted fraction of labeled samples in the leaf of each row of ``X``."""
    counts = tree.value[tree.apply(X)]
    normalizer = np.sum(counts, axis=1)
    normalizer[normalizer == 0.0] = 1.0
    return counts[:, 1] / normalizer


def _check_prior(model, s):
    if model.prior is None:
        raise ValueError(
            f"{type(model).__name__} needs the class prior: pass prior=P(y=1), the fraction "
            "of positives in the training data."
        )
    return check_prior(model.prior, s)


def _tree_parameters(model, shape):
    """Validate the tree parameters, as scikit-learn 1.0 resolves them."""
    n_samples, n_features = shape
    max_depth = model.max_depth
    if max_depth is None:
        max_depth = _MAX_INT
    elif (
        isinstance(max_depth, bool) or not isinstance(max_depth, numbers.Integral) or max_depth < 1
    ):
        raise ValueError(f"max_depth must be None or an int >= 1, got {max_depth!r}.")

    min_samples_leaf = _count_or_fraction(
        model.min_samples_leaf, "min_samples_leaf", n_samples, lowest=1, highest_fraction=0.5
    )
    min_samples_split = _count_or_fraction(
        model.min_samples_split, "min_samples_split", n_samples, lowest=2, highest_fraction=1.0
    )

    max_features = model.max_features
    if max_features is None:
        resolved = n_features
    elif isinstance(max_features, str):
        if max_features == "sqrt":
            resolved = max(1, int(np.sqrt(n_features)))
        elif max_features == "log2":
            resolved = max(1, int(np.log2(n_features)))
        else:
            raise ValueError(
                f"max_features must be 'sqrt', 'log2', an int, a float or None, got "
                f"{max_features!r}."
            )
    elif isinstance(max_features, numbers.Integral) and not isinstance(max_features, bool):
        resolved = int(max_features)
    elif isinstance(max_features, numbers.Real) and not isinstance(max_features, bool):
        resolved = max(1, int(max_features * n_features)) if max_features > 0 else 0
    else:
        raise ValueError(
            f"max_features must be 'sqrt', 'log2', an int, a float or None, got {max_features!r}."
        )
    if not 0 < resolved <= n_features:
        raise ValueError(
            f"max_features={max_features!r} must give between 1 and the number of features "
            f"({n_features})."
        )

    decrease = model.min_impurity_decrease
    if (
        isinstance(decrease, bool)
        or not isinstance(decrease, numbers.Real)
        or not 0 <= decrease < math.inf
    ):
        raise ValueError(f"min_impurity_decrease must be a finite number >= 0, got {decrease!r}.")
    return {
        "max_features": resolved,
        "max_depth": int(max_depth),
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_impurity_decrease": float(decrease),
    }


def _count_or_fraction(value, name, n_samples, *, lowest, highest_fraction):
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        if value < lowest:
            raise ValueError(f"{name} must be an int >= {lowest} or a fraction, got {value}.")
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not 0 < value <= highest_fraction:
            raise ValueError(
                f"{name} must be an int >= {lowest} or a float in (0, {highest_fraction}], "
                f"got {value}."
            )
        return max(lowest, math.ceil(value * n_samples))
    raise ValueError(f"{name} must be an int or a float, got {value!r}.")


def _bootstrap_size(max_samples, n_unlabeled):
    if max_samples is None:
        return n_unlabeled
    if isinstance(max_samples, numbers.Integral) and not isinstance(max_samples, bool):
        if max_samples < 1:
            raise ValueError(f"max_samples must be a positive int, got {max_samples}.")
        return int(max_samples)
    if isinstance(max_samples, numbers.Real) and not isinstance(max_samples, bool):
        if not 0 < max_samples <= 1:
            raise ValueError(
                f"max_samples must be in (0, 1] when it is a float, got {max_samples}."
            )
        return max(1, round(n_unlabeled * max_samples))
    raise ValueError(f"max_samples must be an int, a float or None, got {max_samples!r}.")
