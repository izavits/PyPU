"""PU Extra Trees: extremely randomized trees grown on a PU risk."""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed

from pypu.base import BasePUClassifier
from pypu.utils.validation import check_prior

_MAX_INT = np.iinfo(np.int32).max
_LEAF = -1
_RISK_ESTIMATORS = ("uPU", "nnPU")
_LOSSES = ("quadratic", "logistic")
_SCENARIOS = ("single", "case_control")


@dataclass
class Tree:
    """A fitted PU extra tree, stored as arrays indexed by node; the root is node 0.

    Nodes are numbered in the order the original code creates them: when a node is split,
    its two children get the next two numbers, and the left subtree is grown first. A
    sample goes to the left child when its value of ``feature`` is at most ``threshold``.

    Attributes
    ----------
    children_left : ndarray of shape (n_nodes,)
        Left child of each node, ``-1`` for leaves.
    children_right : ndarray of shape (n_nodes,)
        Right child of each node, ``-1`` for leaves.
    feature : ndarray of shape (n_nodes,)
        Feature each node splits on, ``-1`` for leaves.
    threshold : ndarray of shape (n_nodes,)
        Split point of each node, NaN for leaves.
    vstar : ndarray of shape (n_nodes,)
        The risk's estimate of the share of positives in each leaf (``v*`` in the original
        code), which can exceed 1 and is infinite in leaves without unlabeled samples; NaN
        for internal nodes.
    value : ndarray of shape (n_nodes,)
        Positive-class probability of each leaf, ``vstar`` clipped to ``[0, 1]`` (1 where it
        is infinite); NaN for internal nodes.
    risk_reduction : ndarray of shape (n_nodes,)
        Decrease of the risk achieved by each node's split, 0 for leaves.
    """

    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    vstar: np.ndarray
    value: np.ndarray
    risk_reduction: np.ndarray

    @property
    def node_count(self) -> int:
        """Number of nodes in the tree."""
        return int(self.feature.size)

    def apply(self, X):
        """Return the index of the leaf each row of ``X`` falls in.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        leaves : ndarray of shape (n_samples,)
            Leaf indices.
        """
        node = np.zeros(X.shape[0], dtype=np.intp)
        active = np.flatnonzero(self.feature[node] != _LEAF)
        while active.size:
            current = node[active]
            goes_left = X[active, self.feature[current]] <= self.threshold[current]
            node[active] = np.where(
                goes_left, self.children_left[current], self.children_right[current]
            )
            active = active[self.feature[node[active]] != _LEAF]
        return node


class PUExtraTrees(BasePUClassifier):
    r"""Extra trees grown by greedily minimizing the uPU or nnPU risk.

    PU Extra Trees [1]_ is a forest of extremely randomized trees whose splits minimize a PU
    estimate of the risk instead of an impurity computed from class labels. At each node,
    ``max_features`` of the features that vary in the node are drawn at random, and
    ``max_candidates`` split points are drawn uniformly for each of them; the split that
    minimizes the risk of the two children is kept. With the quadratic loss the risk plays
    the role of the Gini impurity, with the logistic loss that of the entropy. Trees are
    grown on all the training samples (no bootstrap).

    In a node holding :math:`n_\text{L}` of the :math:`N_\text{L}` labeled samples and
    :math:`n_\text{U}` of the :math:`N_\text{U}` unlabeled ones, the positive and negative
    weights are :math:`W_+ = \pi_\text{U} \, n_\text{L} / N_\text{L}` and
    :math:`W_- = n_\text{U} / N_\text{U} - W_+`, where :math:`\pi_\text{U}` is the share of
    positives among the unlabeled samples. The node's estimate of the share of positives is
    :math:`v^* = W_+ / (W_+ + W_-)` and its risk, with the quadratic loss,
    :math:`4 (W_+ + W_-) \, v^* (1 - v^*)`. The unbiased risk (uPU) can go below zero, where
    :math:`v^* > 1`; the non-negative risk (nnPU) counts such nodes as pure, with risk 0.

    The trees are those of the authors' code [2]_ for the same data and seed, with the labeled
    samples as the positive set and the unlabeled samples as the unlabeled set, as PU-Bench
    [3]_ runs it; the probabilities follow PU-Bench's addition to that code (see Notes).

    Parameters
    ----------
    prior : float
        Class prior :math:`\pi = P(y = 1)` of the population, in ``(0, 1)``: of all the
        training samples in the single-training-set scenario, where it must be above the
        fraction of labeled samples, and of the unlabeled samples in the case-control
        scenario. Required; the default ``None`` raises an error in ``fit``.
    scenario : {"single", "case_control"}, default="single"
        How the PU data were sampled. ``"single"``: one sample of the population, with some
        of its positives labeled; the risk then uses the share of positives among the
        unlabeled samples, :math:`\pi_\text{U} = (\pi n - N_\text{L}) / (n - N_\text{L})`
        for :math:`n` training samples. ``"case_control"``: the labeled positives and the
        unlabeled samples drawn separately, the unlabeled ones from the population, as made
        by :func:`~pypu.datasets.make_case_control`; then :math:`\pi_\text{U} = \pi`.
    n_estimators : int, default=100
        Number of trees.
    risk_estimator : {"nnPU", "uPU"}, default="nnPU"
        The PU risk minimized: non-negative (nnPU) or unbiased (uPU).
    loss : {"quadratic", "logistic"}, default="quadratic"
        Loss of the risk: :math:`(1 - v y)^2` or :math:`\ln(1 + e^{-v y})`.
    max_depth : int or None, default=None
        Maximum depth of the trees; ``None`` grows them until the leaves cannot be split.
    min_samples_leaf : int, default=1
        A node with at most this many samples becomes a leaf. As in the original code, a
        split can still leave fewer samples in a child (unlike scikit-learn's parameter of
        the same name).
    max_features : int, "sqrt" or None, default="sqrt"
        Number of features drawn at each split: an int is a count, ``"sqrt"`` is
        :math:`\lceil \sqrt{d} \rceil` for :math:`d` features (rounded up, as in the
        original code) and ``None`` all features. Only features that vary in the node are
        drawn, so a node may draw fewer.
    max_candidates : int, default=1
        Number of split points drawn for each feature.
    n_jobs : int, default=None
        Number of jobs that build the trees in parallel. ``None`` means 1 unless in a
        :func:`joblib.parallel_backend` context, and ``-1`` uses all processors. The result
        does not depend on it.
    random_state : int, RandomState instance or None, default=None
        Seed for the features and split points drawn in each tree.

    Attributes
    ----------
    trees_ : list of Tree
        The fitted trees.
    unlabeled_prior_ : float
        Share of positives among the unlabeled training samples, :math:`\pi_\text{U}`, used
        by the risk: derived from ``prior`` according to ``scenario``.
    max_features_ : int
        Number of features drawn at each split.
    feature_importances_ : ndarray of shape (`n_features_in_`,)
        Risk reduction of each feature: the sum of the risk decreases of the splits on it,
        averaged over the trees, as in the original code. It is not normalized, and with
        ``risk_estimator="uPU"`` it can be infinite (see Notes).
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
    NNPUBoost : Gradient boosting on the nnPU objective of the original nnPU experiments.
    PUHellingerForest : A random forest with a PU Hellinger-distance split criterion.

    Notes
    -----
    ``predict_proba`` is the mean over the trees of the leaf estimates :math:`v^*`, clipped
    to ``[0, 1]``, as PU-Bench computes it, except in leaves without unlabeled samples:
    there :math:`v^*` is infinite and PU-Bench counts 0.5, PyPU 1, as the leaf's vote in the
    original code. With ``risk_estimator="uPU"`` the trees grow until each leaf holds only
    labeled or only unlabeled samples, so PU-Bench's rule would keep every probability at
    or below 0.5. ``predict`` thresholds the probability at 0.5. The original code predicts
    instead by a majority vote of the trees, each predicting positive where
    :math:`v^* > 0.5`; the two can differ near the decision boundary.

    In the uPU risk, a leaf holding labeled samples only has risk :math:`-\infty` with the
    quadratic loss, and so has any node with :math:`v^* > 1` with the logistic loss. A split
    creating such a leaf has an infinite risk reduction, which ``feature_importances_``
    then carries.

    References
    ----------
    .. [1] J. Wilton, A. M. Y. Koay, R. K. L. Ko, M. Xu and N. Ye. "Positive-unlabeled
       learning using random forests via recursive greedy risk minimization." NeurIPS 2022.
    .. [2] https://github.com/jonathanwilton/PUExtraTrees (MIT license).
    .. [3] https://github.com/XiXiphus/PU-Bench (MIT license).

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.metrics import roc_auc_score
    >>> from pypu.datasets import make_pu_split
    >>> from pypu.models import PUExtraTrees
    >>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
    >>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
    >>> forest = PUExtraTrees(prior=y_train.mean(), n_estimators=20, random_state=0)
    >>> forest = forest.fit(X_train, s_train)
    >>> float(roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])) > 0.9
    True
    """

    def __init__(
        self,
        prior=None,
        scenario="single",
        n_estimators=100,
        risk_estimator="nnPU",
        loss="quadratic",
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        max_candidates=1,
        n_jobs=None,
        random_state=None,
    ):
        self.prior = prior
        self.scenario = scenario
        self.n_estimators = n_estimators
        self.risk_estimator = risk_estimator
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_candidates = max_candidates
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _fit(self, X, s):
        params = self._tree_parameters(X.shape[1])
        n_estimators = _positive_int(self.n_estimators, "n_estimators")
        self.unlabeled_prior_ = _unlabeled_prior(self, s)
        self.max_features_ = params["max_features"]

        # As scikit-learn's forests: one seed per tree, drawn in turn.
        random_state = check_random_state(self.random_state)
        seeds = [random_state.randint(_MAX_INT) for _ in range(n_estimators)]
        X = np.asarray(X, dtype=np.float64)
        labeled = s == 1
        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(build_tree)(X, labeled, self.unlabeled_prior_, seed, **params) for seed in seeds
        )

        # As the original: each tree's importances, divided by the number of trees, in turn.
        importances = np.zeros(X.shape[1])
        for tree in self.trees_:
            importances += _tree_importances(tree, X.shape[1]) / n_estimators
        self.feature_importances_ = importances

    def _predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        values = np.stack([tree.value[tree.apply(X)] for tree in self.trees_])
        return np.mean(values, axis=0)

    def _tree_parameters(self, n_features):
        if self.risk_estimator not in _RISK_ESTIMATORS:
            raise ValueError(
                f"risk_estimator must be 'nnPU' or 'uPU', got {self.risk_estimator!r}."
            )
        if self.loss not in _LOSSES:
            raise ValueError(f"loss must be 'quadratic' or 'logistic', got {self.loss!r}.")
        max_depth = self.max_depth
        if max_depth is not None:
            max_depth = _positive_int(max_depth, "max_depth", allow_none=True)
        max_features = self.max_features
        if max_features is None:
            resolved = n_features
        elif max_features == "sqrt":
            resolved = int(np.ceil(np.sqrt(n_features)))
        elif isinstance(max_features, numbers.Integral) and not isinstance(max_features, bool):
            if not 1 <= max_features <= n_features:
                raise ValueError(
                    f"max_features={max_features} must be between 1 and the number of "
                    f"features ({n_features})."
                )
            resolved = int(max_features)
        else:
            raise ValueError(f"max_features must be 'sqrt', an int or None, got {max_features!r}.")
        return {
            "risk_estimator": self.risk_estimator,
            "loss": self.loss,
            "max_depth": max_depth,
            "min_samples_leaf": _positive_int(self.min_samples_leaf, "min_samples_leaf"),
            "max_features": resolved,
            "max_candidates": _positive_int(self.max_candidates, "max_candidates"),
        }


def build_tree(
    X,
    labeled,
    prior,
    seed,
    *,
    risk_estimator,
    loss,
    max_depth,
    min_samples_leaf,
    max_features,
    max_candidates,
):
    """Grow one PU extra tree, drawing at random as the original ``PUExtraTree.fit``.

    The original draws from NumPy's global generator: seeding it with ``seed`` before
    ``PUExtraTree.fit`` gives this tree.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training samples, float64.
    labeled : ndarray of shape (n_samples,)
        Boolean mask of the labeled samples, the positive set; the others form the
        unlabeled set.
    prior : float
        Share of positives among the unlabeled samples, the original code's ``pi``.
    seed : int
        Seed of the tree's ``RandomState``.
    risk_estimator, loss, max_depth, min_samples_leaf, max_features, max_candidates
        As in :class:`PUExtraTrees`, validated; ``max_features`` is a count.

    Returns
    -------
    tree : Tree
        The fitted tree.
    """
    random_state = np.random.RandomState(seed)
    n_labeled = int(np.count_nonzero(labeled))
    n_unlabeled = labeled.size - n_labeled

    def risk(n_l, n_u):
        """Return ``(risk, vstar)`` of nodes with ``n_l`` labeled and ``n_u`` unlabeled."""
        n_l = np.asarray(n_l, dtype=np.float64)
        n_u = np.asarray(n_u, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_pos = n_l * prior / n_labeled
            w_neg = n_u / n_unlabeled - w_pos
            weight = w_pos + w_neg
            vstar = np.where(weight == 0, np.inf, w_pos / weight)
            if loss == "quadratic":
                value = 4 * weight * vstar * (1 - vstar)
                if risk_estimator == "uPU":
                    value = np.where(vstar == np.inf, -np.inf, value)
                else:
                    value = np.where(vstar > 1, 0.0, value)
            else:
                value = weight * (-vstar * np.log(vstar) - (1 - vstar) * np.log(1 - vstar))
                value = np.where((vstar == 0) | (vstar == 1), 0.0, value)
                value = np.where(vstar > 1, -np.inf if risk_estimator == "uPU" else 0.0, value)
        return value, vstar

    children_left, children_right, feature, threshold = [_LEAF], [_LEAF], [_LEAF], [np.nan]
    vstar_leaf, reduction = [np.nan], [0.0]
    # Depth first, left subtree first, as the original's recursion.
    stack = [(0, 0, np.arange(labeled.size))]
    while stack:
        node, depth, rows = stack.pop()
        X_node, labeled_node = X[rows], labeled[rows]
        n_l = int(np.count_nonzero(labeled_node))
        n_u = rows.size - n_l
        node_risk, node_vstar = risk(n_l, n_u)
        node_risk, node_vstar = float(node_risk), float(node_vstar)
        # uPU nodes with labeled samples can have a negative risk and still be split.
        lowest_risk = 0.0 if risk_estimator == "nnPU" or n_l == 0 else -np.inf
        can_improve = node_risk > lowest_risk
        # A node can only be empty when a split point rounds to a feature's highest value,
        # which takes adjacent floating-point values; the original then fails.
        spread = np.ptp(X_node, axis=0) if rows.size else np.zeros(X.shape[1])
        if not (
            can_improve
            and (max_depth is None or depth < max_depth)
            and min_samples_leaf < rows.size
            and spread.sum() > 0
        ):
            if node_vstar == 0.5:
                # The original breaks the tie of the leaf's vote at random; the vote is not
                # used here, but the draw is kept so that the trees stay the same.
                random_state.binomial(1, 0.5)
            vstar_leaf[node] = node_vstar
            continue

        varying = np.flatnonzero(spread > 0)
        features = random_state.choice(varying, size=min(max_features, varying.size), replace=False)
        columns = X_node[:, features]
        # The two lowest and two highest distinct values of each feature in the node.
        lowest, highest = columns.min(axis=0), columns.max(axis=0)
        second_lowest = np.where(columns > lowest, columns, np.inf).min(axis=0)
        second_highest = np.where(columns < highest, columns, -np.inf).max(axis=0)
        low = lowest + 2 * (second_lowest - lowest) / 5
        high = second_highest + 3 * (highest - second_highest) / 5
        cuts = random_state.uniform(np.repeat(low, max_candidates), np.repeat(high, max_candidates))
        candidates = np.repeat(features, max_candidates)

        goes_left = X_node[:, candidates] <= cuts
        left_l = np.count_nonzero(goes_left & labeled_node[:, np.newaxis], axis=0)
        left_u = np.count_nonzero(goes_left, axis=0) - left_l
        split_risk = risk(left_l, left_u)[0] + risk(n_l - left_l, n_u - left_u)[0]
        best = int(np.argmin(split_risk))

        left = len(feature)
        for nodes, value in (
            (children_left, _LEAF),
            (children_right, _LEAF),
            (feature, _LEAF),
            (threshold, np.nan),
            (vstar_leaf, np.nan),
            (reduction, 0.0),
        ):
            nodes.extend((value, value))
        children_left[node], children_right[node] = left, left + 1
        feature[node], threshold[node] = int(candidates[best]), float(cuts[best])
        reduction[node] = node_risk - float(split_risk[best])
        mask = goes_left[:, best]
        stack.append((left + 1, depth + 1, rows[~mask]))
        stack.append((left, depth + 1, rows[mask]))

    vstar = np.array(vstar_leaf)
    # Infinite in leaves without unlabeled samples, which then count as positive.
    value = np.clip(vstar, 0.0, 1.0)
    return Tree(
        children_left=np.array(children_left, dtype=np.intp),
        children_right=np.array(children_right, dtype=np.intp),
        feature=np.array(feature, dtype=np.intp),
        threshold=np.array(threshold),
        vstar=vstar,
        value=value,
        risk_reduction=np.array(reduction),
    )


def _tree_importances(tree, n_features):
    """Sum each feature's risk reductions over the splits of ``tree``, node by node."""
    importances = np.zeros(n_features)
    internal = tree.feature != _LEAF
    np.add.at(importances, tree.feature[internal], tree.risk_reduction[internal])
    return importances


def _unlabeled_prior(model, s):
    """Return the share of positives among the unlabeled samples implied by the prior."""
    if model.scenario not in _SCENARIOS:
        raise ValueError(f"scenario must be one of {list(_SCENARIOS)}, got {model.scenario!r}.")
    if model.prior is None:
        raise ValueError(
            f"{type(model).__name__} needs the class prior: pass prior=P(y=1), the fraction "
            "of positives in the population."
        )
    prior = check_prior(model.prior)
    if model.scenario == "case_control":
        return prior
    n_samples, n_labeled = s.size, int(np.count_nonzero(s))
    unlabeled_prior = (prior * n_samples - n_labeled) / (n_samples - n_labeled)
    if unlabeled_prior <= 0:
        raise ValueError(
            f"prior={prior} must be above the fraction of labeled samples "
            f"({n_labeled / n_samples:.4f}): {type(model).__name__} needs positives among the "
            "unlabeled samples. Check that you passed the class prior P(y=1) and not the "
            "label frequency P(s=1 | y=1), or pass scenario='case_control' if the labeled "
            "positives were drawn apart from the unlabeled samples."
        )
    return unlabeled_prior


def _positive_int(value, name, *, allow_none=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        expected = "None or an int >= 1" if allow_none else "an int >= 1"
        raise ValueError(f"{name} must be {expected}, got {value!r}.")
    return int(value)
