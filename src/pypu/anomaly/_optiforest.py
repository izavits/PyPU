"""OptIForest, the optimal isolation forest of Xiang et al.

Ported from the authors' code (https://github.com/xiagll/OptIForest, ``detectors/``; no
license is stated, see THIRD_PARTY_NOTICES.md), as copied unchanged into the iFPNU
experiments: the sampling, the locality-sensitive hash families, the trees, their
hierarchical regrouping and the scores. The original draws its random numbers from NumPy's
global generator (and, for ``branch=0``, from Python's ``random``); this port draws them from
one generator, in the same order, and scores many samples at once.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.spatial import distance
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

_POOL_SIZE = 50  # hash functions drawn per tree
_BIN_WIDTH = 4  # of the E2LSH hash functions


def _euclidean_rows(u, v):
    diff = u - v
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def _cityblock_rows(u, v):
    return np.abs(u - v).sum(axis=1)


def _cosine_rows(u, v):
    u, v = np.broadcast_arrays(u, v)
    uv = np.einsum("ij,ij->i", u, v)
    uu = np.einsum("ij,ij->i", u, u)
    vv = np.einsum("ij,ij->i", v, v)
    return np.clip(1.0 - uv / np.sqrt(uu * vv), 0.0, 2.0)


# lsh_family -> (hash family, distance of the regrouping, the same distance row by row).
_FAMILIES = {
    "L2OPT": ("normal", distance.euclidean, _euclidean_rows),
    "L1OPT": ("cauchy", distance.cityblock, _cityblock_rows),
    "ALOPT": ("angle", distance.cosine, _cosine_rows),
}


class _HashPool:
    """The hash functions of one tree: E2LSH (L2OPT, L1OPT) or random hyperplanes (ALOPT)."""

    def __init__(self, family, n_features, size, rng):
        self.family = family
        weights, offsets = [], []
        for _ in range(size):
            if family == "cauchy":
                weights.append(rng.standard_cauchy(n_features))
            else:
                weights.append(rng.normal(0.0, 1.0, n_features))
            if family != "angle":
                offsets.append(rng.uniform(0.0, _BIN_WIDTH))
        self.weights = np.array(weights)
        self.offsets = np.array(offsets)

    def __call__(self, X, index):
        projection = X @ self.weights[index]
        if self.family == "angle":
            return np.where(projection < 0, -1, 1)
        return np.floor((projection + self.offsets[index]) / _BIN_WIDTH).astype(np.int64)


class _HashNode:
    """A node of the hash tree: its children by hash value, in order of first appearance."""

    __slots__ = ("adjust", "center", "children", "hash_index", "size")

    def __init__(self, center, size, children, hash_index):
        self.center = center
        self.size = size
        self.children = children
        self.hash_index = hash_index
        self.adjust = 0.0  # average path length below a leaf; set once the tree is grown


class _GroupNode:
    """A node of the regrouping: a sample goes to the child its center would move least."""

    __slots__ = ("center", "centers", "children", "size", "sizes")

    def __init__(self, children):
        self.children = children
        self.centers = np.array([child.center for child in children])
        self.sizes = np.array([child.size for child in children])
        self.size = sum(child.size for child in children)
        weighted = [child.center * child.size for child in children]
        self.center = np.sum(weighted, axis=0) / self.size


class _Tree:
    """One OptIForest tree: a hash tree, regrouped above its subtrees of ``threshold`` samples."""

    def __init__(self, X, hashes, threshold, branch, granularity, distances, rng):
        self.hashes = hashes
        self.granularity = granularity
        self.distance, self.distance_rows = distances
        self.depth_limit = 2 * np.log2(X.shape[0]) + 0.8327
        root = self._grow(X, 0)
        n_internal, n_children = self._count_branches(root)
        self.branch_factor = 2.0 if n_internal == 0 else n_children * 1.0 / n_internal
        self.reference = self._average_path_length(X.shape[0])
        self._set_leaf_adjustments(root)
        groups = [] if root.size <= threshold else self._groups(root, threshold)
        self.root = self._regroup(root, groups, branch, rng)

    def _grow(self, X, hash_index):
        n_samples = X.shape[0]
        center = np.sum(X, axis=0) / n_samples
        if n_samples == 1 or hash_index > self.depth_limit:
            return _HashNode(center, n_samples, {}, hash_index)
        index = hash_index
        keys = self.hashes(X, index)
        # Hash functions that keep all the samples together are skipped.
        while np.all(keys == keys[0]) and index <= self.depth_limit:
            index += 1
            keys = self.hashes(X, index)
        if index > self.depth_limit:
            return _HashNode(center, n_samples, {}, index)
        values, first = np.unique(keys, return_index=True)
        children = {}
        for value in values[np.argsort(first)]:
            children[int(value)] = self._grow(X[keys == value], index + 1)
        return _HashNode(center, n_samples, children, index)

    def _count_branches(self, node):
        if not node.children:
            return 0, 0
        n_internal, n_children = 1, len(node.children)
        for child in node.children.values():
            internal, children = self._count_branches(child)
            n_internal += internal
            n_children += children
        return n_internal, n_children

    def _average_path_length(self, n_samples):
        """Average path length of a random tree of ``n_samples`` with this branching factor."""
        if n_samples <= 1:
            return 0
        if n_samples <= round(self.branch_factor):
            return 1
        numerator = np.log(n_samples) + np.log(self.branch_factor - 1.0) + 0.5772
        return numerator / np.log(self.branch_factor) - 0.5

    def _set_leaf_adjustments(self, node):
        if not node.children:
            node.adjust = self._average_path_length(node.size)
        for child in node.children.values():
            self._set_leaf_adjustments(child)

    def _groups(self, node, threshold, found=None):
        """Find the largest subtrees of at most ``threshold`` samples, and the larger leaves."""
        found = [] if found is None else found
        if not node.children:
            found.append(node)
        for child in node.children.values():
            if node.size > threshold and child.size <= threshold:
                found.append(child)
            else:
                self._groups(child, threshold, found)
        return found

    def _regroup(self, root, groups, branch, rng):
        """Merge the closest groups until at most ``branch`` remain, under a new root."""
        if len(groups) <= 1:
            return root
        groups = list(groups)
        while len(groups) > 1:
            n_merged = branch
            if n_merged == 0:
                n_merged = 2 if rng.random_sample() <= 3 - math.e else 3
            if n_merged >= len(groups):
                return _GroupNode(groups)
            merged = self._closest(groups, n_merged)
            node = _GroupNode([groups[i] for i in merged])
            for i in merged:  # in decreasing order
                del groups[i]
            groups.append(node)
        return groups[0]  # pragma: no cover - the loop always returns

    def _closest(self, groups, n_merged):
        """Find the two (``n_merged=2``) or three closest groups; their indices, largest first."""
        best, closest = math.inf, None
        n_groups = len(groups)
        if n_merged == 2:
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    if groups[i].size == groups[j].size:
                        spread = self.distance(groups[i].center, groups[j].center) * groups[i].size
                    else:
                        spread = self._spread(groups, [i, j])
                    if spread < best:
                        best, closest = spread, [j, i]
        else:
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    for h in range(j + 1, n_groups):
                        spread = self._spread(groups, [i, j, h])
                        if spread < best:
                            best, closest = spread, [h, j, i]
        return closest

    def _spread(self, groups, indices):
        """Size-weighted distance of the groups' centers to their common center."""
        size = 0
        center = np.zeros(len(groups[0].center))
        for i in indices:
            size += groups[i].size
            center += groups[i].center * groups[i].size
        center = center / size
        spread = 0
        for i in indices:
            spread += self.distance(center, groups[i].center) * groups[i].size
        return spread

    def scores(self, X):
        """Per-tree anomaly scores: 2 to the minus the path length over the reference."""
        paths = np.empty(X.shape[0])
        self._descend(self.root, np.arange(X.shape[0]), 0, X, paths)
        return 2.0 ** (-1.0 * paths / self.reference)

    def _descend(self, node, rows, depth, X, paths):
        if rows.size == 0:
            return
        g = self.granularity
        if isinstance(node, _GroupNode):
            nearest = self._nearest(node, X[rows])
            for position, child in enumerate(node.children):
                self._descend(child, rows[nearest == position], depth + 1, X, paths)
        elif not node.children:
            paths[rows] = depth * np.power(1.0 * node.hash_index / max(depth, 1.0), g) + node.adjust
        else:
            keys = self.hashes(X[rows], node.hash_index)
            found = np.zeros(rows.size, dtype=bool)
            for value, child in node.children.items():
                here = keys == value
                found |= here
                self._descend(child, rows[here], depth + 1, X, paths)
            # A hash value unseen in training ends the path one level down.
            depth += 1
            paths[rows[~found]] = depth * np.power(1.0 * (node.hash_index + 1) / max(depth, 1), g)

    def _nearest(self, node, X):
        """Position of the child whose center the sample would move least, if added to it."""
        best = np.full(X.shape[0], np.inf)
        nearest = np.zeros(X.shape[0], dtype=np.intp)
        for position, (center, size) in enumerate(zip(node.centers, node.sizes, strict=True)):
            moved = (center * size + X) / (size + 1)
            cost = self.distance_rows(X, moved) + self.distance_rows(center, moved) * size
            better = cost < best
            best[better] = cost[better]
            nearest[better] = position
        return nearest


class OptIForest(BaseEstimator):
    r"""OptIForest: an isolation forest of hash trees, regrouped by clustering.

    OptIForest [1]_ isolates samples with locality-sensitive hash functions instead of random
    thresholds. Each tree is grown on ``max_samples`` training samples: a node splits its
    samples by the value of a hash function, into as many children as values, skipping the
    functions that leave the samples together, until a node holds one sample or the depth
    exceeds :math:`2 \log_2 \psi + 0.8327`, with :math:`\psi` samples per tree. The tree is
    then regrouped above its largest subtrees of at most ``threshold`` samples: those are
    merged bottom-up by clustering, the closest first, until at most ``branch`` groups remain
    under a new root. Below it, a sample goes to the group whose center it would move least.

    The anomaly score of a sample is the mean over the trees of :math:`2^{-h(x) / c}`, in
    :math:`(0, 1]`, where :math:`h(x)` is the number of hash functions on its path, plus the
    average path length below the leaf it reaches, and :math:`c` the average path length of a
    tree with the same mean branching factor.

    It ports the authors' code. The defaults are the settings with which IFPNU's original
    experiments used it, and ``IFPNU(detector="OIF")`` uses them.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_samples : int, default=512
        Number of training samples per tree, drawn without replacement, at least 2; lowered
        to the number of samples when larger.
    lsh_family : {"L2OPT", "L1OPT", "ALOPT"}, default="L2OPT"
        Hash functions and the distance of the regrouping: ``"L2OPT"``, E2LSH with Gaussian
        projections and the Euclidean distance; ``"L1OPT"``, E2LSH with Cauchy projections
        and the Manhattan distance; ``"ALOPT"``, the sign of random projections and the
        cosine distance.
    threshold : int, default=403
        Largest number of training samples in a subtree that the regrouping keeps whole.
        Trees of at most ``threshold`` samples are not regrouped.
    branch : int, default=20
        Largest number of children of the new root, ``0`` or at least 2. The groups are
        merged three at a time, or two at a time for ``branch=2``; ``0`` chooses two
        (with probability :math:`3 - e`) or three at random for each merge.
    granularity : float, default=1
        How a path counts the hash functions that left the samples together: with ``1`` it
        counts them all, with ``0`` it counts only the nodes on the path.
    random_state : int, RandomState instance or None, default=None
        Seed for the samples of each tree, the hash functions and, with ``branch=0``, the
        merges. ``None`` uses NumPy's global generator, as the original does.

    Attributes
    ----------
    max_samples_ : int
        Number of training samples per tree.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.

    See Also
    --------
    ExtendedIsolationForest : Isolation trees whose splits have random orientations.
    pypu.models.IFPNU : Takes the samples it scores as most anomalous as possible positives.

    Notes
    -----
    The random numbers are drawn in the original's order, from one generator, so that from the
    same NumPy seed it gives the same scores (with ``branch=0``, when the original's draws
    from Python's ``random`` come from NumPy too).

    References
    ----------
    .. [1] H. Xiang, X. Zhang, H. Hu, L. Qi, W. Dou, M. Dras, A. Beheshti and X. Xu.
       "OptIForest: optimal isolation forest for anomaly detection." IJCAI 2023.

    Examples
    --------
    >>> import numpy as np
    >>> from pypu.anomaly import OptIForest
    >>> X = np.random.default_rng(0).normal(size=(500, 2))
    >>> X[0] = [6.0, 6.0]
    >>> detector = OptIForest(n_estimators=20, random_state=0).fit(X)
    >>> int(np.argmin(detector.score_samples(X)))  # the lowest score: the most anomalous
    0
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples=512,
        lsh_family="L2OPT",
        threshold=403,
        branch=20,
        granularity=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.lsh_family = lsh_family
        self.threshold = threshold
        self.branch = branch
        self.granularity = granularity
        self.random_state = random_state

    def fit(self, X, y=None):
        """Grow the trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : None
            Ignored.

        Returns
        -------
        self : object
            The fitted detector.
        """
        X = validate_data(self, X, reset=True, dtype=np.float64, ensure_min_samples=2)
        n_samples, n_features = X.shape
        n_estimators = _check_int(self.n_estimators, "n_estimators", 1)
        max_samples = _check_int(self.max_samples, "max_samples", 2)
        threshold = _check_int(self.threshold, "threshold", 1)
        branch = _check_int(self.branch, "branch", 0)
        if branch == 1:
            raise ValueError("branch must be 0 or an int >= 2, got 1.")
        granularity = self.granularity
        if (
            isinstance(granularity, bool)
            or not isinstance(granularity, numbers.Real)
            or not 0 <= granularity < math.inf
        ):
            raise ValueError(f"granularity must be a number >= 0, got {granularity!r}.")
        if self.lsh_family not in _FAMILIES:
            names = ", ".join(repr(name) for name in _FAMILIES)
            raise ValueError(f"lsh_family must be one of {names}, got {self.lsh_family!r}.")
        family, *distances = _FAMILIES[self.lsh_family]

        max_samples = min(n_samples, max_samples)
        # The deepest hash function a tree can use; the original draws 50, and more on demand.
        depth_limit = 2 * np.log2(max_samples) + 0.8327
        pool_size = max(_POOL_SIZE, math.floor(depth_limit) + 2)
        rng = check_random_state(self.random_state)
        # In the original's order: the samples of every tree, then the hash functions of every
        # tree, then the trees.
        samples = [_draw_samples(n_samples, max_samples, rng) for _ in range(n_estimators)]
        pools = [_HashPool(family, n_features, pool_size, rng) for _ in range(n_estimators)]
        self._trees = [
            _Tree(X[rows], pool, threshold, branch, granularity, distances, rng)
            for rows, pool in zip(samples, pools, strict=True)
        ]
        self.max_samples_ = max_samples
        return self

    def score_samples(self, X):
        """Return the opposite of the anomaly score of each sample.

        As in scikit-learn's :class:`~sklearn.ensemble.IsolationForest`, the lower, the more
        abnormal: the values are the mean per-tree scores negated, in :math:`[-1, 0)`, the
        original's ``decision_function``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The opposite of the anomaly scores.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        total = np.zeros(X.shape[0])
        for tree in self._trees:  # summed tree by tree, as the original does
            total += tree.scores(X)
        total /= len(self._trees)
        return -1.0 * total


def _draw_samples(n_samples, size, rng):
    """Draw ``size`` sample indices without replacement, one at a time, as the original."""
    remaining = list(range(n_samples))
    drawn = [remaining.pop(rng.randint(0, len(remaining), 1)[0]) for _ in range(size)]
    return np.sort(drawn)


def _check_int(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}.")
    return int(value)
