"""Building and applying PU Hellinger trees, in NumPy.

The builder follows the depth-first builder and the best splitter of scikit-learn 1.0, which
the original PU Hellinger tree code forked, step for step: the random draws of features, the
handling of constant features, the float32 comparisons and the order in which nodes are
built. So, given the same data and seed, it builds the same trees. Only the split criterion
differs from scikit-learn's: the PU Hellinger distance of :func:`_hellinger_children`.
"""

# Adapted from scikit-learn's sklearn/tree/_splitter.pyx, _tree.pyx and _utils.pyx
# (BSD-3-Clause), as forked by the PU Hellinger trees code. See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import numpy as np

_RAND_R_MAX = 0x7FFFFFFF
# scikit-learn stores it as float32, so the comparisons it enters are made in float32.
_FEATURE_THRESHOLD = np.float32(1e-7)
_EPSILON = np.finfo(np.float64).eps
TREE_LEAF = -1
TREE_UNDEFINED = -2


class _RandR:
    """The xorshift generator scikit-learn's trees use to draw features (``our_rand_r``)."""

    def __init__(self, seed: int):
        self.state = int(seed)

    def rand_int(self, low: int, high: int) -> int:
        """Return a random integer in ``[low, high)``."""
        state = self.state or 1
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        self.state = state
        return low + (state % (_RAND_R_MAX + 1)) % (high - low)


def _hellinger_children(left, right, weight):
    """Return the PU Hellinger impurities of the left and right children of candidate splits.

    ``left`` and ``right`` have shape ``(2, n_candidates)``: the weighted numbers of unlabeled
    and of labeled samples in each child. ``weight`` is ``prior / P(s = 1)``, the inverse
    label frequency: each child is estimated to hold ``min(n, weight * n_labeled)``
    positives. A split is scored by the sum of the two impurities, the squared Hellinger
    distance between the positive and the negative class over the two children; it is
    ``-inf`` when the node has no labeled sample. The arithmetic follows the original
    criterion operation by operation.
    """
    n_left = left[0] + left[1]
    n_right = right[0] + right[1]
    weighted_left = weight * left[1]
    weighted_right = weight * right[1]
    capped_left = weighted_left > n_left
    capped_right = weighted_right > n_right
    pos_left = np.where(capped_left, n_left, weighted_left)
    neg_left = np.where(capped_left, 0.0, n_left - pos_left)
    pos_right = np.where(capped_right, n_right, weighted_right)
    neg_right = np.where(capped_right, 0.0, n_right - pos_right)

    n_neg = neg_right + neg_left
    n_pos = pos_right + pos_left
    has_neg = n_neg > 0
    has_pos = n_pos > 0
    n_neg = np.where(n_neg == 0.0, 1.0, n_neg)
    n_pos = np.where(n_pos == 0.0, 1.0, n_pos)

    impurity_left = (
        np.where(has_neg, np.sqrt(neg_left / n_neg), 0.0)
        - np.where(has_pos, np.sqrt(pos_left / n_pos), 0.0)
    ) ** 2
    impurity_right = (
        np.where(has_neg, np.sqrt(neg_right / n_neg), 0.0)
        - np.where(has_pos, np.sqrt(pos_right / n_pos), 0.0)
    ) ** 2
    impurity_left = np.where(has_pos, impurity_left, -np.inf)
    impurity_right = np.where(has_pos, impurity_right, -np.inf)
    return impurity_left, impurity_right


class _Splitter:
    """scikit-learn's best splitter, with its state shared across the nodes of a tree."""

    def __init__(self, X, class_weights, *, weight, max_features, min_samples_leaf, seed):
        self.X = X  # float32, (n_samples, n_features)
        self.class_weights = class_weights  # (2, n_samples): weights of unlabeled, labeled
        self.weight = weight
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.rng = _RandR(seed)
        n_features = X.shape[1]
        self.features = list(range(n_features))
        self.constant_features = [0] * n_features

    def node_split(self, samples, n_known_constants):
        """Return the best split of the node and the updated number of constant features.

        The split is ``None`` when no valid split exists, else ``(feature, threshold,
        impurity_left, impurity_right, improvement)``.
        """
        features, rng = self.features, self.rng
        n_samples = samples.size
        f_i = len(features)
        n_visited = n_found_constants = n_drawn_constants = 0
        n_total_constants = n_known_constants
        best = None
        best_proxy = -np.inf

        while f_i > n_total_constants and (
            n_visited < self.max_features or n_visited <= n_found_constants + n_drawn_constants
        ):
            n_visited += 1
            f_j = rng.rand_int(n_drawn_constants, f_i - n_found_constants)
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = (
                    features[f_j],
                    features[n_drawn_constants],
                )
                n_drawn_constants += 1
                continue
            f_j += n_found_constants
            feature = features[f_j]
            values = self.X[samples, feature]
            order = np.argsort(values, kind="stable")
            xf = values[order]
            if xf[-1] <= xf[0] + _FEATURE_THRESHOLD:
                features[f_j], features[n_total_constants] = (
                    features[n_total_constants],
                    features[f_j],
                )
                n_found_constants += 1
                n_total_constants += 1
                continue
            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]

            # Candidate positions p split xf[:p] | xf[p:] between distinct values.
            positions = np.flatnonzero(xf[1:] > xf[:-1] + _FEATURE_THRESHOLD) + 1
            msl = self.min_samples_leaf
            positions = positions[(positions >= msl) & (n_samples - positions >= msl)]
            if positions.size == 0:
                continue
            cumulative = np.cumsum(self.class_weights[:, samples[order]], axis=1)
            left = cumulative[:, positions - 1]
            right = cumulative[:, -1:] - left
            impurity_left, impurity_right = _hellinger_children(left, right, self.weight)
            proxy = impurity_right + impurity_left
            k = int(np.argmax(proxy))
            if proxy[k] > best_proxy:
                best_proxy = proxy[k]
                p = positions[k]
                threshold = float(xf[p - 1]) / 2.0 + float(xf[p]) / 2.0
                if threshold == float(xf[p]) or np.isinf(threshold):
                    threshold = float(xf[p - 1])
                best = (feature, threshold, impurity_left[k], impurity_right[k])

        # Constant features keep their order for the descendants (scikit-learn's invariant).
        features[:n_known_constants] = self.constant_features[:n_known_constants]
        self.constant_features[n_known_constants:n_total_constants] = features[
            n_known_constants:n_total_constants
        ]
        if best is None:
            return None, n_total_constants
        feature, threshold, impurity_left, impurity_right = best
        improvement = float(np.sqrt(impurity_right + impurity_left))
        return (feature, threshold, impurity_left, impurity_right, improvement), n_total_constants


def build_tree(
    X,
    s,
    sample_weight,
    *,
    prior,
    max_features,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    min_impurity_decrease,
    seed,
):
    """Build a PU Hellinger tree; return its arrays.

    ``X`` is float32. ``s`` holds the PU labels of all training samples and
    ``sample_weight`` their weights (bootstrap counts in a forest), ``None`` meaning ones;
    samples of weight 0 are left out of the tree. ``seed`` is the generator state that
    scikit-learn draws as ``random_state.randint(0, RAND_R_MAX)``.
    """
    n_total = s.size
    n_unlabeled = float(np.sum(s == 0))
    # The inverse label frequency, computed as the original does, on all training samples.
    weight = prior / ((n_total - n_unlabeled) / n_total)
    if sample_weight is None:
        sample_weight = np.ones(n_total)
    class_weights = np.vstack([sample_weight * (s == 0), sample_weight * (s == 1)])
    splitter = _Splitter(
        X,
        class_weights,
        weight=weight,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )
    min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

    nodes = {
        "children_left": [],
        "children_right": [],
        "feature": [],
        "threshold": [],
        "value": [],
        "n_node_samples": [],
        "improvement": [],
    }
    # Stack records: samples, depth, parent, is_left, impurity, n_constant_features.
    stack = [(np.flatnonzero(sample_weight != 0), 0, TREE_UNDEFINED, False, np.inf, 0)]
    first = True
    while stack:
        samples, depth, parent, is_left, impurity, n_constant_features = stack.pop()
        n_node_samples = samples.size
        value = class_weights[:, samples].sum(axis=1)
        is_leaf = depth >= max_depth or n_node_samples < min_samples_split
        if first:
            impurity = 2.0  # the original's node impurity: the number of classes
            first = False
        # The original's min_impurity_split of 0 makes children with zero impurity leaves.
        is_leaf = is_leaf or impurity <= 0.0
        split = None
        if not is_leaf:
            split, n_constant_features = splitter.node_split(samples, n_constant_features)
            is_leaf = split is None or split[4] + _EPSILON < min_impurity_decrease

        node_id = len(nodes["feature"])
        if parent != TREE_UNDEFINED:
            side = "children_left" if is_left else "children_right"
            nodes[side][parent] = node_id
        nodes["children_left"].append(TREE_LEAF)
        nodes["children_right"].append(TREE_LEAF)
        nodes["value"].append(value)
        nodes["n_node_samples"].append(n_node_samples)
        if is_leaf:
            nodes["feature"].append(TREE_UNDEFINED)
            nodes["threshold"].append(float(TREE_UNDEFINED))
            nodes["improvement"].append(0.0)
            continue
        feature, threshold, impurity_left, impurity_right, improvement = split
        nodes["feature"].append(feature)
        nodes["threshold"].append(threshold)
        nodes["improvement"].append(improvement)
        goes_left = X[samples, feature] <= threshold
        stack.append(
            (samples[~goes_left], depth + 1, node_id, False, impurity_right, n_constant_features)
        )
        stack.append(
            (samples[goes_left], depth + 1, node_id, True, impurity_left, n_constant_features)
        )

    return Tree(
        children_left=np.asarray(nodes["children_left"], dtype=np.intp),
        children_right=np.asarray(nodes["children_right"], dtype=np.intp),
        feature=np.asarray(nodes["feature"], dtype=np.intp),
        threshold=np.asarray(nodes["threshold"], dtype=np.float64),
        value=np.asarray(nodes["value"], dtype=np.float64).reshape(-1, 2),
        n_node_samples=np.asarray(nodes["n_node_samples"], dtype=np.intp),
        improvement=np.asarray(nodes["improvement"], dtype=np.float64),
    )


class Tree:
    """The arrays of a fitted tree, indexed by node, in scikit-learn's layout.

    Attributes
    ----------
    children_left, children_right : ndarray of int
        Children of each node; ``-1`` for leaves.
    feature : ndarray of int
        Feature each node splits on; ``-2`` for leaves.
    threshold : ndarray of float
        Samples with ``X[:, feature] <= threshold`` go left; ``-2`` for leaves.
    value : ndarray of shape (node_count, 2)
        Weighted numbers of unlabeled and of labeled samples in each node.
    n_node_samples : ndarray of int
        Number of (distinct) training samples in each node.
    improvement : ndarray of float
        Hellinger distance of each node's split; ``0`` for leaves.
    """

    def __init__(
        self,
        *,
        children_left,
        children_right,
        feature,
        threshold,
        value,
        n_node_samples,
        improvement,
    ):
        self.children_left = children_left
        self.children_right = children_right
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.n_node_samples = n_node_samples
        self.improvement = improvement

    @property
    def node_count(self) -> int:
        """Number of nodes."""
        return int(self.feature.size)

    def apply(self, X):
        """Return the leaf reached by each row of ``X`` (float32)."""
        node = np.zeros(X.shape[0], dtype=np.intp)
        rows = np.arange(X.shape[0])
        left, right = self.children_left, self.children_right
        active = left[node] != TREE_LEAF
        while active.any():
            idx = rows[active]
            current = node[idx]
            goes_left = X[idx, self.feature[current]] <= self.threshold[current]
            node[idx] = np.where(goes_left, left[current], right[current])
            active[idx] = left[node[idx]] != TREE_LEAF
        return node
