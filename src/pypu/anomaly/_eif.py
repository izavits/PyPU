"""The extended isolation forest of Hariri, Carrasco Kind and Brunner.

Copyright (c) 2018 University of Illinois at Urbana-Champaign. All rights reserved.
Developed by Matias Carrasco Kind and Sahand Hariri, NCSA/UIUC.

Ported from the authors' ``eif`` package, version 2.0.2 (https://github.com/sahandha/eif,
University of Illinois/NCSA Open Source License; see THIRD_PARTY_NOTICES.md): the trees, the
splits and the scores of its C++ ``iForest``, with the random numbers drawn in the order of its
pure-Python version, ``eif_old.py``. The package itself only installs from source, with a C++
compiler and Cython, which no longer works on current Python versions.
"""

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

# Euler's constant, rounded as in eif.
_EULER = 0.5772156649


def _c_factor(n):
    """Average path length of an unsuccessful search in a binary search tree of ``n`` nodes."""
    return 2.0 * (np.log(n - 1) + _EULER) - (2.0 * (n - 1.0) / (n * 1.0))


class _Tree:
    """One extended isolation tree, its nodes in arrays (preorder: left subtree first)."""

    def __init__(self, X, max_depth, extension_level, rng):
        n_features = X.shape[1]
        n_zero = n_features - extension_level - 1
        normals, points, children, leaf_paths = [], [], [], []
        # (samples, depth, parent, side): the left child is built before the right one, as in
        # eif's recursion, so that the random numbers are drawn in the same order.
        stack = [(X, 0, -1, 0)]
        while stack:
            X_node, depth, parent, side = stack.pop()
            node = len(children)
            if parent >= 0:
                children[parent][side] = node
            children.append([-1, -1])
            n_node = X_node.shape[0]
            if depth >= max_depth or n_node <= 1:
                normals.append(np.zeros(n_features))
                points.append(np.zeros(n_features))
                # An unsplit leaf adds the average depth at which it would isolate its samples.
                leaf_paths.append(_c_factor(n_node) if n_node > 1 else 0.0)
                continue
            # A random normal vector with extension_level + 1 non-zero coordinates, and a random
            # point of the node's bounding box for the hyperplane to pass through.
            zero = rng.choice(n_features, n_zero, replace=False)
            normal = rng.normal(0, 1, n_features)
            normal[zero] = 0
            point = rng.uniform(X_node.min(axis=0), X_node.max(axis=0))
            normals.append(normal)
            points.append(point)
            leaf_paths.append(np.nan)
            go_left = (X_node - point).dot(normal) < 0
            stack.append((X_node[~go_left], depth + 1, node, 1))
            stack.append((X_node[go_left], depth + 1, node, 0))
        self.normals = np.array(normals)
        self.points = np.array(points)
        self.children = np.array(children, dtype=np.intp)
        self.leaf_paths = np.array(leaf_paths)

    def path_lengths(self, X):
        """Depth at which each sample reaches a leaf, plus the leaf's average path length."""
        node = np.zeros(X.shape[0], dtype=np.intp)
        depth = np.zeros(X.shape[0])
        active = np.flatnonzero(self.children[node, 0] >= 0)
        while active.size:
            current = node[active]
            projection = np.einsum(
                "ij,ij->i", X[active] - self.points[current], self.normals[current]
            )
            node[active] = np.where(
                projection < 0, self.children[current, 0], self.children[current, 1]
            )
            depth[active] += 1
            active = active[self.children[node[active], 0] >= 0]
        return depth + self.leaf_paths[node]


class ExtendedIsolationForest(BaseEstimator):
    r"""Extended isolation forest: isolation trees whose splits have random orientations.

    An isolation forest scores a sample by how quickly random splits isolate it from the
    others: anomalies are isolated after few splits. The extended isolation forest [1]_ splits
    each node by a hyperplane with a random normal vector, instead of a threshold on one
    feature, which removes the bands that axis-parallel splits leave in the scores.
    ``extension_level`` sets how many coordinates of each normal vector can be non-zero:
    ``extension_level + 1``. With ``0`` the splits are axis-parallel, as in the standard
    isolation forest; with ``n_features - 1`` they can take any orientation.

    Each tree is grown on ``max_samples`` training samples, drawn without replacement, until a
    node holds at most one sample or reaches ``max_depth``. A node's hyperplane passes through
    a point drawn uniformly from the bounding box of the node's samples. The anomaly score of a
    sample is :math:`2^{-E[h(x)] / c(\psi)}`, in :math:`(0, 1]`, where :math:`h(x)` is the depth
    of the leaf that a tree puts it in, plus :math:`c(m)` for a leaf of :math:`m > 1` training
    samples, :math:`\psi` is the number of samples per tree, and
    :math:`c(m) = 2 (\ln(m - 1) + \gamma) - 2 (m - 1) / m` is the average path length of an
    unsuccessful search in a binary search tree.

    It ports the authors' ``eif`` package (version 2.0.2). The defaults are the settings with
    which IFPNU's original experiments used it, and ``IFPNU(detector="EIF")`` uses them.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_samples : "auto" or int, default="auto"
        Number of training samples per tree, at least 2. ``"auto"`` uses
        ``min(256, n_samples)``; a larger int than the number of samples is lowered to it,
        with a warning.
    extension_level : int, default=1
        Number of coordinates of each normal vector that can be non-zero, minus one, in
        ``[0, n_features - 1]``.
    max_depth : int or None, default=None
        Depth at which the trees stop growing. ``None`` uses
        :math:`\lceil \log_2 \psi \rceil`, the average depth of a binary search tree with
        :math:`\psi` nodes.
    random_state : int, RandomState instance or None, default=None
        Seed for the samples of each tree, the normal vectors and the points of the
        hyperplanes.

    Attributes
    ----------
    max_samples_ : int
        Number of training samples per tree.
    max_depth_ : int
        Depth at which the trees stopped growing.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``, when ``X`` has string feature names.

    See Also
    --------
    sklearn.ensemble.IsolationForest : The standard isolation forest.
    pypu.models.IFPNU : Takes the samples it scores as most anomalous as possible positives.

    Notes
    -----
    The random numbers are drawn as in eif's pure-Python version, ``eif_old.py``, so that, from
    the same NumPy seed and samples per tree, it gives the same scores. The C++ version, which
    the experiments ran without a seed, draws them differently: with another seed, its scores
    differ as much as two runs of the C++ version do.

    References
    ----------
    .. [1] S. Hariri, M. Carrasco Kind and R. J. Brunner. "Extended isolation forest." IEEE
       Transactions on Knowledge and Data Engineering, 33(4), 2021.

    Examples
    --------
    >>> import numpy as np
    >>> from pypu.anomaly import ExtendedIsolationForest
    >>> X = np.random.default_rng(0).normal(size=(500, 2))
    >>> X[0] = [6.0, 6.0]
    >>> detector = ExtendedIsolationForest(random_state=0).fit(X)
    >>> int(np.argmin(detector.score_samples(X)))  # the lowest score: the most anomalous
    0
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        extension_level=1,
        max_depth=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level
        self.max_depth = max_depth
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
        extension_level = _check_int(self.extension_level, "extension_level", 0)
        if extension_level > n_features - 1:
            raise ValueError(
                f"extension_level must be at most n_features - 1, got {extension_level} with "
                f"n_features = {n_features}."
            )
        if isinstance(self.max_samples, str) and self.max_samples == "auto":
            max_samples = min(256, n_samples)
        else:
            max_samples = _check_int(self.max_samples, "max_samples", 2, "'auto' or an int")
            if max_samples > n_samples:
                warnings.warn(
                    f"max_samples ({max_samples}) is larger than the number of samples "
                    f"({n_samples}); each tree uses all of them.",
                    UserWarning,
                    stacklevel=2,
                )
                max_samples = n_samples
        if self.max_depth is None:
            max_depth = math.ceil(math.log2(max_samples))
        else:
            max_depth = _check_int(self.max_depth, "max_depth", 0, "None or an int")

        rng = check_random_state(self.random_state)
        trees = []
        for _ in range(n_estimators):
            samples = rng.choice(n_samples, max_samples, replace=False)
            trees.append(_Tree(X[samples], max_depth, extension_level, rng))
        self._trees = trees
        self.max_samples_ = max_samples
        self.max_depth_ = max_depth
        return self

    def score_samples(self, X):
        r"""Return the opposite of the anomaly score of each sample.

        As in scikit-learn's :class:`~sklearn.ensemble.IsolationForest`, the lower, the more
        abnormal: the values are the anomaly scores :math:`2^{-E[h(x)] / c(\psi)}` negated,
        in :math:`[-1, 0)`.

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
        for tree in self._trees:  # summed tree by tree, as eif does
            total += tree.path_lengths(X)
        mean_path = total / len(self._trees)
        return -(2.0 ** (-mean_path / _c_factor(self.max_samples_)))


def _check_int(value, name, minimum, kind="an int"):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be {kind} >= {minimum}, got {value!r}.")
    return int(value)
