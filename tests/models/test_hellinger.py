from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from pypu.models import PUHellingerForest, PUHellingerTree
from pypu.models._hellinger_tree import TREE_LEAF, _hellinger_children, _RandR

REFERENCE = Path(__file__).with_name("data") / "hellinger_reference.npz"
TREE_ARRAYS = (
    "children_left",
    "children_right",
    "feature",
    "threshold",
    "value",
    "n_node_samples",
)

# Reference cases of data/make_hellinger_reference.py -> (dataset, matching parameters).
TREE_CASES = {
    "tree_experiments": ("plain", {"random_state": 0, "max_depth": 5}),
    "tree_full": ("plain", {"random_state": 1}),
    "tree_sqrt_leaf": (
        "plain",
        {"random_state": 2, "max_features": "sqrt", "min_samples_leaf": 5, "min_samples_split": 12},
    ),
    "tree_min_decrease": (
        "ties",
        {"random_state": 7, "max_features": 3, "min_impurity_decrease": 0.28},
    ),
    "tree_ties": ("ties", {"random_state": 4, "max_features": 3, "max_depth": 8}),
    "tree_ties_all": ("ties", {"random_state": 5}),
}
FOREST_CASES = {
    "forest_experiments": ("plain", {"random_state": 0}),
    "forest_small": (
        "ties",
        {
            "random_state": 6,
            "n_estimators": 10,
            "max_depth": 4,
            "max_features": None,
            "max_samples": 150,
        },
    ),
}


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as data:
        return {key: data[key] for key in data.files}


def split(reference, dataset):
    X, s = reference[f"{dataset}_X"], reference[f"{dataset}_s"]
    return X[:400], s[:400], X[400:], float(reference[f"{dataset}_prior"])


@pytest.mark.parametrize("case", list(TREE_CASES))
def test_tree_reproduces_the_original_implementation(reference, case):
    dataset, params = TREE_CASES[case]
    X_train, s_train, X_test, prior = split(reference, dataset)
    model = PUHellingerTree(prior=prior, **params).fit(X_train, s_train)
    for name in TREE_ARRAYS:
        np.testing.assert_array_equal(getattr(model.tree_, name), reference[f"{case}_{name}"])
    np.testing.assert_array_equal(model.predict_proba(X_test)[:, 1], reference[f"{case}_proba"])
    np.testing.assert_allclose(
        model.predict_pu_proba(X_test)[:, 1], reference[f"{case}_pu_proba"], rtol=0, atol=1e-15
    )


@pytest.mark.parametrize("case", list(FOREST_CASES))
def test_forest_reproduces_the_original_implementation(reference, case):
    dataset, params = FOREST_CASES[case]
    X_train, s_train, X_test, prior = split(reference, dataset)
    model = PUHellingerForest(prior=prior, **params).fit(X_train, s_train)
    node_counts = [tree.node_count for tree in model.trees_]
    np.testing.assert_array_equal(node_counts, reference[f"{case}_node_counts"])
    for name in TREE_ARRAYS:
        np.testing.assert_array_equal(
            getattr(model.trees_[0], name), reference[f"{case}_first_{name}"]
        )
    np.testing.assert_allclose(
        model.predict_proba(X_test)[:, 1], reference[f"{case}_proba"], rtol=0, atol=1e-12
    )


class TestCriterion:
    def test_hand_computed_example(self):
        # Left: 3 unlabeled, 1 labeled; right: 5 unlabeled, 1 labeled; 1 / c = 2.
        left, right = np.array([[3.0], [1.0]]), np.array([[5.0], [1.0]])
        impurity_left, impurity_right = _hellinger_children(left, right, 2.0)
        # Estimated positives 2 and 2, negatives 2 and 4.
        np.testing.assert_allclose(impurity_left, (np.sqrt(2 / 6) - np.sqrt(2 / 4)) ** 2)
        np.testing.assert_allclose(impurity_right, (np.sqrt(4 / 6) - np.sqrt(2 / 4)) ** 2)

    def test_positives_are_capped_at_the_child_size(self):
        # 1 / c = 3 would make 6 positives out of 4 samples on the left.
        left, right = np.array([[2.0], [2.0]]), np.array([[6.0], [0.0]])
        impurity_left, impurity_right = _hellinger_children(left, right, 3.0)
        # Left: 4 positives, 0 negatives; right: 0 positives, 6 negatives.
        np.testing.assert_allclose(impurity_left + impurity_right, 2.0)

    def test_nodes_without_labeled_samples_score_minus_infinity(self):
        left, right = np.array([[3.0], [0.0]]), np.array([[5.0], [0.0]])
        impurity_left, impurity_right = _hellinger_children(left, right, 2.0)
        assert impurity_left[0] == impurity_right[0] == -np.inf


def test_xorshift_generator():
    first, second = _RandR(12345), _RandR(12345)
    draws = [first.rand_int(0, 7) for _ in range(100)]
    assert draws == [second.rand_int(0, 7) for _ in range(100)]
    assert set(draws) <= set(range(7)) and len(set(draws)) > 1
    assert _RandR(0).rand_int(0, 1000) == _RandR(1).rand_int(0, 1000)  # 0 counts as 1


@pytest.fixture(scope="module")
def pu_classification():
    X, y = make_classification(
        n_samples=1200, n_features=8, n_informative=5, weights=[0.75], random_state=0
    )
    s = ((y == 1) & (np.random.default_rng(0).random(y.size) < 0.4)).astype(int)
    return X, y, s


def test_nodes_without_labeled_samples_are_leaves(pu_classification):
    X, y, s = pu_classification
    tree = PUHellingerTree(prior=y.mean(), random_state=0).fit(X, s).tree_
    internal = tree.children_left != TREE_LEAF
    assert np.all(tree.value[internal, 1] > 0)


def test_ranks_well(pu_classification):
    X, y, s = pu_classification
    tree = PUHellingerTree(prior=y.mean(), max_depth=6, random_state=0).fit(X, s)
    forest = PUHellingerForest(prior=y.mean(), n_estimators=30, random_state=0).fit(X, s)
    assert roc_auc_score(y, tree.predict_proba(X)[:, 1]) > 0.8
    assert roc_auc_score(y, forest.predict_proba(X)[:, 1]) > 0.9


@pytest.mark.parametrize(
    ("max_samples", "n_draws"), [(None, "n_unlabeled"), (100, 100), (0.5, "half")]
)
def test_forest_bootstrap_samples(pu_classification, max_samples, n_draws):
    X, y, s = pu_classification
    n_unlabeled = int(np.sum(s == 0))
    expected = {"n_unlabeled": n_unlabeled, "half": round(n_unlabeled / 2)}.get(n_draws, n_draws)
    forest = PUHellingerForest(
        prior=y.mean(), n_estimators=3, max_depth=2, max_samples=max_samples, random_state=0
    ).fit(X, s)
    for tree in forest.trees_:
        # Root: the weighted unlabeled draws and every labeled positive once.
        np.testing.assert_array_equal(tree.value[0], [expected, s.sum()])


def test_forest_n_jobs_does_not_change_the_result(pu_classification):
    X, y, s = pu_classification
    serial = PUHellingerForest(prior=y.mean(), n_estimators=6, random_state=0).fit(X, s)
    parallel = PUHellingerForest(prior=y.mean(), n_estimators=6, random_state=0, n_jobs=2)
    parallel.fit(X, s)
    np.testing.assert_array_equal(serial.predict_proba(X), parallel.predict_proba(X))


def test_pu_proba_scales_the_labeled_odds(pu_classification):
    X, y, s = pu_classification
    prior = y.mean()
    tree = PUHellingerTree(prior=prior, max_depth=3, random_state=0).fit(X, s)
    root = tree.tree_.value[0]
    counts = tree.tree_.value[tree.tree_.apply(X.astype(np.float32))]
    with np.errstate(divide="ignore"):
        expected = np.minimum(1.0, prior * (counts[:, 1] / counts[:, 0]) / (root[1] / root[0]))
    np.testing.assert_allclose(tree.predict_pu_proba(X)[:, 1], expected)


@pytest.mark.parametrize("model", [PUHellingerTree, PUHellingerForest])
def test_requires_the_prior(pu_data, model):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="needs the class prior"):
        model().fit(X, s)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"max_depth": 0}, "max_depth must be None or an int >= 1"),
        ({"min_samples_split": 1}, "min_samples_split must be an int >= 2"),
        ({"min_samples_split": 1.5}, r"min_samples_split must be .* in \(0, 1.0\]"),
        ({"min_samples_leaf": 0}, "min_samples_leaf must be an int >= 1"),
        ({"min_samples_leaf": 0.7}, r"min_samples_leaf must be .* in \(0, 0.5\]"),
        ({"max_features": "auto"}, "max_features must be 'sqrt', 'log2'"),
        ({"max_features": 9}, "between 1 and the number of features"),
        ({"min_impurity_decrease": -1.0}, "min_impurity_decrease must be"),
    ],
)
def test_tree_rejects_invalid_parameters(pu_data, params, match):
    X, y, s = pu_data
    with pytest.raises(ValueError, match=match):
        PUHellingerTree(prior=y.mean(), **params).fit(X, s)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_estimators": 0}, "n_estimators must be a positive int"),
        ({"max_samples": 0}, "max_samples must be a positive int"),
        ({"max_samples": 1.5}, r"max_samples must be in \(0, 1\]"),
        ({"max_samples": "all"}, "max_samples must be an int, a float or None"),
    ],
)
def test_forest_rejects_invalid_parameters(pu_data, params, match):
    X, y, s = pu_data
    with pytest.raises(ValueError, match=match):
        PUHellingerForest(prior=y.mean(), **params).fit(X, s)
