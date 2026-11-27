import numpy as np
import pytest
import scipy.sparse
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state

from pypu.models import PUExtraTrees
from pypu.models.extra_trees import _LEAF, _MAX_INT, build_tree


class OriginalPUExtraTree:
    """Reference: one tree of PU Extra Trees, as PU-Bench runs it.

    ``PUExtraTree`` from train/puet/tree.py of https://github.com/XiXiphus/PU-Bench (MIT
    license), which is ``tree.py`` of https://github.com/jonathanwilton/PUExtraTrees (MIT
    license, commit 5cb15d7) with the leaf estimates ``vstar`` and ``predict_proba`` added.
    The logic is copied unchanged for the uPU and nnPU risks; the supervised (PN) branches,
    the methods not used here, comments, docstrings and formatting differ. It draws from
    NumPy's global generator.
    """

    def __init__(
        self,
        risk_estimator="nnPU",
        loss="quadratic",
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        max_candidates=1,
    ):
        self.risk_estimator = risk_estimator
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_candidates = max_candidates
        self.is_trained = False
        self.leaf_count = 0
        self.nodes = {
            (0, 0): {
                "data": None,
                "j": None,
                "xi": None,
                "g": None,
                "vstar": None,
                "is_leaf": None,
                "risk_reduction": None,
            }
        }

    def create_successor(self, node, side):
        row, column = node
        empty = {
            "data": None,
            "j": None,
            "xi": None,
            "g": None,
            "vstar": None,
            "is_leaf": None,
            "loss": None,
            "risk_reduction": None,
        }
        if side in ["T", "L"]:
            self.nodes[(row + 1, 2 * column)] = dict(empty)
        elif side in ["F", "R"]:
            self.nodes[(row + 1, 2 * column + 1)] = dict(empty)

    def get_parent(self, node, return_truth_val):
        parent = (node[0] - 1, node[1] // 2)
        if return_truth_val:
            if node[1] % 2 == 0:
                return parent, True
            else:
                return parent, False
        else:
            return parent

    def fit(self, pi, P=None, U=None):
        X = np.concatenate((P, U), axis=0)
        y = np.concatenate((np.ones(len(P)), np.zeros(len(U))))
        y = y.astype(np.int8).flatten()
        n, self.d = X.shape
        n_p = (y == 1).sum()
        n_u = (y == 0).sum()
        self.pi = pi

        mf = self.max_features
        if isinstance(mf, str):
            mf_lower = mf.lower()
            if mf_lower == "sqrt":
                self.max_features = int(np.ceil(np.sqrt(self.d)))
            elif mf_lower in ("all", "auto", "max"):
                self.max_features = int(self.d)
            else:
                self.max_features = int(np.ceil(np.sqrt(self.d)))
        elif isinstance(mf, (int, np.integer)):
            if mf < 1:
                self.max_features = int(max(1, np.ceil(np.sqrt(self.d))))
            elif mf > self.d:
                self.max_features = int(self.d)
            else:
                self.max_features = int(mf)
        else:
            self.max_features = int(np.ceil(np.sqrt(self.d)))

        self.nodes[(0, 0)]["data"] = scipy.sparse.coo_matrix(np.ones(n).astype(bool))

        def data_at_node(node):
            if self.nodes[node]["data"] is not None:
                return self.nodes[node]["data"].toarray()[0]
            else:
                parent_node, relationship = self.get_parent(node, True)
                ind_parent = self.nodes[parent_node]["data"].toarray()[0].copy()
                checks = (
                    X[ind_parent, self.nodes[parent_node]["j"]] <= self.nodes[parent_node]["xi"]
                ) == relationship
                ind_parent[ind_parent] = checks.flatten()
                self.nodes[node]["data"] = scipy.sparse.coo_matrix(ind_parent)
                return ind_parent

        def impurity_node(y_sigma):
            Wp = (y_sigma == 1).sum() * self.pi / n_p
            Wn = (y_sigma == 0).sum() / n_u - Wp
            if Wp + Wn == 0:  # noqa: SIM108 - kept as in the original
                vstar = float("inf")
            else:
                vstar = Wp / (Wp + Wn)

            if self.loss == "quadratic":
                if self.risk_estimator == "uPU" and vstar == float("inf"):
                    return -float("inf")
                elif self.risk_estimator == "nnPU" and vstar > 1:
                    return 0
                else:
                    return 4 * (Wp + Wn) * vstar * (1 - vstar)
            elif self.loss == "logistic":
                if self.risk_estimator == "uPU" and vstar > 1:
                    return -float("inf")
                elif self.risk_estimator in ["uPU", "nnPU", "PN"] and vstar in [0, 1]:  # noqa: SIM114 - kept as in the original
                    return 0
                elif self.risk_estimator == "nnPU" and vstar > 1:
                    return 0
                else:
                    return (Wp + Wn) * (-vstar * np.log(vstar) - (1 - vstar) * np.log(1 - vstar))

        def impurity_split(sigma, j, xi):
            mask = (X[sigma, j] <= xi).flatten()
            imT = impurity_node(y[sigma][mask])
            imF = impurity_node(y[sigma][~mask])
            return imT + imF

        def regional_prediction_function(y_sigma):
            Wp = (y_sigma == 1).sum() * self.pi / n_p
            Wn = (y_sigma == 0).sum() / n_u - Wp
            if Wp + Wn == 0:  # noqa: SIM108 - kept as in the original
                vstar = float("inf")
            else:
                vstar = Wp / (Wp + Wn)

            if vstar > 0.5:
                return 1, vstar
            elif vstar < 0.5:
                return -1, vstar
            elif vstar == 0.5:
                return 2 * np.random.binomial(1, 0.5) - 1, vstar  # noqa: NPY002
            else:
                return (2 * int(vstar > 0.5) - 1), vstar

        def construct_subtree(node, sigma):
            impurity = impurity_node(y[sigma])
            if self.risk_estimator in ["nnPU", "PN"]:
                c1 = impurity > 0
            elif self.risk_estimator == "uPU":
                if y[sigma].sum() == 0:  # noqa: SIM108 - kept as in the original
                    c1 = impurity > 0
                else:
                    c1 = impurity > -float("inf")
            if self.max_depth is None:  # noqa: SIM108 - kept as in the original
                c2 = True
            else:
                c2 = node[0] < self.max_depth
            c3 = self.min_samples_leaf < sigma.sum()
            att_ptp = np.ptp(X[sigma], axis=0)
            c4 = att_ptp.sum() > 0
            if c1 * c2 * c3 * c4 == 0:
                self.nodes[node]["is_leaf"] = True
                lab, vstar = regional_prediction_function(y[sigma])
                self.nodes[node]["g"] = lab
                self.nodes[node]["vstar"] = vstar
                self.nodes[node]["risk_reduction"] = 0
                self.leaf_count += 1
            else:
                self.nodes[node]["is_leaf"] = False
                atts = []
                for i in range(self.d):
                    if att_ptp[i] > 0:
                        atts += [i]
                attributes = np.random.choice(  # noqa: NPY002
                    atts, size=min(self.max_features, len(atts)), replace=False
                )
                candidates = []
                candidate_attributes = []
                candidate_cut_points = []
                for i in range(len(attributes)):
                    for _ in range(self.max_candidates):
                        a_, b_, c_, d_ = np.unique(X[sigma, attributes[i]])[[0, 1, -2, -1]]
                        cut_point = np.random.uniform(  # noqa: NPY002
                            a_ + 2 * (b_ - a_) / 5, c_ + 3 * (d_ - c_) / 5
                        )
                        candidates += [[attributes[i], cut_point]]
                        candidate_attributes += [attributes[i]]
                        candidate_cut_points += [cut_point]
                impurities = []
                for i in range(len(candidates)):
                    impurities += [
                        impurity_split(sigma, candidate_attributes[i], candidate_cut_points[i])
                    ]
                minimiser = np.argmin(impurities)
                best_attribute = candidate_attributes[minimiser]
                best_cut_point = candidate_cut_points[minimiser]
                self.nodes[node]["j"] = int(best_attribute)
                self.nodes[node]["xi"] = best_cut_point
                self.nodes[node]["risk_reduction"] = impurity - impurities[minimiser]
                self.create_successor(node, "T")
                self.create_successor(node, "F")
                succs = ((node[0] + 1, 2 * node[1]), (node[0] + 1, 2 * node[1] + 1))
                sigma_T = data_at_node(succs[0])
                sigma_F = data_at_node(succs[1])
                construct_subtree(succs[0], sigma_T)
                construct_subtree(succs[1], sigma_F)

        construct_subtree((0, 0), np.ones(n).astype(bool))
        self.is_trained = True
        return self

    def predict_proba(self, X):
        probas = np.zeros((len(X), 2))
        for i in range(len(X)):
            X_ = X[i]
            a, b = (0, 0)
            tnode = self.nodes[a, b]
            while not tnode["is_leaf"]:
                check = X_[tnode["j"]] <= tnode["xi"]
                if check:  # noqa: SIM108 - kept as in the original
                    b = 2 * b
                else:
                    b = 2 * b + 1
                a += 1
                tnode = self.nodes.get((a, b))
                if tnode is None:
                    parent_node = self.nodes[self.get_parent((a, b), False)]
                    tnode = {"is_leaf": True, "vstar": parent_node.get("vstar", 0.5)}
                    break
            if tnode["is_leaf"]:
                vstar = tnode["vstar"]
                if vstar == float("inf") or vstar is None:
                    vstar = 0.5
                elif vstar > 1.0:
                    vstar = 1.0
                elif vstar < 0.0:
                    vstar = 0.0
                probas[i, 1] = vstar
                probas[i, 0] = 1 - vstar
        return probas

    def feature_importances(self):
        impurities = np.zeros([self.d])
        for node in self.nodes:
            if self.nodes[node]["j"] is not None:
                impurities[self.nodes[node]["j"]] += self.nodes[node]["risk_reduction"]
        return impurities


def original_tree(X, s, prior, seed, **params):
    """Fit the reference tree as PU-Bench does, after seeding NumPy's global generator."""
    np.random.seed(seed)  # noqa: NPY002
    return OriginalPUExtraTree(**params).fit(pi=prior, P=X[s == 1], U=X[s == 0])


def original_arrays(tree):
    """Return the reference tree's nodes as arrays, in the order they were created."""
    nodes = list(tree.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    arrays = {name: [] for name in ("left", "right", "feature", "threshold", "vstar", "risk")}
    for depth, position in nodes:
        node = tree.nodes[depth, position]
        leaf = node["is_leaf"]
        arrays["left"].append(_LEAF if leaf else index[depth + 1, 2 * position])
        arrays["right"].append(_LEAF if leaf else index[depth + 1, 2 * position + 1])
        arrays["feature"].append(_LEAF if leaf else node["j"])
        arrays["threshold"].append(np.nan if leaf else node["xi"])
        arrays["vstar"].append(node["vstar"] if leaf else np.nan)
        arrays["risk"].append(node["risk_reduction"])
    return {name: np.array(values, dtype=float) for name, values in arrays.items()}


def reference_proba(tree, X):
    """The reference tree's ``predict_proba``, with 1 in leaves without unlabeled samples.

    There ``vstar`` is infinite: PU-Bench counts 0.5, PyPU 1, the leaf's vote in the
    original code.
    """
    proba = tree.predict_proba(X)[:, 1]
    leaf_vstar = np.empty(len(X))
    for i, x in enumerate(X):
        depth, position = 0, 0
        while not tree.nodes[depth, position]["is_leaf"]:
            node = tree.nodes[depth, position]
            depth, position = depth + 1, 2 * position + (x[node["j"]] > node["xi"])
        leaf_vstar[i] = tree.nodes[depth, position]["vstar"]
    no_unlabeled = leaf_vstar == np.inf
    np.testing.assert_array_equal(proba[no_unlabeled], 0.5)
    return np.where(no_unlabeled, 1.0, proba)


def assert_same_tree(tree, reference):
    expected = original_arrays(reference)
    np.testing.assert_array_equal(tree.children_left, expected["left"])
    np.testing.assert_array_equal(tree.children_right, expected["right"])
    np.testing.assert_array_equal(tree.feature, expected["feature"])
    np.testing.assert_array_equal(tree.threshold, expected["threshold"])
    np.testing.assert_array_equal(tree.vstar, expected["vstar"])
    np.testing.assert_array_equal(tree.risk_reduction, expected["risk"])


def ported_params(params):
    """``build_tree``'s arguments for the reference parameters."""
    n_features = params.pop("n_features")
    max_features = params.get("max_features", "sqrt")
    if max_features == "sqrt":
        max_features = int(np.ceil(np.sqrt(n_features)))
    elif max_features == "all":
        max_features = n_features
    return {
        "risk_estimator": params.get("risk_estimator", "nnPU"),
        "loss": params.get("loss", "quadratic"),
        "max_depth": params.get("max_depth"),
        "min_samples_leaf": params.get("min_samples_leaf", 1),
        "max_features": max_features,
        "max_candidates": params.get("max_candidates", 1),
    }


@pytest.fixture(scope="module")
def plain():
    """Continuous features, half of the positives labeled; ``(X, y, s)``."""
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, weights=[0.65], random_state=0
    )
    s = ((y == 1) & (np.random.default_rng(1).random(y.size) < 0.5)).astype(int)
    return X, y, s


@pytest.fixture(scope="module")
def ties():
    """Few distinct values with repeated rows, as many labeled as unlabeled samples.

    With the case-control prior 0.5, a leaf holding as many labeled as unlabeled samples has
    ``vstar == 0.5`` exactly, where the original breaks the tie of its vote at random.
    """
    rng = np.random.default_rng(2)
    X = rng.integers(0, 3, size=(240, 4)).astype(float)
    s = np.zeros(240, dtype=int)
    s[rng.permutation(240)[:120]] = 1
    return X, s


# Reference cases: dataset and the original tree's parameters.
TREE_CASES = {
    "nnpu_quadratic": ("plain", {}),
    "upu_quadratic": ("plain", {"risk_estimator": "uPU"}),
    "nnpu_logistic": ("plain", {"loss": "logistic"}),
    "upu_logistic": ("plain", {"risk_estimator": "uPU", "loss": "logistic"}),
    "limits": (
        "plain",
        {"max_depth": 4, "min_samples_leaf": 5, "max_features": "all", "max_candidates": 3},
    ),
    "ties": ("ties", {"max_features": 2}),
    "ties_logistic": ("ties", {"loss": "logistic", "max_candidates": 2}),
}


def case_data(case, plain, ties):
    dataset, params = TREE_CASES[case]
    if dataset == "plain":
        X, y, s = plain
        prior = (y.mean() * y.size - s.sum()) / (y.size - s.sum())
    else:
        (X, s), prior = ties, 0.5
    return X, s, prior, dict(params)


@pytest.mark.parametrize("case", list(TREE_CASES))
@pytest.mark.parametrize("seed", [0, 7, 12345])
def test_tree_reproduces_the_original_implementation(plain, ties, case, seed):
    X, s, prior, params = case_data(case, plain, ties)
    reference = original_tree(X, s, prior, seed, **params)
    tree = build_tree(X, s == 1, prior, seed, **ported_params({**params, "n_features": X.shape[1]}))
    assert_same_tree(tree, reference)
    np.testing.assert_array_equal(tree.value[tree.apply(X)], reference_proba(reference, X))


def test_ties_draw_like_the_original(ties):
    """The tie-breaking draws happen in the ties case, so the test above covers them."""
    X, s = ties
    tree = build_tree(X, s == 1, 0.5, 0, **ported_params({"n_features": 4, "max_features": 2}))
    assert np.count_nonzero(tree.vstar == 0.5) > 0


@pytest.mark.parametrize(
    "params",
    [
        {"random_state": 0},
        {"random_state": 3, "risk_estimator": "uPU", "loss": "logistic", "max_features": None},
    ],
)
def test_forest_reproduces_the_original_implementation(plain, params):
    X, y, s = plain
    prior = float(y.mean())
    model = PUExtraTrees(prior=prior, n_estimators=5, **params).fit(X, s)
    random_state = check_random_state(params["random_state"])
    seeds = [random_state.randint(_MAX_INT) for _ in range(5)]
    reference_params = {
        key: value for key, value in params.items() if key in ("risk_estimator", "loss")
    }
    if params.get("max_features", "sqrt") is None:
        reference_params["max_features"] = "all"
    trees = [
        original_tree(X, s, model.unlabeled_prior_, seed, **reference_params) for seed in seeds
    ]
    for tree, reference in zip(model.trees_, trees, strict=True):
        assert_same_tree(tree, reference)
    # The mean of PU-Bench's PUExtraTrees.predict_proba, as reference_proba changes it, and
    # the original feature_importances.
    expected_proba = np.mean([reference_proba(tree, X) for tree in trees], axis=0)
    np.testing.assert_array_equal(model.predict_proba(X)[:, 1], expected_proba)
    expected_importances = np.zeros(X.shape[1])
    for tree in trees:
        expected_importances += tree.feature_importances() / len(trees)
    np.testing.assert_array_equal(model.feature_importances_, expected_importances)


def test_unlabeled_prior(plain):
    X, y, s = plain
    n, n_labeled = y.size, s.sum()
    single = PUExtraTrees(prior=0.4, n_estimators=2, random_state=0).fit(X, s)
    assert single.unlabeled_prior_ == pytest.approx((0.4 * n - n_labeled) / (n - n_labeled))
    case_control = PUExtraTrees(
        prior=0.4, scenario="case_control", n_estimators=2, random_state=0
    ).fit(X, s)
    assert case_control.unlabeled_prior_ == 0.4


def test_single_scenario_needs_a_prior_above_the_labeled_fraction(plain):
    X, _, s = plain
    with pytest.raises(ValueError, match="above the fraction of labeled samples"):
        PUExtraTrees(prior=0.9 * s.mean()).fit(X, s)
    # Allowed when the labeled positives were drawn apart from the unlabeled samples.
    PUExtraTrees(prior=0.9 * s.mean(), scenario="case_control", n_estimators=2).fit(X, s)


def test_leaf_probabilities(plain):
    """Clipped at 1, and 1 in leaves without unlabeled samples, where PU-Bench gives 0.5."""
    X, y, s = plain
    model = PUExtraTrees(prior=y.mean(), n_estimators=5, random_state=0).fit(X, s)
    kinds = np.zeros(3, dtype=int)
    for tree in model.trees_:
        leaves = tree.feature == _LEAF
        np.testing.assert_array_equal(np.isnan(tree.value), ~leaves)
        no_unlabeled = leaves & (tree.vstar == np.inf)
        np.testing.assert_array_equal(tree.value[no_unlabeled], 1.0)
        over = leaves & (tree.vstar > 1) & ~no_unlabeled
        np.testing.assert_array_equal(tree.value[over], 1.0)
        under = leaves & (tree.vstar <= 1)
        np.testing.assert_array_equal(tree.value[under], tree.vstar[under])
        kinds += [no_unlabeled.sum(), over.sum(), under.sum()]
    assert np.all(kinds > 0)
    expected = np.mean([tree.value[tree.apply(X)] for tree in model.trees_], axis=0)
    np.testing.assert_array_equal(model.predict_proba(X)[:, 1], expected)


def test_upu_leaves_hold_only_labeled_or_only_unlabeled_samples(plain):
    """So with PU-Bench's 0.5 for the labeled-only leaves, uPU would predict no positive."""
    X, y, s = plain
    model = PUExtraTrees(prior=y.mean(), n_estimators=5, risk_estimator="uPU", random_state=0)
    model.fit(X, s)
    for tree in model.trees_:
        leaf_vstar = tree.vstar[tree.feature == _LEAF]
        assert set(np.unique(leaf_vstar)) == {0.0, np.inf}
    assert model.predict(X).sum() > 0


def test_upu_importances_can_be_infinite(plain):
    X, y, s = plain
    nnpu = PUExtraTrees(prior=y.mean(), n_estimators=5, random_state=0).fit(X, s)
    assert np.all(np.isfinite(nnpu.feature_importances_))
    upu = PUExtraTrees(prior=y.mean(), n_estimators=5, risk_estimator="uPU", random_state=0)
    assert np.any(np.isinf(upu.fit(X, s).feature_importances_))


def test_split_on_adjacent_floats_can_leave_an_empty_child():
    """The split point can round to the highest value: the empty child becomes a leaf."""
    low = 1.0
    X = np.array([[low], [np.nextafter(low, 2.0)]] * 3)
    s = np.array([1, 0, 0, 1, 0, 0])
    empty = 0
    for seed in range(20):
        tree = build_tree(X, s == 1, 0.5, seed, **ported_params({"n_features": 1}))
        empty += np.count_nonzero((tree.feature == _LEAF) & (tree.vstar == np.inf))
        assert tree.value[tree.apply(X)].shape == (6,)
    assert empty > 0


def test_max_depth_and_max_features(plain):
    X, y, s = plain
    model = PUExtraTrees(prior=y.mean(), n_estimators=3, max_depth=3, random_state=0).fit(X, s)
    assert model.max_features_ == 3  # ceil(sqrt(6)), as the original
    for tree in model.trees_:
        depth = np.zeros(tree.node_count, dtype=int)
        for node in range(tree.node_count):
            if tree.feature[node] != _LEAF:
                depth[[tree.children_left[node], tree.children_right[node]]] = depth[node] + 1
        assert depth.max() <= 3


def test_n_jobs_does_not_change_the_result(plain):
    X, y, s = plain
    serial = PUExtraTrees(prior=y.mean(), n_estimators=6, random_state=0).fit(X, s)
    parallel = PUExtraTrees(prior=y.mean(), n_estimators=6, n_jobs=2, random_state=0).fit(X, s)
    np.testing.assert_array_equal(serial.predict_proba(X), parallel.predict_proba(X))


def test_ranks_well():
    X, y = make_classification(
        n_samples=1200, n_features=8, n_informative=5, weights=[0.75], random_state=0
    )
    s = ((y == 1) & (np.random.default_rng(0).random(y.size) < 0.4)).astype(int)
    model = PUExtraTrees(prior=y.mean(), n_estimators=30, random_state=0).fit(X[:900], s[:900])
    assert roc_auc_score(y[900:], model.predict_proba(X[900:])[:, 1]) > 0.9


def test_requires_the_prior(plain):
    X, _, s = plain
    with pytest.raises(ValueError, match="needs the class prior"):
        PUExtraTrees().fit(X, s)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"scenario": "two_sample"}, "scenario must be one of"),
        ({"risk_estimator": "PN"}, "risk_estimator must be"),
        ({"loss": "hinge"}, "loss must be"),
        ({"n_estimators": 0}, "n_estimators must be"),
        ({"max_depth": 0}, "max_depth must be None or an int"),
        ({"min_samples_leaf": 0}, "min_samples_leaf must be"),
        ({"min_samples_leaf": 1.5}, "min_samples_leaf must be"),
        ({"max_features": "log2"}, "max_features must be"),
        ({"max_features": 0.5}, "max_features must be"),
        ({"max_features": 0}, "between 1 and the number of features"),
        ({"max_features": 7}, "between 1 and the number of features"),
        ({"max_candidates": 0}, "max_candidates must be"),
    ],
)
def test_rejects_invalid_parameters(plain, params, match):
    X, y, s = plain
    with pytest.raises(ValueError, match=match):
        PUExtraTrees(prior=y.mean(), **params).fit(X, s)
