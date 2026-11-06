import math

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_consistent_length

from pypu.datasets import (
    label_sar_cluster,
    label_sar_confidence,
    label_sar_knn,
    label_sar_propensity,
    label_scar,
)
from pypu.datasets._common import weighted_choice


def legacy_create_pu_data(
    X, y, ratio=None, label_assumption="SCAR", n_neighbors=None, random_state=123
):
    """Reference for ``legacy=True``: ``create_pu_data`` from iFPNU/data_loaders/pu_data.py.

    The logic is copied unchanged; only comments and formatting differ.
    """
    y_ = y.copy()
    check_consistent_length(X, y_)
    n_pos = np.sum(y_ == 1)
    random_state = check_random_state(random_state)

    if ratio is None:
        return y_
    if n_neighbors is None:
        n_neighbors = math.floor(math.sqrt(len(y_)))
    if label_assumption == "SCAR":
        p_ = None
    elif label_assumption == "SAR":
        dist_matrix = pairwise_distances(X, n_jobs=-1)
        ix_pos = np.where(y == 1)[0]
        ix_neg = np.where(y == 0)[0]
        pos_dist_mt = dist_matrix[ix_pos][:, ix_neg]
        k_mean_dist = np.mean(pos_dist_mt[:, :n_neighbors], axis=1)
        sample_weight = np.array(
            [k_mean_dist[i] / np.sum(k_mean_dist) for i, _ in enumerate(k_mean_dist)]
        )
        p_ = sample_weight.reshape(
            -1,
        )
    else:
        raise ValueError("Labeling mechanism not valid; Use SCAR or SAR.")

    if label_assumption == "SAR":
        size_ = min(int(np.ceil(ratio * n_pos)), len(np.nonzero(p_)[0]))
    else:
        size_ = int(np.ceil(ratio * n_pos))
    indices = random_state.choice(range(n_pos), size_, replace=False, p=p_)
    pos_y = y_[y_ == 1].copy()
    pos_y[indices] = 0
    y_[y_ == 1] = pos_y
    return y_


@pytest.fixture
def data():
    return make_classification(
        n_samples=300, n_features=5, n_informative=3, weights=[0.6], random_state=0
    )


@pytest.fixture
def legacy_data():
    """Exactly 100 positives, so that ceil(0.3 * 100) and ceil((1 - 0.7) * 100) differ."""
    rng = np.random.default_rng(0)
    y = np.array([1] * 100 + [0] * 150)
    rng.shuffle(y)
    X = rng.normal(size=(y.size, 4)) + y[:, None]
    return X, y


def mean_distance_to_negatives(X, y, n_neighbors):
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X[y == 0])
    return nn.kneighbors(X[y == 1])[0].mean(axis=1)


# Every mechanism with the same call signature.
MECHANISMS = {
    "scar": lambda X, y, **kw: label_scar(y, **kw),
    "sar_knn_near": lambda X, y, **kw: label_sar_knn(X, y, bias="near", **kw),
    "sar_knn_far": lambda X, y, **kw: label_sar_knn(X, y, bias="far", **kw),
    "sar_propensity": label_sar_propensity,
    "sar_cluster": label_sar_cluster,
    "sar_confidence_easy": lambda X, y, **kw: label_sar_confidence(X, y, bias="easy", **kw),
    "sar_confidence_hard": lambda X, y, **kw: label_sar_confidence(X, y, bias="hard", **kw),
}


@pytest.fixture(params=list(MECHANISMS))
def mechanism(request):
    return request.param, MECHANISMS[request.param]


class TestAllMechanisms:
    @pytest.mark.parametrize("label_frequency", [0.1, 0.5, 0.9])
    def test_labels_a_share_of_the_positives(self, data, mechanism, label_frequency):
        X, y = data
        name, label = mechanism
        s = label(X, y, label_frequency=label_frequency, random_state=0)
        assert s.dtype == np.int64
        assert s.shape == y.shape
        assert set(np.unique(s)) == {0, 1}
        assert np.all(s <= y), "a negative was labeled"
        n_positive = int(y.sum())
        if name == "sar_propensity":
            # Independent draws: the count is binomial around label_frequency * n_positive.
            sd = math.sqrt(n_positive * label_frequency * (1 - label_frequency))
            assert abs(s.sum() - label_frequency * n_positive) < 4 * sd
        else:
            assert s.sum() == math.floor(label_frequency * n_positive + 0.5)

    def test_same_seed_same_labels(self, data, mechanism):
        X, y = data
        _, label = mechanism
        first = label(X, y, random_state=1)
        np.testing.assert_array_equal(first, label(X, y, random_state=1))
        assert not np.array_equal(first, label(X, y, random_state=2))

    def test_accepts_dataframes_and_lists(self, data, mechanism):
        X, y = data
        _, label = mechanism
        expected = label(X, y, random_state=0)
        from_pandas = label(pd.DataFrame(X), pd.Series(y), random_state=0)
        from_lists = label(X.tolist(), y.tolist(), random_state=0)
        np.testing.assert_array_equal(from_pandas, expected)
        np.testing.assert_array_equal(from_lists, expected)

    @pytest.mark.parametrize(
        ("y_values", "match"),
        [
            ([0, 1, 2], "true binary labels"),
            ([-1, 1], "true binary labels"),
            (["a", "b"], "true binary labels"),
            ([1], "no negative"),
            ([0], "no positive"),
        ],
    )
    def test_rejects_invalid_true_labels(self, data, mechanism, y_values, match):
        X, _ = data
        _, label = mechanism
        y = np.resize(np.array(y_values), X.shape[0])
        with pytest.raises(ValueError, match=match):
            label(X, y)

    @pytest.mark.parametrize("value", [0, 0.0, -0.1, 1.5, float("nan")])
    def test_rejects_label_frequency_out_of_range(self, data, mechanism, value):
        X, y = data
        _, label = mechanism
        with pytest.raises(ValueError, match="label_frequency must be in"):
            label(X, y, label_frequency=value)

    @pytest.mark.parametrize("value", ["0.5", True, None])
    def test_rejects_label_frequency_of_wrong_type(self, data, mechanism, value):
        X, y = data
        _, label = mechanism
        with pytest.raises(TypeError, match="label_frequency must be a float"):
            label(X, y, label_frequency=value)

    def test_rejects_too_small_label_frequency(self, data, mechanism):
        X, y = data
        name, label = mechanism
        match = "No positive was labeled" if name == "sar_propensity" else "labels none of"
        with pytest.raises(ValueError, match=match):
            label(X, y, label_frequency=1e-6, random_state=0)


def test_rejects_empty_true_labels():
    with pytest.raises(ValueError, match="y is empty"):
        label_scar([])


@pytest.mark.parametrize(
    "label",
    [label_sar_knn, label_sar_propensity, label_sar_cluster, label_sar_confidence],
)
def test_rejects_inconsistent_lengths(data, label):
    X, y = data
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        label(X, y[:-1])


@pytest.mark.parametrize(
    "label",
    [
        MECHANISMS["scar"],
        MECHANISMS["sar_knn_near"],
        MECHANISMS["sar_knn_far"],
        label_sar_cluster,
        label_sar_confidence,
    ],
)
def test_label_frequency_one_labels_every_positive(data, label):
    X, y = data
    np.testing.assert_array_equal(label(X, y, label_frequency=1.0, random_state=0), y)


class TestLabelScar:
    def test_every_positive_is_equally_likely(self):
        y = np.array([1] * 20 + [0] * 20)
        n_draws = 2000
        counts = sum(label_scar(y, label_frequency=0.25, random_state=i) for i in range(n_draws))
        frequency = counts[:20] / n_draws
        tolerance = 4 * math.sqrt(0.25 * 0.75 / n_draws)
        np.testing.assert_allclose(frequency, 0.25, atol=tolerance)

    @pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("seed", [0, 1, 123])
    def test_legacy_reproduces_create_pu_data(self, legacy_data, ratio, seed):
        X, y = legacy_data
        expected = legacy_create_pu_data(X, y, ratio, "SCAR", random_state=seed)
        s = label_scar(y, label_frequency=1 - ratio, legacy=True, random_state=seed)
        np.testing.assert_array_equal(s, expected)

    def test_legacy_count_follows_the_old_rounding(self, legacy_data):
        _, y = legacy_data
        # The old code hid ceil(0.3 * 100) = 30 positives; 1 - 0.7 must not make it 31.
        assert label_scar(y, label_frequency=0.7, legacy=True, random_state=0).sum() == 70


class TestLabelSarKnn:
    @pytest.mark.parametrize("bias", ["near", "far"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_bias_direction(self, data, bias, seed):
        X, y = data
        distance = mean_distance_to_negatives(X, y, n_neighbors=5)
        s = label_sar_knn(X, y, label_frequency=0.5, n_neighbors=5, bias=bias, random_state=seed)
        labeled = s[y == 1] == 1
        labeled_mean, unlabeled_mean = distance[labeled].mean(), distance[~labeled].mean()
        if bias == "near":
            assert labeled_mean < unlabeled_mean
        else:
            assert labeled_mean > unlabeled_mean

    def test_default_n_neighbors_is_sqrt_n_samples(self, data):
        X, y = data  # 300 samples: floor(sqrt(300)) = 17
        np.testing.assert_array_equal(
            label_sar_knn(X, y, random_state=0),
            label_sar_knn(X, y, n_neighbors=17, random_state=0),
        )

    def test_default_n_neighbors_is_capped_at_the_number_of_negatives(self):
        X, y = make_classification(n_samples=100, weights=[0.05], flip_y=0, random_state=0)
        n_negatives = int((y == 0).sum())
        assert n_negatives < 10
        np.testing.assert_array_equal(
            label_sar_knn(X, y, random_state=0),
            label_sar_knn(X, y, n_neighbors=n_negatives, random_state=0),
        )

    @pytest.mark.parametrize("bias", ["near", "far"])
    def test_positives_on_top_of_negatives_are_drawn_last(self, bias):
        # Three positives coincide with negatives, so their distance is 0.
        X = np.array([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [5.0], [6.0], [7.0], [8.0]])
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        on_negatives = np.array([3, 4, 5])
        for seed in range(20):
            s = label_sar_knn(
                X, y, label_frequency=3 / 7, n_neighbors=1, bias=bias, random_state=seed
            )
            if bias == "far":  # drawn to be labeled: never labeled
                assert s[on_negatives].sum() == 0
            else:  # drawn to be hidden: never hidden
                assert s[on_negatives].sum() == 3

    @pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("seed", [0, 1, 123])
    def test_legacy_reproduces_create_pu_data(self, legacy_data, ratio, seed):
        X, y = legacy_data
        expected = legacy_create_pu_data(X, y, ratio, "SAR", random_state=seed)
        s = label_sar_knn(X, y, label_frequency=1 - ratio, legacy=True, random_state=seed)
        np.testing.assert_array_equal(s, expected)

    def test_legacy_with_explicit_n_neighbors(self, legacy_data):
        X, y = legacy_data
        expected = legacy_create_pu_data(X, y, 0.4, "SAR", n_neighbors=5, random_state=0)
        s = label_sar_knn(X, y, label_frequency=0.6, n_neighbors=5, legacy=True, random_state=0)
        np.testing.assert_array_equal(s, expected)

    def test_legacy_only_has_near_bias(self, data):
        X, y = data
        with pytest.raises(ValueError, match="only had bias='near'"):
            label_sar_knn(X, y, bias="far", legacy=True)

    @pytest.mark.parametrize("n_neighbors", [0, -1, 2.5, True])
    def test_rejects_invalid_n_neighbors(self, data, n_neighbors):
        X, y = data
        with pytest.raises(ValueError, match="n_neighbors must be an integer"):
            label_sar_knn(X, y, n_neighbors=n_neighbors)

    def test_rejects_more_neighbors_than_negatives(self, data):
        X, y = data
        with pytest.raises(ValueError, match="larger than the number of negatives"):
            label_sar_knn(X, y, n_neighbors=int((y == 0).sum()) + 1)

    def test_rejects_unknown_bias(self, data):
        X, y = data
        with pytest.raises(ValueError, match="bias must be one of"):
            label_sar_knn(X, y, bias="middle")


class TestLabelSarPropensity:
    @pytest.mark.parametrize("label_frequency", [0.05, 0.3, 0.8])
    @pytest.mark.parametrize("strength", [0.0, 1.0, 5.0])
    def test_mean_propensity_of_positives_is_the_label_frequency(
        self, data, label_frequency, strength
    ):
        X, y = data
        _, e = label_sar_propensity(
            X,
            y,
            label_frequency=label_frequency,
            strength=strength,
            return_propensity=True,
            random_state=0,
        )
        assert e.shape == y.shape
        assert np.all((e > 0) & (e < 1))
        assert e[y == 1].mean() == pytest.approx(label_frequency, abs=1e-9)

    def test_zero_strength_gives_a_constant_propensity(self, data):
        X, y = data
        _, e = label_sar_propensity(X, y, label_frequency=0.3, strength=0, return_propensity=True)
        np.testing.assert_allclose(e, 0.3)

    @pytest.mark.parametrize("sign", [1, -1])
    def test_propensity_follows_the_sign_of_coef(self, data, sign):
        X, y = data
        s, e = label_sar_propensity(
            X, y, features=[0], coef=[sign], strength=2.0, return_propensity=True, random_state=0
        )
        positives = y == 1
        assert np.sign(np.corrcoef(X[positives, 0], e[positives])[0, 1]) == sign
        labeled = s[positives] == 1
        x0 = X[positives, 0]
        assert np.sign(x0[labeled].mean() - x0[~labeled].mean()) == sign

    def test_propensity_depends_only_on_the_selected_features(self, data):
        X, y = data
        X_changed = X.copy()
        X_changed[:, 2:] = np.random.default_rng(0).normal(size=(X.shape[0], X.shape[1] - 2))
        _, e = label_sar_propensity(X, y, features=[0, 1], return_propensity=True)
        _, e_changed = label_sar_propensity(X_changed, y, features=[0, 1], return_propensity=True)
        np.testing.assert_allclose(e, e_changed)

    def test_realised_label_frequency_is_close_to_the_target(self):
        X, y = make_classification(n_samples=5000, random_state=0)
        s = label_sar_propensity(X, y, label_frequency=0.3, strength=2.0, random_state=0)
        assert s[y == 1].mean() == pytest.approx(0.3, abs=0.03)

    def test_feature_constant_among_positives(self, data):
        X, y = data
        X = X.copy()
        X[y == 1, 0] = 1.0
        _, e = label_sar_propensity(X, y, features=[0], return_propensity=True)
        np.testing.assert_allclose(e[y == 1], 0.5)

    def test_rejects_label_frequency_one(self, data):
        X, y = data
        with pytest.raises(ValueError, match=r"label_frequency must be in \(0, 1\)"):
            label_sar_propensity(X, y, label_frequency=1.0)

    @pytest.mark.parametrize("features", [[], [0.5], [5], [-1], [[0, 1]]])
    def test_rejects_invalid_features(self, data, features):
        X, y = data
        with pytest.raises(ValueError, match="features must be"):
            label_sar_propensity(X, y, features=features)

    def test_rejects_coef_of_wrong_length(self, data):
        X, y = data
        with pytest.raises(ValueError, match="one weight per selected feature"):
            label_sar_propensity(X, y, features=[0, 1], coef=[1.0])

    @pytest.mark.parametrize("strength", [float("nan"), float("inf"), "1", True])
    def test_rejects_invalid_strength(self, data, strength):
        X, y = data
        with pytest.raises(ValueError, match="strength must be"):
            label_sar_propensity(X, y, strength=strength)


class TestLabelSarCluster:
    @pytest.fixture
    def blobs(self):
        # Positives in four well separated, equal blobs; negatives in a fifth.
        centers = [[0, 0], [20, 0], [0, 20], [20, 20], [-20, -20]]
        X, blob = make_blobs(
            n_samples=[40, 40, 40, 40, 100], centers=centers, cluster_std=1.0, random_state=0
        )
        return X, (blob < 4).astype(int), blob

    @pytest.mark.parametrize(("label_frequency", "n_blobs"), [(0.25, 1), (0.5, 2), (0.6, 3)])
    def test_labels_come_from_the_fewest_clusters_needed(self, blobs, label_frequency, n_blobs):
        X, y, blob = blobs
        for seed in range(5):
            s = label_sar_cluster(
                X, y, label_frequency=label_frequency, n_clusters=4, random_state=seed
            )
            assert np.unique(blob[s == 1]).size == n_blobs

    def test_one_cluster_labels_uniformly(self, blobs):
        X, y, blob = blobs
        s = label_sar_cluster(X, y, label_frequency=0.5, n_clusters=1, random_state=0)
        assert np.unique(blob[s == 1]).size == 4

    @pytest.mark.parametrize("n_clusters", [0, 2.0, None])
    def test_rejects_invalid_n_clusters(self, data, n_clusters):
        X, y = data
        with pytest.raises(ValueError, match="n_clusters must be an integer"):
            label_sar_cluster(X, y, n_clusters=n_clusters)

    def test_rejects_more_clusters_than_positives(self, data):
        X, y = data
        with pytest.raises(ValueError, match="larger than the number of positives"):
            label_sar_cluster(X, y, n_clusters=int(y.sum()) + 1)


class TestLabelSarConfidence:
    @pytest.mark.parametrize("bias", ["easy", "hard"])
    def test_bias_direction(self, data, bias):
        X, y = data
        proba = cross_val_predict(LogisticRegression(), X, y, cv=5, method="predict_proba")
        proba = proba[y == 1, 1]
        s = label_sar_confidence(X, y, label_frequency=0.3, bias=bias, random_state=0)
        labeled = s[y == 1] == 1
        if bias == "easy":
            assert proba[labeled].mean() > proba[~labeled].mean()
        else:
            assert proba[labeled].mean() < proba[~labeled].mean()

    def test_uses_the_given_estimator_and_cv(self, data):
        X, y = data
        default = label_sar_confidence(X, y, random_state=0)
        tree = label_sar_confidence(
            X,
            y,
            estimator=DecisionTreeClassifier(max_depth=2, random_state=0),
            cv=3,
            random_state=0,
        )
        assert tree.sum() == default.sum()
        assert not np.array_equal(tree, default)

    def test_rejects_estimator_without_predict_proba(self, data):
        X, y = data
        with pytest.raises(TypeError, match="must implement predict_proba"):
            label_sar_confidence(X, y, estimator=LinearSVC())

    def test_rejects_unknown_bias(self, data):
        X, y = data
        with pytest.raises(ValueError, match="bias must be one of"):
            label_sar_confidence(X, y, bias="medium")


class TestWeightedChoice:
    def test_zero_weights_are_drawn_last(self):
        weights = [0.0, 1.0, 0.0, 2.0]
        for seed in range(20):
            rng = check_random_state(seed)
            assert set(weighted_choice(rng, weights, 2)) == {1, 3}
            three = weighted_choice(rng, weights, 3)
            assert len(set(three)) == 3
            assert {1, 3} <= set(three)

    def test_all_zero_weights_draw_uniformly(self):
        drawn = weighted_choice(check_random_state(0), np.zeros(5), 5)
        assert sorted(drawn) == [0, 1, 2, 3, 4]

    def test_empty_draw(self):
        assert weighted_choice(check_random_state(0), [1.0, 2.0], 0).size == 0
