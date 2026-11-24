import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from pypu.datasets import label_scar, make_case_control
from pypu.metrics import (
    lee_liu_score,
    pu_accuracy_score,
    pu_f1_score,
    pu_precision_score,
    pu_recall_score,
    pu_roc_auc_score,
)
from pypu.models import NaivePU


def proxy_accuracy(s, y_pred, prior, setting):
    """Reference for ``pu_accuracy_score``: Definition 1 (Eq. 3) of Wang et al., ICLR 2026.

    ``setting`` is "OS" (one sample: single-training-set) or "TS" (two samples:
    case-control).
    """
    s, y_pred = np.asarray(s), np.asarray(y_pred)
    first = 2 * prior * np.mean(y_pred[s == 1] >= 1)
    if setting == "TS":
        return first + np.mean(y_pred[s == 0] < 1)
    return first + np.mean(y_pred < 1)


def pu_bench_proxy_accuracy(s, y_pred, prior, scenario):
    """PU-Bench's proxy accuracy, from ``evaluate_proxy_metrics`` in train/utils/metrics.py.

    From https://github.com/XiXiphus/PU-Bench (MIT license), with the counts computed from
    arrays; the code released with Wang et al. computes the same.
    """
    s, y_pred = np.asarray(s), np.asarray(y_pred)
    correct_p = np.sum(y_pred[s == 1] == 1)
    correct_u = np.sum(y_pred[s == 0] == 0)
    total_p, total_u = np.sum(s == 1), np.sum(s == 0)
    if scenario == "case-control":
        return 2 * prior * (correct_p / total_p) + (correct_u / total_u)
    return 2 * prior * (correct_p / total_p) + (correct_p + correct_u) / (total_p + total_u)


@pytest.fixture
def exact():
    """Data where the labeled positives have exactly the recall of all positives."""
    y = np.array([1] * 10 + [0] * 10)
    s = np.array([1] * 5 + [0] * 15)
    y_pred = np.array([1, 1, 1, 0, 0] * 2 + [1, 1] + [0] * 8)
    return y, s, y_pred


@pytest.fixture(scope="module")
def large():
    """SCAR data and predictions of a classifier trained on the true labels."""
    X, y = make_classification(n_samples=20000, weights=[0.7], flip_y=0.05, random_state=0)
    s = label_scar(y, label_frequency=0.3, random_state=0)
    model = LogisticRegression().fit(X, y)
    return y, s, model.predict(X), model.predict_proba(X)[:, 1]


class TestExactEstimates:
    def test_recall(self, exact):
        y, s, y_pred = exact
        assert pu_recall_score(s, y_pred) == pytest.approx(recall_score(y, y_pred))

    def test_precision(self, exact):
        y, s, y_pred = exact
        assert pu_precision_score(s, y_pred, prior=0.5) == pytest.approx(precision_score(y, y_pred))

    def test_f1(self, exact):
        y, s, y_pred = exact
        assert pu_f1_score(s, y_pred, prior=0.5) == pytest.approx(f1_score(y, y_pred))

    def test_accuracy(self, exact):
        y, s, y_pred = exact
        assert pu_accuracy_score(s, y_pred, prior=0.5) == pytest.approx(accuracy_score(y, y_pred))

    def test_accuracy_case_control(self, exact):
        # Read as case-control data, the unlabeled samples are the population sample: a
        # third of them positive, with the recall of the labeled positives.
        y, s, y_pred = exact
        unlabeled = s == 0
        estimate = pu_accuracy_score(s, y_pred, prior=1 / 3, scenario="case_control")
        assert estimate == pytest.approx(accuracy_score(y[unlabeled], y_pred[unlabeled]))

    def test_lee_liu_is_precision_times_recall_over_prior(self, exact):
        y, s, y_pred = exact
        expected = precision_score(y, y_pred) * recall_score(y, y_pred) / 0.5
        assert lee_liu_score(s, y_pred) == pytest.approx(expected)

    def test_roc_auc(self):
        y = [1, 1, 0, 1, 1, 0, 0, 0]
        s = [1, 0, 0, 0, 1, 0, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1]
        assert pu_roc_auc_score(s, scores, prior=0.5) == pytest.approx(roc_auc_score(y, scores))


class TestEstimatesOnLargeData:
    def test_classification_metrics(self, large):
        y, s, y_pred, _ = large
        prior = y.mean()
        assert pu_recall_score(s, y_pred) == pytest.approx(recall_score(y, y_pred), abs=0.02)
        assert pu_precision_score(s, y_pred, prior) == pytest.approx(
            precision_score(y, y_pred), abs=0.02
        )
        assert pu_f1_score(s, y_pred, prior) == pytest.approx(f1_score(y, y_pred), abs=0.02)
        assert pu_accuracy_score(s, y_pred, prior) == pytest.approx(
            accuracy_score(y, y_pred), abs=0.02
        )

    def test_roc_auc(self, large):
        y, s, _, scores = large
        naive = roc_auc_score(s, scores)
        corrected = pu_roc_auc_score(s, scores, y.mean())
        assert corrected == pytest.approx(roc_auc_score(y, scores), abs=0.01)
        assert corrected > naive + 0.05


class TestPuAccuracy:
    @pytest.mark.parametrize(("scenario", "setting"), [("single", "OS"), ("case_control", "TS")])
    def test_is_the_proxy_accuracy_minus_the_prior(self, large, scenario, setting):
        y, s, y_pred, _ = large
        prior = float(y.mean())
        expected = proxy_accuracy(s, y_pred, prior, setting) - prior
        estimate = pu_accuracy_score(s, y_pred, prior, scenario=scenario)
        assert estimate == pytest.approx(expected, rel=1e-12)

    def test_case_control_matches_pu_bench(self, large):
        y, s, y_pred, _ = large
        prior = float(y.mean())
        expected = pu_bench_proxy_accuracy(s, y_pred, prior, "case-control") - prior
        estimate = pu_accuracy_score(s, y_pred, prior, scenario="case_control")
        assert estimate == pytest.approx(expected, rel=1e-12)

    def test_case_control_estimates_the_accuracy_on_the_population(self):
        X, y = make_classification(n_samples=40000, weights=[0.7], flip_y=0.05, random_state=1)
        model = LogisticRegression().fit(X, y)
        X_cc, s_cc, y_cc = make_case_control(X, y, label_frequency=0.5, random_state=0)
        unlabeled = s_cc == 0
        prior = float(y_cc[unlabeled].mean())
        estimate = pu_accuracy_score(s_cc, model.predict(X_cc), prior, scenario="case_control")
        assert estimate == pytest.approx(accuracy_score(y, model.predict(X)), abs=0.02)

    def test_ranks_like_accuracy_where_the_released_code_does_not(self):
        rng = np.random.default_rng(0)
        n, prior = 200_000, 0.4
        y = (rng.random(n) < prior).astype(int)
        s = label_scar(y, label_frequency=0.5, random_state=0)

        def predictions(tpr, tnr):
            u = rng.random(n)
            return np.where(y == 1, u < tpr, u >= tnr).astype(int)

        high_recall, high_specificity = predictions(0.95, 0.70), predictions(0.70, 0.90)
        assert accuracy_score(y, high_recall) < accuracy_score(y, high_specificity)
        assert pu_accuracy_score(s, high_recall, prior) < pu_accuracy_score(
            s, high_specificity, prior
        )
        # The single-sample formula of the released code ranks them the other way.
        assert pu_bench_proxy_accuracy(s, high_recall, prior, "single") > (
            pu_bench_proxy_accuracy(s, high_specificity, prior, "single")
        )

    def test_estimates_are_clipped(self):
        assert pu_accuracy_score([1, 0, 0, 0], [1, 0, 0, 0], prior=0.9) == 1.0
        assert pu_accuracy_score([1, 0, 0, 0], [0, 1, 1, 1], prior=0.5) == 0.0

    def test_case_control_prior_may_be_below_the_labeled_fraction(self):
        s, y_pred = [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimate = pu_accuracy_score(s, y_pred, prior=0.5, scenario="case_control")
        assert estimate == pytest.approx(2 * 0.5 * 1 + 0.5 - 0.5)
        with pytest.warns(UserWarning, match="lower than the fraction of labeled"):
            pu_accuracy_score(s, y_pred, prior=0.5)

    def test_case_control_needs_unlabeled_samples(self):
        with pytest.raises(ValueError, match="no unlabeled sample"):
            pu_accuracy_score([1, 1], [1, 0], prior=0.5, scenario="case_control")

    def test_rejects_unknown_scenario(self):
        with pytest.raises(ValueError, match="scenario must be one of"):
            pu_accuracy_score([1, 0], [1, 0], prior=0.5, scenario="case-control")


class TestEdgeCases:
    def test_perfect_predictions(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        s = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        assert lee_liu_score(s, y) == pytest.approx(1 / 0.4)
        assert pu_precision_score(s, y, prior=0.4) == pytest.approx(1.0)
        assert pu_f1_score(s, y, prior=0.4) == pytest.approx(1.0)

    def test_no_positive_prediction(self):
        s, y_pred = [1, 0, 0, 0], [0, 0, 0, 0]
        assert lee_liu_score(s, y_pred) == 0.0
        assert pu_recall_score(s, y_pred) == 0.0
        assert pu_precision_score(s, y_pred, prior=0.5) == 0.0
        assert pu_f1_score(s, y_pred, prior=0.5) == 0.0

    def test_estimates_are_clipped(self):
        # Recall 1 with a quarter of the samples predicted positive and a prior of 1/2
        # would give precision 2.
        assert pu_precision_score([1, 0, 0, 0], [1, 0, 0, 0], prior=0.5) == 1.0
        assert pu_f1_score([1, 0, 0, 0], [1, 0, 0, 0], prior=0.9) == 1.0
        # Labeled positives ranked below every unlabeled sample.
        assert pu_roc_auc_score([0, 0, 0, 1], [0.9, 0.8, 0.7, 0.1], prior=0.5) == 0.0

    def test_prior_below_the_labeled_fraction_warns(self):
        s, y_pred = [1, 1, 0, 0], [1, 1, 0, 0]
        with pytest.warns(UserWarning, match="lower than the fraction of labeled"):
            pu_precision_score(s, y_pred, prior=0.25)
        with pytest.warns(UserWarning, match="lower than the fraction of labeled"):
            auc = pu_roc_auc_score(s, [0.9, 0.8, 0.2, 0.1], prior=0.25)
        assert auc == 1.0


class TestInputValidation:
    @pytest.mark.parametrize(
        "metric",
        [
            lee_liu_score,
            pu_recall_score,
            lambda s, y_pred: pu_precision_score(s, y_pred, 0.5),
            lambda s, y_pred: pu_f1_score(s, y_pred, 0.5),
            lambda s, y_pred: pu_accuracy_score(s, y_pred, 0.5),
            lambda s, y_pred: pu_accuracy_score(s, y_pred, 0.5, scenario="case_control"),
        ],
    )
    def test_prediction_metrics(self, metric):
        with pytest.raises(ValueError, match="threshold them first"):
            metric([1, 0, 0], [0.9, 0.2, 0.1])
        with pytest.raises(ValueError, match="y_pred must hold predicted labels"):
            metric([1, 0, 0], [1, 2, 0])
        with pytest.raises(ValueError, match="no labeled positive"):
            metric([0, 0, 0], [1, 0, 0])
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            metric([1, 0, 0], [1, 0])
        # A single predicted class is fine.
        metric([1, 0, 0], [1, 1, 1])

    def test_roc_auc(self):
        with pytest.raises(ValueError, match="no unlabeled sample"):
            pu_roc_auc_score([1, 1], [0.9, 0.1], prior=0.5)
        with pytest.raises(ValueError, match=r"predict_proba\(X\)\[:, 1\]"):
            pu_roc_auc_score([1, 0], [[0.2, 0.8], [0.7, 0.3]], prior=0.5)
        with pytest.raises(ValueError, match="prior must be in"):
            pu_roc_auc_score([1, 0], [0.9, 0.1], prior=0)


def test_model_selection_with_pu_scorers(pu_data):
    X, y, s = pu_data
    prior = float(y.mean())
    scorers = {
        "lee_liu": make_scorer(lee_liu_score),
        "f1": make_scorer(pu_f1_score, prior=prior),
        "accuracy": make_scorer(pu_accuracy_score, prior=prior),
        "auc": make_scorer(pu_roc_auc_score, response_method="predict_proba", prior=prior),
    }
    search = GridSearchCV(
        NaivePU(prior=prior),
        {"estimator": [LogisticRegression(C=0.001), LogisticRegression(C=1.0)]},
        scoring=scorers,
        refit="auc",
        cv=3,
    ).fit(X, s)
    for name in scorers:
        assert np.all(np.isfinite(search.cv_results_[f"mean_test_{name}"]))
    model = search.best_estimator_
    expected = pu_roc_auc_score(s, model.predict_proba(X)[:, 1], prior)
    assert scorers["auc"](model, X, s) == pytest.approx(expected)
