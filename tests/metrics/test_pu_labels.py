import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from pypu.datasets import label_scar
from pypu.metrics import (
    lee_liu_score,
    pu_f1_score,
    pu_precision_score,
    pu_recall_score,
    pu_roc_auc_score,
)
from pypu.models import NaivePU


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

    def test_roc_auc(self, large):
        y, s, _, scores = large
        naive = roc_auc_score(s, scores)
        corrected = pu_roc_auc_score(s, scores, y.mean())
        assert corrected == pytest.approx(roc_auc_score(y, scores), abs=0.01)
        assert corrected > naive + 0.05


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
