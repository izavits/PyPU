import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_recall_curve

from pypu.metrics import f1_at_prior, max_f1_score, precision_at_k, r_precision, recall_at_k

# ---------------------------------------------------------------- references from the
# iFPNU experiments (pu/evaluation.py and train_and_validation.py), logic unchanged.


def legacy_rank_at_n(n, y_pred_proba, y_test):
    top_n = np.argsort(y_pred_proba)[::-1][:n]
    correct = sum(1 for index in top_n if y_test[index] == 1)
    all_in_test_pos = sum(1 for i in y_test if i == 1)
    return float(correct / n), float(correct / all_in_test_pos)


def legacy_r_precision_at_n(y_pred_proba, y_test):
    num_positives_in_test = sum(1 for y in y_test if y == 1)
    precision, _ = legacy_rank_at_n(num_positives_in_test, y_pred_proba, y_test)
    return precision


def legacy_f1_at_prior_threshold(y_true, scores, prior):
    threshold = np.quantile(scores, 1 - prior)
    predictions = (np.asarray(scores) >= threshold).astype(int)
    f1_neg, f1_pos = f1_score(y_true, predictions, labels=[0, 1], average=None, zero_division=0)
    return f1_pos, f1_neg, (f1_pos + f1_neg) / 2


def legacy_best_f1(y_test, pred_proba):
    p, r, _ = precision_recall_curve(y_test, pred_proba)
    numerator = 2 * r * p
    denom = r + p
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    return np.max(f1_scores)


# ----------------------------------------------------------------


@pytest.fixture(params=[0, 1, 2])
def scored(request):
    """Labels and untied scores that rank positives higher on average."""
    rng = np.random.default_rng(request.param)
    y = (rng.random(300) < 0.3).astype(int)
    y[:2] = [0, 1]
    scores = rng.normal(size=y.size) + 1.5 * y
    return y, scores


class TestRanking:
    @pytest.mark.parametrize("k", [1, 10, 50, 300])
    def test_matches_the_legacy_rank_at_n(self, scored, k):
        y, scores = scored
        precision, recall = legacy_rank_at_n(k, scores, y)
        assert precision_at_k(y, scores, k=k) == pytest.approx(precision)
        assert recall_at_k(y, scores, k=k) == pytest.approx(recall)

    def test_r_precision_matches_the_legacy_function(self, scored):
        y, scores = scored
        assert r_precision(y, scores) == pytest.approx(legacy_r_precision_at_n(scores, y))

    def test_default_k_is_the_number_of_positives(self, scored):
        y, scores = scored
        k = int(y.sum())
        assert precision_at_k(y, scores) == precision_at_k(y, scores, k=k)
        assert recall_at_k(y, scores) == recall_at_k(y, scores, k=k)
        assert recall_at_k(y, scores) == pytest.approx(r_precision(y, scores))

    def test_ties_at_the_cutoff_count_in_proportion(self):
        y = [1, 0, 1, 0]
        scores = [1.0, 0.5, 0.5, 0.5]
        # Top 2: the first sample, then one of three tied samples, one of which is positive.
        assert precision_at_k(y, scores, k=2) == pytest.approx((1 + 1 / 3) / 2)
        assert recall_at_k(y, scores, k=2) == pytest.approx((1 + 1 / 3) / 2)

    def test_constant_scores_give_the_prior(self):
        y = [1, 0, 0, 1, 0]
        assert precision_at_k(y, np.zeros(5), k=3) == pytest.approx(0.4)

    def test_perfect_ranking(self):
        y = [0, 1, 0, 1]
        assert r_precision(y, [0.1, 0.9, 0.2, 0.8]) == 1.0
        assert recall_at_k(y, [0.1, 0.9, 0.2, 0.8], k=4) == 1.0

    @pytest.mark.parametrize("k", [0, 5, 1.5, True, -1])
    def test_rejects_invalid_k(self, k):
        with pytest.raises(ValueError, match=r"k must be an integer in \[1, 4\]"):
            precision_at_k([1, 0, 1, 0], [0.4, 0.3, 0.2, 0.1], k=k)


class TestF1AtPrior:
    @pytest.mark.parametrize("prior", [0.1, 0.3, 0.5])
    def test_matches_the_legacy_function(self, scored, prior):
        y, scores = scored
        f1_pos, f1_neg, f1_macro = legacy_f1_at_prior_threshold(y, scores, prior)
        assert f1_at_prior(y, scores, prior) == pytest.approx(f1_pos)
        assert f1_at_prior(y, scores, prior, average="macro") == pytest.approx(f1_macro)
        np.testing.assert_allclose(f1_at_prior(y, scores, prior, average=None), [f1_neg, f1_pos])

    def test_weighted_average_is_a_float(self, scored):
        y, scores = scored
        assert isinstance(f1_at_prior(y, scores, 0.3, average="weighted"), float)

    def test_validates_the_prior(self, scored):
        y, scores = scored
        with pytest.raises(ValueError, match="prior must be in"):
            f1_at_prior(y, scores, 1.0)
        with pytest.raises(TypeError, match="prior must be a float"):
            f1_at_prior(y, scores, "0.3")


class TestMaxF1:
    def test_matches_the_legacy_computation(self, scored):
        y, scores = scored
        assert max_f1_score(y, scores) == pytest.approx(legacy_best_f1(y, scores))

    def test_is_the_best_f1_over_all_thresholds(self, scored):
        y, scores = scored
        best = max(f1_score(y, (scores >= t).astype(int)) for t in np.unique(scores))
        assert max_f1_score(y, scores) == pytest.approx(best)

    def test_perfect_ranking(self):
        assert max_f1_score([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.0


@pytest.mark.parametrize(
    "metric",
    [
        precision_at_k,
        recall_at_k,
        r_precision,
        max_f1_score,
        lambda y, scores: f1_at_prior(y, scores, 0.5),
    ],
)
class TestInputValidation:
    def test_rejects_invalid_true_labels(self, metric):
        with pytest.raises(ValueError, match="y_true must hold the true binary labels"):
            metric([1, 2, 0], [0.3, 0.2, 0.1])
        with pytest.raises(ValueError, match="y_true has no positive"):
            metric([0, 0, 0], [0.3, 0.2, 0.1])

    def test_rejects_2d_scores(self, metric):
        with pytest.raises(ValueError, match=r"predict_proba\(X\)\[:, 1\]"):
            metric([1, 0], [[0.2, 0.8], [0.7, 0.3]])

    def test_rejects_non_finite_scores(self, metric):
        with pytest.raises(ValueError, match="NaN"):
            metric([1, 0], [np.nan, 0.3])

    def test_rejects_inconsistent_lengths(self, metric):
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            metric([1, 0, 1], [0.3, 0.2])
