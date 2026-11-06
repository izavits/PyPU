import numpy as np
import pytest

from pypu.utils import check_prior, check_pu_labels, threshold_from_prior


class TestCheckPULabels:
    @pytest.mark.parametrize(
        "s",
        [
            [0, 1, 0, 1],
            np.array([0, 1, 0, 1], dtype=np.int32),
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([False, True, False, True]),
        ],
    )
    def test_valid_labels(self, s):
        result = check_pu_labels(s)
        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, [0, 1, 0, 1])

    def test_column_vector_is_flattened_with_warning(self):
        from sklearn.exceptions import DataConversionWarning

        with pytest.warns(DataConversionWarning):
            result = check_pu_labels(np.array([[0], [1], [1]]))
        np.testing.assert_array_equal(result, [0, 1, 1])

    @pytest.mark.parametrize(
        "s, match",
        [
            ([], "empty"),
            ([-1, 1, -1], "map them to 0"),
            ([0, 1, 2], "Only binary classification"),
            ([0.0, 0.5, 1.0], "Unknown label type: continuous"),
            (["u", "p", "u"], "must be 1"),
            ([0, 0, 0], "no labeled positive"),
            ([1, 1, 1], "no unlabeled sample"),
            ([0.0, np.nan, 1.0], "NaN"),
        ],
    )
    def test_invalid_labels(self, s, match):
        with pytest.raises(ValueError, match=match):
            check_pu_labels(s)


class TestCheckPrior:
    @pytest.mark.parametrize("prior", [0.01, 0.5, 0.99, np.float32(0.3)])
    def test_valid(self, prior):
        assert check_prior(prior) == pytest.approx(float(prior))

    @pytest.mark.parametrize("prior", [0, 1, -0.1, 1.5])
    def test_out_of_range(self, prior):
        with pytest.raises(ValueError, match="open interval"):
            check_prior(prior)

    @pytest.mark.parametrize("prior", [True, "0.3", None])
    def test_wrong_type(self, prior):
        with pytest.raises(TypeError):
            check_prior(prior)

    def test_warns_when_prior_below_labeled_fraction(self):
        s = np.array([1, 1, 0, 0])
        with pytest.warns(UserWarning, match="label frequency"):
            check_prior(0.2, s)

    def test_no_warning_when_prior_is_plausible(self, recwarn):
        check_prior(0.6, np.array([1, 1, 0, 0]))
        assert not recwarn.list


class TestThresholdFromPrior:
    @pytest.mark.parametrize("prior", [0.1, 0.25, 0.5])
    def test_labels_prior_fraction_positive(self, prior):
        scores = np.random.default_rng(0).random(1000)
        threshold = threshold_from_prior(scores, prior)
        assert np.mean(scores >= threshold) == pytest.approx(prior, abs=0.01)

    def test_rejects_invalid_prior(self):
        with pytest.raises(ValueError):
            threshold_from_prior([0.1, 0.2], 1.0)
