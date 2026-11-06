import numpy as np
import pytest

from pypu.datasets import (
    label_frequency_from_pollution,
    label_scar,
    pollution_from_label_frequency,
    pu_summary,
)


def test_summary_values():
    y = np.array([1] * 40 + [0] * 60)
    s = label_scar(y, label_frequency=0.5, random_state=0)
    assert pu_summary(y, s) == {
        "n_samples": 100,
        "n_positive": 40,
        "n_labeled": 20,
        "n_unlabeled": 80,
        "prior": 0.4,
        "label_frequency": 0.5,
        "pollution": 0.25,
    }


def test_summary_types():
    summary = pu_summary([1, 1, 0, 0], [1, 0, 0, 0])
    for key in ("n_samples", "n_positive", "n_labeled", "n_unlabeled"):
        assert type(summary[key]) is int
    for key in ("prior", "label_frequency", "pollution"):
        assert type(summary[key]) is float


def test_summary_rejects_labeled_negatives():
    with pytest.raises(ValueError, match="1 labeled samples"):
        pu_summary([1, 1, 0, 0], [1, 0, 1, 0])


def test_summary_rejects_invalid_pu_labels():
    with pytest.raises(ValueError, match="no labeled positive"):
        pu_summary([1, 1, 0, 0], [0, 0, 0, 0])


def test_summary_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        pu_summary([1, 1, 0, 0], [1, 0, 0])


def test_pollution_matches_the_summary():
    # 40 positives, half labeled: 20 of the 80 unlabeled samples are positive.
    assert pollution_from_label_frequency(0.5, prior=0.4) == pytest.approx(0.25)
    assert label_frequency_from_pollution(0.25, prior=0.4) == pytest.approx(0.5)


@pytest.mark.parametrize("label_frequency", [0.01, 0.3, 0.5, 0.99, 1.0])
@pytest.mark.parametrize("prior", [0.05, 0.5, 0.9])
def test_conversions_are_inverse(label_frequency, prior):
    pollution = pollution_from_label_frequency(label_frequency, prior)
    assert 0 <= pollution < prior
    assert label_frequency_from_pollution(pollution, prior) == pytest.approx(label_frequency)


def test_no_pollution_means_every_positive_is_labeled():
    assert label_frequency_from_pollution(0, prior=0.3) == 1.0
    assert pollution_from_label_frequency(1, prior=0.3) == 0.0


@pytest.mark.parametrize("pollution", [-0.1, 0.3, 0.5])
def test_rejects_pollution_outside_range(pollution):
    with pytest.raises(ValueError, match=r"pollution must be in \[0, prior\)"):
        label_frequency_from_pollution(pollution, prior=0.3)


@pytest.mark.parametrize("pollution", ["0.1", None, False])
def test_rejects_pollution_of_wrong_type(pollution):
    with pytest.raises(TypeError, match="pollution must be a float"):
        label_frequency_from_pollution(pollution, prior=0.3)


def test_conversions_validate_prior_and_label_frequency():
    with pytest.raises(ValueError, match="prior must be in"):
        pollution_from_label_frequency(0.5, prior=1.0)
    with pytest.raises(ValueError, match="label_frequency must be in"):
        pollution_from_label_frequency(0.0, prior=0.5)
    with pytest.raises(ValueError, match="prior must be in"):
        label_frequency_from_pollution(0.1, prior=0.0)
