import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.datasets import make_classification

from pypu.datasets import make_case_control


@pytest.fixture
def data():
    return make_classification(n_samples=1000, n_features=4, weights=[0.7], random_state=0)


@pytest.mark.parametrize("label_frequency", [0.2, 0.5, 0.8])
def test_case_control_proportions(data, label_frequency):
    X, y = data
    X_cc, s, y_cc = make_case_control(X, y, label_frequency=label_frequency, random_state=0)
    n_positive = int(y.sum())
    assert X_cc.shape == (s.size, X.shape[1])
    assert s.dtype == np.int64
    assert y_cc.dtype == np.int64
    assert np.all(s <= y_cc), "a negative was labeled"
    # Every positive is kept, a label_frequency share of them labeled.
    assert y_cc.sum() == n_positive
    assert s.sum() == math.floor(label_frequency * n_positive + 0.5)  # half up
    # The unlabeled set has the class prior of the full data, up to rounding.
    unlabeled = s == 0
    assert y_cc[unlabeled].mean() == pytest.approx(y.mean(), abs=1 / unlabeled.sum())


def test_returns_rows_of_X_in_their_original_order(data):
    X, y = data
    frame = pd.DataFrame(X, index=np.arange(1000, 2000))
    X_cc, _, y_cc = make_case_control(frame, y, random_state=0)
    assert isinstance(X_cc, pd.DataFrame)
    assert X_cc.index.is_monotonic_increasing
    assert X_cc.index.is_unique
    rows = X_cc.index.to_numpy() - 1000
    np.testing.assert_array_equal(X_cc.to_numpy(), X[rows])
    np.testing.assert_array_equal(y_cc, y[rows])


def test_accepts_sparse_and_lists(data):
    X, y = data
    expected, s_expected, _ = make_case_control(X, y, random_state=0)
    X_sparse, s_sparse, _ = make_case_control(sparse.csr_matrix(X), y, random_state=0)
    X_list, s_list, _ = make_case_control(X.tolist(), list(y), random_state=0)
    np.testing.assert_array_equal(X_sparse.toarray(), expected)
    np.testing.assert_array_equal(np.asarray(X_list), expected)
    np.testing.assert_array_equal(s_sparse, s_expected)
    np.testing.assert_array_equal(s_list, s_expected)


def test_same_seed_same_data(data):
    X, y = data
    first = make_case_control(X, y, random_state=3)
    second = make_case_control(X, y, random_state=3)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_rejects_label_frequency_one(data):
    X, y = data
    with pytest.raises(ValueError, match=r"label_frequency must be in \(0, 1\)"):
        make_case_control(X, y, label_frequency=1.0)


def test_rejects_labeling_every_positive():
    y = np.array([1] * 10 + [0] * 10)
    with pytest.raises(ValueError, match="leaving none for the unlabeled set"):
        make_case_control(np.zeros((20, 1)), y, label_frequency=0.99)


def test_rejects_too_few_negatives():
    y = np.array([1] * 10 + [0])
    with pytest.raises(ValueError, match="Too few negatives"):
        make_case_control(np.zeros((11, 1)), y, label_frequency=0.9)


def test_rejects_inconsistent_lengths(data):
    X, y = data
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        make_case_control(X, y[:-1])
