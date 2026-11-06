import math

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from pypu.datasets import PUSplit, label_sar_knn, make_pu_split
from pypu.datasets._split import _MECHANISMS


@pytest.fixture
def data():
    return make_classification(n_samples=400, n_features=5, weights=[0.7], random_state=0)


def test_split_shapes_and_labels(data):
    X, y = data
    split = make_pu_split(X, y, label_frequency=0.4, random_state=0)
    assert isinstance(split, PUSplit)
    X_train, X_test, s_train, y_train, y_test = split
    assert X_train.shape == (300, 5)
    assert X_test.shape == (100, 5)
    assert s_train.shape == y_train.shape == (300,)
    assert y_test.shape == (100,)
    assert split.s_train is s_train
    assert np.all(s_train <= y_train)
    assert s_train.sum() == math.floor(0.4 * y_train.sum() + 0.5)


def test_split_is_stratified(data):
    X, y = data
    split = make_pu_split(X, y, random_state=0)
    assert split.y_train.mean() == pytest.approx(y.mean(), abs=0.01)
    assert split.y_test.mean() == pytest.approx(y.mean(), abs=0.01)


def test_passes_parameters_to_the_mechanism(data):
    X, y = data
    split = make_pu_split(
        X, y, label_frequency=0.3, mechanism="sar_knn", bias="far", n_neighbors=5, random_state=0
    )
    rng = check_random_state(0)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=rng)
    expected = label_sar_knn(
        X_train, y_train, label_frequency=0.3, bias="far", n_neighbors=5, random_state=rng
    )
    np.testing.assert_array_equal(split.X_train, X_train)
    np.testing.assert_array_equal(split.s_train, expected)


@pytest.mark.parametrize("mechanism", list(_MECHANISMS))
def test_every_mechanism(data, mechanism):
    X, y = data
    split = make_pu_split(X, y, mechanism=mechanism, random_state=0)
    assert 0 < split.s_train.sum() < split.y_train.sum()
    assert np.all(split.s_train <= split.y_train)


def test_keeps_dataframes(data):
    X, y = data
    split = make_pu_split(pd.DataFrame(X), pd.Series(y), test_size=50, random_state=0)
    assert isinstance(split.X_train, pd.DataFrame)
    assert len(split.X_test) == 50


def test_same_seed_same_split(data):
    X, y = data
    first = make_pu_split(X, y, mechanism="sar_cluster", random_state=5)
    second = make_pu_split(X, y, mechanism="sar_cluster", random_state=5)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_rejects_unknown_mechanism(data):
    X, y = data
    with pytest.raises(ValueError, match="mechanism must be one of"):
        make_pu_split(X, y, mechanism="SAR")


def test_rejects_return_propensity(data):
    X, y = data
    with pytest.raises(TypeError, match="does not return propensities"):
        make_pu_split(X, y, mechanism="sar_propensity", return_propensity=True)


def test_unknown_mechanism_parameter_raises(data):
    X, y = data
    with pytest.raises(TypeError, match="n_neighbors"):
        make_pu_split(X, y, mechanism="scar", n_neighbors=3)
