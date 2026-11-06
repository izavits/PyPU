import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from pypu.models import NaivePU


def test_default_estimator_is_logistic_regression(pu_data):
    X, _, s = pu_data
    model = NaivePU().fit(X, s)
    assert isinstance(model.estimator_, LogisticRegression)


def test_probabilities_are_those_of_the_wrapped_classifier(pu_data):
    X, _, s = pu_data
    model = NaivePU().fit(X, s)
    expected = LogisticRegression().fit(X, s).predict_proba(X)
    np.testing.assert_allclose(model.predict_proba(X), expected)


def test_estimator_is_cloned(pu_data):
    X, _, s = pu_data
    estimator = RandomForestClassifier(n_estimators=10, random_state=0)
    model = NaivePU(estimator=estimator).fit(X, s)
    assert model.estimator_ is not estimator
    assert not hasattr(estimator, "estimators_")


def test_estimator_without_predict_proba_raises(pu_data):
    X, _, s = pu_data
    with pytest.raises(TypeError, match="predict_proba"):
        NaivePU(estimator=SVC()).fit(X, s)


def test_threshold_defaults_to_half(pu_data):
    X, _, s = pu_data
    assert NaivePU().fit(X, s).threshold_ == 0.5


def test_prior_sets_fraction_predicted_positive(pu_data):
    X, y, s = pu_data
    prior = y.mean()
    model = NaivePU(prior=prior).fit(X, s)
    assert model.predict(X).mean() == pytest.approx(prior, abs=0.01)


def test_prior_improves_recall_over_default_threshold(pu_data):
    # P(s=1|x) underestimates P(y=1|x), so the 0.5 threshold misses positives.
    X, y, s = pu_data
    recall_default = NaivePU().fit(X, s).predict(X)[y == 1].mean()
    recall_prior = NaivePU(prior=y.mean()).fit(X, s).predict(X)[y == 1].mean()
    assert recall_prior > recall_default


def test_invalid_prior_raises(pu_data):
    X, _, s = pu_data
    with pytest.raises(ValueError, match="open interval"):
        NaivePU(prior=1.2).fit(X, s)


def test_grid_search_over_nested_parameters(pu_data):
    X, _, s = pu_data
    search = GridSearchCV(
        NaivePU(estimator=LogisticRegression()),
        {"estimator__C": [0.1, 1.0]},
        cv=3,
        scoring="roc_auc",
    ).fit(X, s)
    assert search.best_params_["estimator__C"] in (0.1, 1.0)
