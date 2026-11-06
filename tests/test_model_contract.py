"""Tests that every registered model satisfies the BasePUClassifier contract.

New models registered in ``pypu.models`` are picked up automatically.
"""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

import pypu.models
from pypu.base import BasePUClassifier

MODEL_NAMES = pypu.models.__all__

# scikit-learn checks that cannot pass for a PU classifier, by design.
_LABELS_NOT_PU = "the check fits with labels other than 0/1, which are not valid PU labels"
PU_EXPECTED_FAILED_CHECKS = {
    "check_fit_score_takes_y": "fit takes PU labels `s`, not `y`",
    "check_estimators_dtypes": _LABELS_NOT_PU,
    "check_classifier_data_not_an_array": _LABELS_NOT_PU,
    "check_classifiers_classes": _LABELS_NOT_PU,
    "check_fit2d_1feature": _LABELS_NOT_PU,
}


# Parameters set on top of the defaults. Models that need the class prior get the prior of the
# pu_data fixture.
MODEL_PARAMS = {"IFPU": {"prior": 0.3}}


def make_model(name):
    model = getattr(pypu.models, name)(**MODEL_PARAMS.get(name, {}))
    if "random_state" in model.get_params():
        model.set_params(random_state=0)
    return model


# The checks fit on their own data, where the fixed prior is often below the labeled fraction,
# and where random labels make ElkanNoto's estimate of the label frequency imply a prior of 1.
@pytest.mark.filterwarnings("ignore:prior=.* is lower than the fraction of labeled:UserWarning")
@pytest.mark.filterwarnings("ignore:The estimated label frequency .* is clipped:UserWarning")
@parametrize_with_checks(
    [make_model(name) for name in MODEL_NAMES],
    expected_failed_checks=lambda estimator: PU_EXPECTED_FAILED_CHECKS,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


@pytest.mark.parametrize("name", MODEL_NAMES)
class TestModelContract:
    def test_is_pu_classifier(self, name):
        assert isinstance(make_model(name), BasePUClassifier)

    def test_fit_returns_self(self, name, pu_data):
        X, _, s = pu_data
        model = make_model(name)
        assert model.fit(X, s) is model

    def test_fitted_attributes(self, name, pu_data):
        X, _, s = pu_data
        model = make_model(name).fit(X, s)
        np.testing.assert_array_equal(model.classes_, [0, 1])
        assert model.n_features_in_ == X.shape[1]
        assert 0.0 <= model.threshold_ <= 1.0

    def test_predict_proba(self, name, pu_data):
        X, _, s = pu_data
        proba = make_model(name).fit(X, s).predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.all((proba >= 0) & (proba <= 1))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_predict_uses_threshold(self, name, pu_data):
        X, _, s = pu_data
        model = make_model(name).fit(X, s)
        y_pred = model.predict(X)
        assert set(np.unique(y_pred)) <= {0, 1}
        expected = (model.predict_proba(X)[:, 1] >= model.threshold_).astype(int)
        np.testing.assert_array_equal(y_pred, expected)

    def test_decision_function_ranks_like_proba(self, name, pu_data):
        X, _, s = pu_data
        model = make_model(name).fit(X, s)
        scores = model.decision_function(X)
        assert scores.shape == (X.shape[0],)
        order = np.argsort(scores, kind="stable")
        assert np.all(np.diff(model.predict_proba(X)[order, 1]) >= -1e-12)

    def test_better_than_random(self, name, pu_data):
        X, y, s = pu_data
        scores = make_model(name).fit(X, s).predict_proba(X)[:, 1]
        assert roc_auc_score(y, scores) > 0.7

    def test_unfitted_raises(self, name, pu_data):
        X, _, _ = pu_data
        model = make_model(name)
        for method in ("predict", "predict_proba", "decision_function"):
            with pytest.raises(NotFittedError):
                getattr(model, method)(X)

    def test_wrong_number_of_features_raises(self, name, pu_data):
        X, _, s = pu_data
        model = make_model(name).fit(X, s)
        with pytest.raises(ValueError, match="features"):
            model.predict(X[:, :-1])

    @pytest.mark.parametrize(
        "bad_s, match",
        [
            (lambda s: np.where(s == 1, 1, -1), "must be 1"),
            (lambda s: np.zeros_like(s), "only one class"),
            (lambda s: np.ones_like(s), "only one class"),
            (lambda s: s[:-1], "inconsistent numbers of samples"),
        ],
    )
    def test_invalid_labels_raise(self, name, pu_data, bad_s, match):
        X, _, s = pu_data
        with pytest.raises(ValueError, match=match):
            make_model(name).fit(X, bad_s(s))

    def test_clone_gives_unfitted_copy(self, name, pu_data):
        X, _, s = pu_data
        model = make_model(name).fit(X, s)
        cloned = clone(model)
        assert cloned.get_params() == model.get_params()
        with pytest.raises(NotFittedError):
            cloned.predict(X)

    def test_pickle_roundtrip(self, name, pu_data):
        X, _, s = pu_data
        model = make_model(name).fit(X, s)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))

    def test_deterministic(self, name, pu_data):
        X, _, s = pu_data
        first = make_model(name).fit(X, s).predict_proba(X)
        second = make_model(name).fit(X, s).predict_proba(X)
        np.testing.assert_allclose(first, second)

    def test_pipeline_and_cross_validation(self, name, pu_data):
        X, _, s = pu_data
        pipeline = make_pipeline(StandardScaler(), make_model(name))
        pipeline.fit(X, s)
        assert pipeline.predict_proba(X).shape == (X.shape[0], 2)
        scores = cross_val_score(pipeline, X, s, cv=3, scoring="roc_auc")
        assert scores.shape == (3,)

    def test_dataframe_feature_names(self, name, pu_data):
        pd = pytest.importorskip("pandas")
        X, _, s = pu_data
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        model = make_model(name).fit(df, s)
        np.testing.assert_array_equal(model.feature_names_in_, df.columns)
        assert model.predict(df).shape == (X.shape[0],)
