"""Generate ``bagging_reference.npz``: outputs of the original PU bagging, for ``test_bagging.py``.

Run it with the Python environment of the PU Hellinger trees experiments (scikit-learn 1.0),
passing the directory of that research code::

    python tests/models/data/make_bagging_reference.py /path/to/PU_Hellinger_Trees

It builds a synthetic PU dataset, fits the original ``BaggingPuClassifier`` with several
settings and saves the data, the test scores, the out-of-bag estimates, and the features and
``random_state`` of each classifier in the ensemble. Nothing runs on import, so pytest can
collect this file safely.
"""

import sys
from pathlib import Path


def cases():
    """Case name -> (base estimator, BaggingPuClassifier parameters)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        # The class defaults, apart from the number of bags.
        "tree_defaults": (None, {"n_estimators": 10, "random_state": 0}),
        # As in the experiments: linear SVMs (which vote) on balanced bags.
        "svm_experiments": (
            SVC(kernel="linear"),
            {"n_estimators": 20, "max_samples": "n_labeled", "random_state": 3, "n_jobs": -1},
        ),
        "tree_subsampled": (
            DecisionTreeClassifier(max_depth=3),
            {
                "n_estimators": 15,
                "max_samples": 0.3,
                "max_features": 0.5,
                "bootstrap": False,
                "oob_score": False,
                "random_state": 7,
            },
        ),
        # No sample_weight in fit, and no random_state parameter.
        "knn_no_weights": (
            KNeighborsClassifier(),
            {"n_estimators": 10, "max_samples": "n_labeled", "random_state": 1},
        ),
        "logreg_bootstrap_features": (
            LogisticRegression(),
            {
                "n_estimators": 10,
                "max_features": 4,
                "bootstrap_features": True,
                "random_state": 5,
            },
        ),
    }


def main(trees_dir: Path) -> None:
    sys.path.insert(0, str(trees_dir))
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np
    import sklearn
    from pos_noisyneg.PU_bagging import BaggingPuClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=5, weights=[0.7], random_state=0
    )
    X_train, X_test, y_train = X[:300], X[300:], y[:300]
    rng = np.random.RandomState(0)
    s = ((y_train == 1) & (rng.rand(y_train.size) < 0.5)).astype(int)

    arrays = {"X_train": X_train, "X_test": X_test, "s": s}
    for name, (estimator, params) in cases().items():
        params = dict(params)
        if params.get("max_samples") == "n_labeled":
            params["max_samples"] = int(s.sum())
        model = BaggingPuClassifier(estimator, **params).fit(X_train, s)
        arrays[f"{name}_scores"] = model.predict_proba(X_test)[:, 1]
        arrays[f"{name}_features"] = np.array(model.estimators_features_)
        # -1 for classifiers without a random_state parameter.
        arrays[f"{name}_random_states"] = np.array(
            [getattr(member, "random_state", -1) for member in model.estimators_]
        )
        if params.get("oob_score", True):
            arrays[f"{name}_oob"] = model.oob_decision_function_
    arrays["sklearn_version"] = sklearn.__version__
    np.savez_compressed(Path(__file__).with_name("bagging_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
