"""Generate ``hellinger_reference.npz``: outputs of the original PU Hellinger trees and forest.

Run it with the Python environment of the PU Hellinger trees experiments (Python 3.8,
scikit-learn 1.0, the compiled ``pu_tree_simplified`` extensions; on Apple silicon under
Rosetta), passing the directory of that research code::

    arch -x86_64 venv/bin/python /path/to/make_hellinger_reference.py /path/to/PU_Hellinger_Trees

It builds two synthetic PU datasets, one of them with constant, duplicated and discrete
columns, fits the original ``DecisionTreeClassifier`` and ``PURandomForestClassifier`` with
several settings, and saves the data, the structure of each tree and the test predictions.
Nothing runs on import, so pytest can collect this file safely.
"""

import sys
from pathlib import Path

# Case name -> (dataset, parameters). "p_y" is passed to fit, as the experiments did.
TREE_CASES = {
    # As in the experiments: max_depth=5, the true prior.
    "tree_experiments": ("plain", {"random_state": 0, "max_depth": 5}),
    "tree_full": ("plain", {"random_state": 1}),
    "tree_sqrt_leaf": (
        "plain",
        {"random_state": 2, "max_features": "sqrt", "min_samples_leaf": 5, "min_samples_split": 12},
    ),
    # Prunes an inner split (improvement 0.274) but not the root (0.292).
    "tree_min_decrease": (
        "ties",
        {"random_state": 7, "max_features": 3, "min_impurity_decrease": 0.28},
    ),
    "tree_ties": ("ties", {"random_state": 4, "max_features": 3, "max_depth": 8}),
    "tree_ties_all": ("ties", {"random_state": 5}),
}
FOREST_CASES = {
    # As in the experiments: the class defaults, bags of n_unlabeled unlabeled samples.
    "forest_experiments": ("plain", {"random_state": 0, "max_samples": "n_unlabeled"}),
    "forest_small": (
        "ties",
        {
            "random_state": 6,
            "n_estimators": 10,
            "max_depth": 4,
            "max_features": None,
            "max_samples": 150,
        },
    ),
}


def datasets():
    import numpy as np
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500, n_features=6, n_informative=4, weights=[0.7], random_state=0
    )
    s = ((y == 1) & (np.random.RandomState(0).rand(y.size) < 0.5)).astype(int)
    # Add a constant column, a copy of column 1 and two discrete columns (many ties).
    X_ties = np.column_stack(
        [X, np.ones(len(X)), X[:, 1], np.round(X[:, 0]), np.round(2 * X[:, 2]) / 2]
    )
    return {"plain": (X, y, s), "ties": (X_ties, y, s)}


def tree_arrays(tree):
    import numpy as np

    t = tree.tree_
    return {
        "children_left": t.children_left.copy(),
        "children_right": t.children_right.copy(),
        "feature": t.feature.copy(),
        "threshold": t.threshold.copy(),
        "value": np.asarray(t.value)[:, 0, :].copy(),
        "n_node_samples": t.n_node_samples.copy(),
    }


def main(trees_dir: Path) -> None:
    sys.path.insert(0, str(trees_dir))
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np
    import sklearn
    from pu_tree_simplified._pu_classes import DecisionTreeClassifier
    from pu_tree_simplified._pu_randomforest import PURandomForestClassifier

    data = datasets()
    arrays = {}
    for name, (X, y, s) in data.items():
        arrays[f"{name}_X"] = X
        arrays[f"{name}_s"] = s
        arrays[f"{name}_prior"] = float(y[:400].mean())

    for case, (dataset, params) in TREE_CASES.items():
        X, y, s = data[dataset]
        X_train, X_test, s_train = X[:400], X[400:], s[:400]
        tree = DecisionTreeClassifier(**params).fit(X_train, s_train, p_y=float(y[:400].mean()))
        for key, value in tree_arrays(tree).items():
            arrays[f"{case}_{key}"] = value
        arrays[f"{case}_proba"] = tree.predict_proba(X_test)[:, 1]
        arrays[f"{case}_pu_proba"] = tree.predict_pu_proba(X_test)[:, 1]

    for case, (dataset, params) in FOREST_CASES.items():
        X, y, s = data[dataset]
        X_train, X_test, s_train = X[:400], X[400:], s[:400]
        params = dict(params)
        if params.get("max_samples") == "n_unlabeled":
            params["max_samples"] = int((s_train == 0).sum())
        forest = PURandomForestClassifier(pu_biased_bootstrap=True, **params)
        forest.fit(X_train, s_train, p_y=float(y[:400].mean()))
        arrays[f"{case}_proba"] = forest.predict_proba(X_test)[:, 1]
        arrays[f"{case}_node_counts"] = np.array([t.tree_.node_count for t in forest.estimators_])
        first = tree_arrays(forest.estimators_[0])
        for key, value in first.items():
            arrays[f"{case}_first_{key}"] = value

    arrays["sklearn_version"] = sklearn.__version__
    np.savez_compressed(Path(__file__).with_name("hellinger_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
