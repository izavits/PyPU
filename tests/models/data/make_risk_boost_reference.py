"""Generate ``risk_boost_reference.npz``: outputs of the original CSBoost PU objectives.

Run it with the Python environment of the PU Hellinger trees experiments (XGBoost 2.1; on
Apple silicon under Rosetta), passing the directory of that research code::

    arch -x86_64 venv/bin/python /path/to/make_risk_boost_reference.py /path/to/PU_Hellinger_Trees

It builds a synthetic PU dataset, fits the original ``CSBoost`` with the ``nnpu``, ``puwce``
(uPU) and ``imbnnpu`` objectives as ``nnPU_clean.py``, ``uPU_clean.py`` and
``imbalancedNNPU_clean.py`` do, and saves the data, the raw test scores, and the gradients
and hessians of each objective at fixed raw scores (which do not depend on the XGBoost
version). Nothing runs on import, so pytest can collect this file safely.
"""

import sys
from pathlib import Path

# Case name -> CSBoost arguments. The first three are the experiments' calls.
CASES = {
    "nnpu_experiments": {"obj": "nnpu", "random_state": 0},
    "upu_experiments": {"obj": "puwce", "random_state": 0},
    "imbnnpu_experiments": {"obj": "imbnnpu", "oversampling": 0.5, "random_state": 0},
    "imbnnpu_oversampling03": {"obj": "imbnnpu", "oversampling": 0.3, "random_state": 1},
}


def main(trees_dir: Path) -> None:
    sys.path.insert(0, str(trees_dir))
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np
    import xgboost
    from ecsmodels.methodologies import cs_boost
    from ecsmodels.methodologies.cs_boost import CSBoost
    from sklearn.datasets import make_classification

    # Record the objective each fit passes to xgboost.train.
    captured = {}
    real_train = cs_boost.xgb.train

    def recording_train(*args, **kwargs):
        captured["obj"] = kwargs.get("obj")
        captured["dtrain"] = kwargs.get("dtrain")
        return real_train(*args, **kwargs)

    cs_boost.xgb.train = recording_train

    X, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, weights=[0.75], random_state=0
    )
    s = ((y == 1) & (np.random.RandomState(0).rand(y.size) < 0.5)).astype(int)
    X_train, X_test, y_train, s_train = X[:450], X[450:], y[:450], s[:450]
    prior, labeled_fraction = float(y_train.mean()), float(s_train.mean())
    # The experiments' cost matrix: 1 for false positives and false negatives.
    cost_matrix = np.zeros((len(X_train), 2, 2))
    cost_matrix[:, 0, 1] = 1.0
    cost_matrix[:, 1, 0] = 1.0

    # Raw scores at which the objectives' gradients and hessians are recorded.
    raw_grid = np.linspace(-8, 8, len(X_train)).astype(np.float32)
    arrays = {
        "X_train": X_train,
        "X_test": X_test,
        "s_train": s_train,
        "y_test": y[450:],
        "prior": prior,
        "raw_grid": raw_grid,
    }
    for name, params in CASES.items():
        booster = CSBoost(prior_y=prior, prob_lab=labeled_fraction, validation=False, **params)
        booster = booster.fit(X_train, s_train, cost_matrix_train=cost_matrix)
        arrays[f"{name}_raw"] = booster.inplace_predict(X_test)
        arrays[f"{name}_n_rounds"] = booster.num_boosted_rounds()
        grad, hess = captured["obj"](raw_grid, captured["dtrain"])
        arrays[f"{name}_grad"] = np.asarray(grad)
        arrays[f"{name}_hess"] = np.asarray(hess)
    arrays["xgboost_version"] = xgboost.__version__
    np.savez_compressed(Path(__file__).with_name("risk_boost_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
