"""Generate ``ifpu_reference.npz``: outputs of the original iFPU code, for ``test_ifpu.py``.

Run it with the Python environment of the original iFPNU experiments, passing the iFPNU
directory of the research code::

    python tests/models/data/make_ifpu_reference.py /path/to/pu_monorepo/iFPNU

It builds a synthetic dataset, makes PU labels with the original ``create_pu_data``, fits the
original iFPU (``LGBMClassifier`` with ``PUFLLossNonNegative``) and saves the data and the
test scores. Nothing runs on import, so pytest can collect this file safely.
"""

import sys
from pathlib import Path

_PUBLISHED = {"gamma": 3, "learning_rate": 0.1, "num_boost_round": 10, "n_estimators": 100}

# Case name -> (labeling mechanism, fraction of positives hidden, iFPU settings). The first two
# use the settings of the original experiments, num_boost_round=10 included.
CASES = {
    "published_scar": ("SCAR", 0.5, _PUBLISHED),
    "published_sar": ("SAR", 0.75, _PUBLISHED),
    "gamma1_50trees": ("SCAR", 0.5, {"gamma": 1, "learning_rate": 0.05, "n_estimators": 50}),
}


def main(ifpnu_dir: Path) -> None:
    sys.path.insert(0, str(ifpnu_dir))
    import warnings

    warnings.filterwarnings("ignore")

    import lightgbm as lgb
    import numpy as np
    from data_loaders.pu_data import create_pu_data
    from pu.fpu_xgb import PUFLLossNonNegative
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=800, n_features=10, n_informative=6, weights=[0.8], random_state=0
    )
    X = X.astype("float32")  # as the original CSV loader does
    train, test = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0, stratify=y)
    X_train, X_test, y_train = X[train], X[test], y[train]
    prior = float(np.mean(y_train))

    arrays = {"X_train": X_train, "X_test": X_test, "prior": prior}
    for name, (mechanism, ratio, settings) in CASES.items():
        settings = dict(settings)
        gamma = settings.pop("gamma")
        s = create_pu_data(X_train, y_train.copy(), ratio, mechanism, None)
        model = lgb.LGBMClassifier(
            objective=PUFLLossNonNegative(gamma=gamma, pi_p=prior),
            max_depth=-1,
            verbosity=-100,
            **settings,
        ).fit(X_train, s)
        arrays[f"{name}_s"] = s.astype("int8")
        arrays[f"{name}_scores"] = 1 / (1 + np.exp(-model.predict(X_test)))
        arrays[f"{name}_n_trees"] = model.booster_.num_trees()
    arrays["lightgbm_version"] = lgb.__version__
    np.savez_compressed(Path(__file__).with_name("ifpu_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
