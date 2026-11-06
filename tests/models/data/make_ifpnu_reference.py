"""Generate ``ifpnu_reference.npz``: outputs of the original iFPNU code, for ``test_ifpnu.py``.

Run it with the Python environment of the original iFPNU experiments, passing the iFPNU
directory of the research code::

    python tests/models/data/make_ifpnu_reference.py /path/to/pu_monorepo/iFPNU

It builds a synthetic dataset and makes PU labels with the original ``create_pu_data``. The
first step follows ``main_2step_ad.py`` (the most anomalous ``int(n * prior)`` samples stay
unlabeled), with PyOD's ``IForest`` seeded, since the detectors of the experiments were not.
The second step trains LightGBM with the original ``pnu_focal_loss`` as the experiments did.
The data, the anomaly scores, the PNU labels and the test scores are saved. Nothing runs on
import, so pytest can collect this file safely.
"""

import sys
from pathlib import Path

_PUBLISHED = {"eta": 0.5, "gamma": 3.0, "rounds": 100, "learning_rate": 0.1}

# Case name -> (labeling mechanism, fraction of positives hidden, settings). The first two use
# the settings of the original experiments.
CASES = {
    "published_scar": ("SCAR", 0.5, _PUBLISHED),
    "published_sar": ("SAR", 0.75, _PUBLISHED),
    "eta08_gamma1": ("SCAR", 0.5, {"eta": 0.8, "gamma": 1.0, "rounds": 30, "learning_rate": 0.05}),
}


def first_step(y_, S1, loss_prior):
    """Turn PU labels into PNU labels, as the DIF, EIF and OptIForest branches of
    ``main_2step_ad.py`` do.
    """
    anomaly_percentage = loss_prior
    num_outliers_to_keep = int(len(S1) * anomaly_percentage)
    anomaly_scores_idx = sorted(range(len(S1)), key=lambda k: S1[k])
    anomaly_scores_idx.reverse()
    SS1 = [0] * len(S1)
    for idx in range(num_outliers_to_keep):
        SS1[anomaly_scores_idx[idx]] = 1
    y_pnu = []
    for i in range(len(y_)):
        if y_[i] == 1.0:
            y_pnu.append(y_[i])
        elif y_[i] == 0.0:
            if SS1[i] == 1:
                y_pnu.append(-1.0)
            else:
                y_pnu.append(y_[i])
    return y_pnu


def main(ifpnu_dir: Path) -> None:
    sys.path.insert(0, str(ifpnu_dir))
    import warnings

    warnings.filterwarnings("ignore")

    import lightgbm as lgb
    import numpy as np
    from data_loaders.pu_data import create_pu_data
    from pu.fpu_xgb import pnu_focal_loss
    from pyod.models.iforest import IForest
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=800, n_features=10, n_informative=6, weights=[0.8], random_state=0
    )
    X = X.astype("float32")  # as the original CSV loader does
    train, test = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0, stratify=y)
    X_train, X_test, y_train = X[train], X[test], y[train]
    loss_prior = float(np.mean(y_train))

    S1 = IForest(random_state=0).fit(X_train).decision_function(X_train)
    arrays = {
        "X_train": X_train,
        "X_test": X_test,
        "prior": loss_prior,
        "anomaly_scores": S1,
    }
    for name, (mechanism, ratio, settings) in CASES.items():
        y_ = create_pu_data(X_train, y_train.copy(), ratio, mechanism, None).flatten()
        y_pnu = np.array(first_step(y_, S1, loss_prior))
        d_train_lgb = lgb.Dataset(X_train, label=np.where(y_train == -1, 0, y_pnu))
        custom_loss = pnu_focal_loss(
            y_true=y_pnu,
            gamma=settings["gamma"],
            eta=settings["eta"],
            pi_p=loss_prior,
            lgb=True,
        )
        params = {
            "objective": custom_loss,
            "verbosity": 0,
            "n_estimators": settings["rounds"],
            "max_depth": -1,
            "learning_rate": settings["learning_rate"],
        }
        clf = lgb.train(params=params, train_set=d_train_lgb, num_boost_round=settings["rounds"])
        arrays[f"{name}_s"] = y_.astype("int8")
        arrays[f"{name}_pnu"] = y_pnu.astype("int8")
        arrays[f"{name}_scores"] = 1 / (1 + np.exp(-clf.predict(X_test)))
        arrays[f"{name}_n_trees"] = clf.num_trees()
    arrays["lightgbm_version"] = lgb.__version__
    np.savez_compressed(Path(__file__).with_name("ifpnu_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
