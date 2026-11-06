"""Generate ``sarem_reference.npz``: outputs of the original SAR-EM code, for ``test_sarem.py``.

Run it with the Python environment of the original iFPNU experiments, passing the iFPNU
directory of the research code::

    python tests/models/data/make_sarem_reference.py /path/to/pu_monorepo/iFPNU

It builds a synthetic dataset, makes PU labels with the original ``create_pu_data``, runs the
SAR-PU ``pu_learn_sar_em`` the experiments used, and saves the data, the test scores and
propensities, the number of iterations and the log-likelihoods. As in the experiments, the
features and the PU labels are float32. Nothing runs on import, so pytest can collect this
file safely.
"""

import sys
from pathlib import Path

# Case name -> (labeling mechanism, fraction of positives hidden, pu_learn_sar_em arguments).
# The first two are the experiments' call, pu_learn_sar_em(X, s, range(X.shape[1])).
CASES = {
    "experiments_scar": ("SCAR", 0.5, {}),
    "experiments_sar": ("SAR", 0.75, {}),
    "propensity_subset": (
        "SAR",
        0.5,
        {
            "propensity_attributes": [0, 2],
            "convergence_window": 5,
            "slope_eps": 1e-3,
            "ll_eps": 1e-3,
            "refit_classifier": False,
        },
    ),
}


def main(ifpnu_dir: Path) -> None:
    sys.path.insert(0, str(ifpnu_dir))
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np
    import sklearn
    from data_loaders.pu_data import create_pu_data
    from models.helper_files.sarpu.pu_learning import pu_learn_sar_em
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=800, n_features=10, n_informative=6, weights=[0.8], random_state=0
    )
    X, y = X.astype("float32"), y.astype("float32")  # as the original CSV loader does
    train, test = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0, stratify=y)
    X_train, X_test, y_train = X[train], X[test], y[train]

    arrays = {"X_train": X_train, "X_test": X_test}
    for name, (mechanism, ratio, settings) in CASES.items():
        settings = dict(settings)
        attributes = settings.pop("propensity_attributes", range(X_train.shape[1]))
        s = create_pu_data(X_train, y_train.copy(), ratio, mechanism, None).flatten()
        classifier, propensity, info = pu_learn_sar_em(X_train, s, attributes, **settings)
        arrays[f"{name}_s"] = s.astype("int8")
        arrays[f"{name}_scores"] = classifier.predict_proba(X_test)
        arrays[f"{name}_propensity"] = propensity.predict_proba(X_test)
        arrays[f"{name}_iterations"] = info["nb_iterations"] + 1
        arrays[f"{name}_loglikelihoods"] = np.asarray(info["loglikelihoods"])
    arrays["sklearn_version"] = sklearn.__version__
    np.savez_compressed(Path(__file__).with_name("sarem_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
