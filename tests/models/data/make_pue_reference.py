"""Generate ``pue_reference.npz``: outputs of the original PUe and KM code.

Used by ``test_pue.py`` and ``tests/test_prior.py``. Run it with the Python environment of
the original iFPNU experiments, passing the iFPNU directory of the research code::

    python tests/models/data/make_pue_reference.py /path/to/pu_monorepo/iFPNU

It builds a synthetic dataset and makes PU labels with the original ``create_pu_data``. For
each case it saves the KM estimates (``km_default`` on the unlabeled and the labeled samples)
and, after ``seed(i)`` as in ``main_PUe.py``, the experiments' ``PUe`` wrapper's prior,
fitted weights and test probabilities. As in the experiments, the features and the PU labels
are float32. Nothing runs on import, so pytest can collect this file safely.
"""

import sys
from pathlib import Path

# Case name -> (labeling mechanism, fraction of positives hidden, seed).
CASES = {
    "scar": ("SCAR", 0.5, 0),
    "sar": ("SAR", 0.75, 3),
}


def main(ifpnu_dir: Path) -> None:
    sys.path.insert(0, str(ifpnu_dir))
    import warnings

    warnings.filterwarnings("ignore")

    import cvxopt
    import numpy as np
    import torch
    from data_loaders.pu_data import create_pu_data
    from models.helper_files import km
    from models.PUe import PUe, seed
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, weights=[0.75], random_state=0
    )
    X, y = X.astype("float32"), y.astype("float32")  # as the original CSV loader does
    train, test = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0, stratify=y)
    X_train, X_test, y_train = X[train], X[test], y[train]

    arrays = {"X_train": X_train, "X_test": X_test, "y_train": y_train}
    for name, (mechanism, ratio, run) in CASES.items():
        s = create_pu_data(X_train, y_train.copy(), ratio, mechanism, None).flatten()
        arrays[f"{name}_s"] = s.astype("int8")
        arrays[f"{name}_seed"] = run
        arrays[f"{name}_km"] = np.array(km.km_default(X_train[s == 0], X_train[s == 1]))

        seed(run)
        model = PUe()
        model.fit(X_train, s)
        km1 = km.km_default(X_train[s == 0], X_train[s == 1])[1]
        arrays[f"{name}_prior"] = (1 - np.mean(s)) * km1 + np.mean(s)
        arrays[f"{name}_proba"] = model.predict_proba(X_test)
        for part, module in (("e", model.e), ("clf", model.clf)):
            arrays[f"{name}_{part}_weight"] = module.linear.weight.detach().numpy()
            arrays[f"{name}_{part}_bias"] = module.linear.bias.detach().numpy()
    arrays["torch_version"] = torch.__version__
    arrays["cvxopt_version"] = cvxopt.__version__
    np.savez_compressed(Path(__file__).with_name("pue_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
