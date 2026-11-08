"""Generate ``rank_pruning_reference.npz``: outputs of the original Rank Pruning code.

Used by ``test_rank_pruning.py``. Pass the directory of the PU Hellinger trees code, whose
``pos_noisyneg/rankpruning.py`` is the rankpruning package of C. G. Northcutt::

    python tests/models/data/make_rank_pruning_reference.py /path/to/PU_Hellinger_Trees

The experiments imported that class but never ran it, so there is no original environment to
match: run this script with PyPU's development environment, and the reference agrees with
PyPU to the last bit. For each case it seeds NumPy's global random state (the original
shuffles its folds with it), fits the original in its PU learning mode and saves the noise
rates, the pruned samples and the test probabilities. Nothing runs on import, so pytest can
collect this file safely.
"""

import sys
from pathlib import Path

# Case name -> (seed, "estimated" or a label frequency to derive frac_pos2neg from).
CASES = {
    "estimated": (0, None),
    "estimated_seed3": (3, None),
    "known_label_frequency": (1, 0.4),
}


def main(trees_dir: Path) -> None:
    sys.path.insert(0, str(trees_dir))
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np
    import sklearn
    from pos_noisyneg.rankpruning import RankPruning
    from sklearn.datasets import make_classification

    from pypu.datasets import label_scar

    X, y = make_classification(
        n_samples=800, n_features=8, n_informative=5, weights=[0.7], random_state=0
    )
    X_train, X_test, y_train = X[:600], X[600:], y[:600]
    s = label_scar(y_train, label_frequency=0.4, random_state=0)

    arrays = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "s": s}
    for name, (seed, label_frequency) in CASES.items():
        frac_pos2neg = None if label_frequency is None else 1 - label_frequency
        model = RankPruning(frac_pos2neg=frac_pos2neg)
        np.random.seed(seed)  # noqa: NPY002 - the original shuffles its folds with it
        model.fit(X_train, s, pulearning=True)
        arrays[f"{name}_seed"] = seed
        arrays[f"{name}_rh1"] = model.rh1
        arrays[f"{name}_py1"] = model.py1
        arrays[f"{name}_pi0"] = model.pi0
        arrays[f"{name}_proba"] = model.predict_proba(X_test)[:, 1]
        arrays[f"{name}_coef"] = model.clf.coef_.ravel()
    arrays["sklearn_version"] = sklearn.__version__
    np.savez_compressed(Path(__file__).with_name("rank_pruning_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
