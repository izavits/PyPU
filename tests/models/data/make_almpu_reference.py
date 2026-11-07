"""Generate ``almpu_reference.npz``: outputs of the original ALM-PU code, for ``test_almpu.py``.

Run it with the Python environment of the original ALM-PU benchmark, passing the ALM-PU
directory of the research code::

    python tests/models/data/make_almpu_reference.py /path/to/pu_monorepo/ALM-PU

It builds a synthetic dataset and makes PU labels with the benchmark's ``create_pu_data``.
For each case it runs the benchmark's steps on one thread, as its workers did: ``set_seed``,
``build_sets``, ``train_phase1`` and ``train``. It saves the data, the trends and the break
that ``jenkspy`` received and returned, the pseudo-labels and the test probabilities of the
positive class. Nothing runs on import, so pytest can collect this file safely.
"""

import sys
from pathlib import Path

# Case name -> (labeling mechanism, fraction of positives hidden, seed, method arguments).
# "experiments" uses the benchmark's settings; the others are short, for quick tests.
CASES = {
    "experiments": ("SCAR", 0.5, 2, {}),
    "short": ("SAR", 0.75, 0, {"warming_epochs": 3, "ft_epochs": 4, "eval_step": 10}),
    "short_no_ema": (
        "SCAR",
        0.25,
        1,
        {"warming_epochs": 3, "ft_epochs": 3, "eval_step": 10, "use_ema": False, "batch_size": 32},
    ),
}


def main(almpu_dir: Path) -> None:
    sys.path.insert(0, str(almpu_dir))
    import argparse
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    import jenkspy
    import numpy as np
    import tabular
    import torch
    from pu_setting import create_pu_data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    torch.set_num_threads(1)  # as the benchmark's workers

    X, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, weights=[0.75], random_state=0
    )
    X, y = X.astype("float32"), y.astype("float32").reshape(-1, 1)  # as the CSV loader
    train, test = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0, stratify=y)
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    captured = {}
    original_breaks = jenkspy.jenks_breaks

    def capturing_breaks(values, n_classes):
        captured["trends"] = np.array(values)
        captured["intervals"] = np.array(original_breaks(values, n_classes=n_classes))
        return captured["intervals"].tolist()

    tabular.jenkspy.jenks_breaks = capturing_breaks
    to_repo = lambda labels: np.where(labels.ravel() == 1, 0, 1)  # noqa: E731

    arrays = {"X_train": X_train, "X_test": X_test, "y_train": y_train.ravel()}
    for name, (mechanism, ratio, seed, settings) in CASES.items():
        parser = argparse.ArgumentParser()
        tabular.add_method_args(parser)
        args = parser.parse_args([])
        for key, value in settings.items():
            setattr(args, key, value)
        args.device = torch.device("cpu")

        s = create_pu_data(X_train, y_train, ratio, mechanism, None).ravel()
        labeled = s == 1
        tabular.set_seed(seed)
        labeled_set, unlabeled_set, test_set, _info = tabular.build_sets(
            X_train[labeled],
            X_train[~labeled],
            to_repo(y_train[~labeled]),
            X_test,
            to_repo(y_test),
        )
        pseudo_targets, _ = tabular.train_phase1(args, labeled_set, unlabeled_set, 8)
        _, scores = tabular.train(args, labeled_set, unlabeled_set, test_set, pseudo_targets, 8)

        arrays[f"{name}_s"] = s.astype("int8")
        arrays[f"{name}_seed"] = seed
        arrays[f"{name}_trends"] = captured["trends"]
        arrays[f"{name}_intervals"] = captured["intervals"]
        arrays[f"{name}_pseudo_targets"] = pseudo_targets  # repo convention: 0 = positive
        arrays[f"{name}_scores"] = scores  # P(positive) on X_test
    arrays["torch_version"] = torch.__version__
    np.savez_compressed(Path(__file__).with_name("almpu_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
