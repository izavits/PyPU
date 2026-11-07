"""Generate ``lbe_reference.npz``: outputs of the original LBE code, for ``test_lbe.py``.

Run it with the Python environment of the original iFPNU experiments, passing the iFPNU
directory of the research code::

    python tests/models/data/make_lbe_reference.py /path/to/pu_monorepo/iFPNU

It builds a synthetic dataset, makes PU labels with the original ``create_pu_data``, seeds
PyTorch, fits the experiments' ``LBE`` wrapper, and saves the data, the test probabilities
and the fitted weights. At the fitted weights it also saves one E-step and the M-step loss
and gradients, which do not depend on the accumulated rounding of a whole fit. As in the
experiments, the features and the PU labels are float32. Nothing runs on import, so pytest
can collect this file safely.
"""

import sys
from pathlib import Path

# Case name -> (labeling mechanism, fraction of positives hidden, torch seed).
CASES = {
    "scar": ("SCAR", 0.5, 0),
    "sar": ("SAR", 0.75, 3),
}


def main(ifpnu_dir: Path) -> None:
    sys.path.insert(0, str(ifpnu_dir))
    import warnings

    warnings.filterwarnings("ignore")

    import numpy as np
    import torch
    from data_loaders.pu_data import create_pu_data
    from models.LBE import LBE
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=600, n_features=8, n_informative=5, weights=[0.75], random_state=0
    )
    X, y = X.astype("float32"), y.astype("float32")  # as the original CSV loader does
    train, test = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0, stratify=y)
    X_train, X_test, y_train = X[train], X[test], y[train]

    arrays = {"X_train": X_train, "X_test": X_test}
    for name, (mechanism, ratio, seed) in CASES.items():
        s = create_pu_data(X_train, y_train.copy(), ratio, mechanism, None).flatten()
        torch.manual_seed(seed)
        model = LBE()
        model.fit(X_train, s)
        arrays[f"{name}_s"] = s.astype("int8")
        arrays[f"{name}_seed"] = seed
        arrays[f"{name}_proba"] = model.predict_proba(X_test)
        arrays[f"{name}_h_weight"] = model.model.h.h[0].weight.detach().numpy()
        arrays[f"{name}_h_bias"] = model.model.h.h[0].bias.detach().numpy()
        arrays[f"{name}_eta_weight"] = model.model.eta.eta[0].weight.detach().numpy()
        arrays[f"{name}_eta_bias"] = model.model.eta.eta[0].bias.detach().numpy()

        lbe = model.model
        X_t, s_t = torch.from_numpy(X_train).float(), torch.from_numpy(s).float()
        posterior = lbe.E_step(X_t, s_t)
        lbe.zero_grad()
        loss = lbe.loss(X_t, s_t, posterior)[0]
        loss.backward()
        arrays[f"{name}_posterior"] = posterior.numpy()
        arrays[f"{name}_loss"] = loss.item()
        arrays[f"{name}_grad_h_weight"] = lbe.h.h[0].weight.grad.numpy()
        arrays[f"{name}_grad_h_bias"] = lbe.h.h[0].bias.grad.numpy()
        arrays[f"{name}_grad_eta_weight"] = lbe.eta.eta[0].weight.grad.numpy()
        arrays[f"{name}_grad_eta_bias"] = lbe.eta.eta[0].bias.grad.numpy()
    arrays["torch_version"] = torch.__version__
    np.savez_compressed(Path(__file__).with_name("lbe_reference.npz"), **arrays)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
