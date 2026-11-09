# Installation

```bash
pip install pypu-learn
```

The package is imported as `pypu` and needs Python 3.11 or newer. (The PyPI name `pypu`
belongs to an unrelated project, hence `pypu-learn`.) It depends on NumPy, SciPy,
scikit-learn and LightGBM.

## Optional extras

Some models need a library that is not installed by default:

| Extra | Installs | Needed by |
| --- | --- | --- |
| `xgboost` | XGBoost | {class}`~pypu.models.UPUBoost`, {class}`~pypu.models.NNPUBoost`, {class}`~pypu.models.ImbalancedNNPUBoost` |
| `torch` | PyTorch, jenkspy | {class}`~pypu.models.LBE`, {class}`~pypu.models.PUe`, {class}`~pypu.models.ALMPU` |
| `cvxopt` | CVXOPT | {class}`~pypu.prior.KMPrior`, and {class}`~pypu.models.PUe` when no prior is given |

```bash
pip install "pypu-learn[xgboost]"
pip install "pypu-learn[torch]"
pip install "pypu-learn[cvxopt]"
pip install "pypu-learn[xgboost,torch,cvxopt]"  # all of them
```

A model whose extra is missing raises an `ImportError` that names the extra to install.

[CVXOPT](https://cvxopt.org) is licensed under the GPL-3.0; PyPU itself stays under the BSD
3-Clause License.

The [examples](auto_examples/index.rst) also need matplotlib: `pip install matplotlib`.

## Platform notes

**macOS.** The LightGBM-based models need the OpenMP runtime, which LightGBM's wheels do
not include:

```bash
brew install libomp
```

PyTorch bundles its own OpenMP runtime, which conflicts with the one LightGBM uses. PyPU
loads LightGBM first and runs its PyTorch models on one thread, so they work together, but
importing `torch` yourself before using a LightGBM-based model can crash LightGBM.

**Linux.** PyPI's PyTorch wheels include CUDA. For the smaller CPU build, install PyTorch
first as described on [pytorch.org](https://pytorch.org), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "pypu-learn[torch]"
```

## Development version

To try features that are not released yet, install from GitHub:

```bash
pip install git+https://github.com/izavits/PyPU.git
```

To work on PyPU itself, see [Contributing](contributing.md).
