# Roadmap

PyPU gathers PU learning methods that currently live in separate research repositories and
exposes them through one scikit-learn style interface (`pypu.base.BasePUClassifier`). This
file tracks what is planned, where the code comes from, and what to watch for when porting.

## Milestones

| Version | Contents |
| --- | --- |
| 0.0.x | Scaffold: packaging, base class, `NaivePU`, CI, release workflow, project documents. Trial release to TestPyPI. |
| 0.1.0 | `pypu.datasets` (PU data from labeled datasets), `pypu.metrics`, class-prior estimation (Elkan–Noto), `IFPU`, `IFPNU`, `SAREM`, `ElkanNoto`, `PUBagging`. |
| 0.2.0 | `PUHellingerTree` and `PUHellingerForest`, rewritten in NumPy. |
| 0.3.0 | Optional PyTorch extra: `NNPU`/`UPU` networks, `LBE`, `PUe` (with the KM prior estimator), `ALMPU`. |
| Later | `SpyEM`, `RankPruning`, loaders for the benchmark datasets, a documentation site, a benchmark table comparing all methods. |

## Methods and their sources

Paths are relative to `~/workspace`. The SAREM, LBE, PUe and `pu_data` files in `pu_learning/`
are identical to the copies in `pu_monorepo/iFPNU/`, so the monorepo is the single source.

| Model | Source | Dependencies |
| --- | --- | --- |
| `IFPU` | `pu_monorepo/iFPNU/pu/fpu_xgb.py` (`PUFLLossNonNegative`), configured in `main.py` | lightgbm |
| `IFPNU` | `pu_monorepo/iFPNU/main_2step_ad.py` (reliable negatives) and `pu/fpu_xgb.py` (`nnpnu_focal_loss`) | lightgbm or xgboost, an anomaly detector |
| `ALMPU` | `pu_monorepo/ALM-PU/tabular.py` | torch, jenkspy |
| `PUHellingerTree`, `PUHellingerForest` | `pu_monorepo/PU_Hellinger_Trees/pu_tree_simplified/` | none after the rewrite |
| `SAREM` | `pu_monorepo/iFPNU/models/SAREM.py`, `models/helper_files/sarpu/` | scikit-learn |
| `LBE` | `pu_monorepo/iFPNU/models/LBE.py`, `models/helper_files/lbe/` | torch |
| `PUe` | `pu_monorepo/iFPNU/models/PUe.py`, `models/helper_files/km.py` | torch, cvxopt |
| `ElkanNoto`, `PUBagging`, `SpyEM`, `RankPruning` | `pu_monorepo/PU_Hellinger_Trees/pos_noisyneg/` | scikit-learn (SpyEM: pomegranate) |
| Focal nnPU losses | `pu_monorepo/iFPNU/pu/losses.py` | torch |

## Data utilities (`pypu.datasets`, 0.1.0)

Ported:

- Uniform (SCAR) selection of the labeled positives.
- KNN-biased (SAR) selection, with the neighbour bug fixed (see below) and a
  `bias="far" | "near"` option.
- Parameterisation by `pollution_ratio` (share of hidden positives among the unlabeled), from
  `make_noisy_negatives`.

New:

- SAR by propensity on chosen features: labeling probability is a sigmoid of selected
  features, so the true propensity is known.
- SAR by cluster: labeled positives come only from some k-means clusters of the positives.
- SAR by classifier confidence: easy positives are more likely to be labeled.
- Case-control sampling: labeled positives drawn separately from the unlabeled population.
- `make_pu_split`: stratified train/test split with PU labels on the training part only.
- `pu_summary`: true prior, label frequency and pollution of the unlabeled set.

## Porting notes

1. **SAR-KNN selection.** `pu_data.py` (lines 44-45) averages the distance to the first
   `n_neighbors` columns of an unsorted distance matrix, i.e. to the first k negatives in row
   order rather than the k nearest. It also builds the full n×n distance matrix. The library
   uses real nearest neighbours; a `legacy=True` flag can reproduce published numbers if
   needed.
2. **iFPU objective.** The non-negativity clip in `PUFLLossNonNegative` is commented out
   (`fpu_xgb.py` lines 48-49), so it computes the unbiased risk. The library default must
   match what the iFPU paper used.
3. **Hellinger trees.** The Cython code is a fork of scikit-learn ~0.24 tree internals: it
   imports `numpy.math` (removed in NumPy 2) and private scikit-learn modules, and only builds
   under Python 3.8. It will be rewritten in NumPy (the split criterion is short) and checked
   against outputs of the old implementation. A compiled version can follow if speed requires.
4. **Licensing.** Upstream code under MIT/BSD (PU_Hellinger_Trees, SAR-PU, pulearn,
   rankpruning) can be ported with attribution in `THIRD_PARTY_NOTICES.md`. No license was
   found for the ALM-PU, LBE, PUe, OptIForest and KM-MPE code: check, or reimplement from
   the papers. `IFPNU` will default to scikit-learn's `IsolationForest` and accept any PyOD
   detector instead of vendoring OptIForest.
5. **Label conventions.** PyPU uses `s = 1` for labeled positives and `s = 0` for unlabeled.
   ALM-PU uses positive = 0 internally, and iFPNU's PNU step uses -1 for unlabeled; convert
   at the boundary.

## Validating ports

The old virtual environments still exist. For each ported method, save predictions of the
original implementation on a fixed dataset and seed as test fixtures, and assert that the
PyPU version reproduces them (within a tolerance, documented when they differ on purpose).

## Decisions

- Distribution name `pypu-learn`, import name `pypu`. A PEP 541 request for the abandoned
  `pypu` name can be filed separately; switching later is a one-line change in
  `pyproject.toml`.
- License: BSD-3-Clause.
- Python 3.11 and newer.
- Core dependencies stay small (numpy, scipy, scikit-learn; lightgbm when `IFPU` lands).
  PyTorch, XGBoost and PyOD are optional extras.
