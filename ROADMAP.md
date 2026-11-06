# Roadmap

PyPU gathers PU learning methods that currently live in separate research repositories and
exposes them through one scikit-learn style interface (`pypu.base.BasePUClassifier`). This
file tracks what is planned, where the code comes from, and what to watch for when porting.

## Milestones

| Version | Contents |
| --- | --- |
| 0.0.x | Scaffold: packaging, base class, `NaivePU`, CI, release workflow, project documents. Trial release to TestPyPI. |
| 0.1.0 | `pypu.datasets` (PU data from labeled datasets), `pypu.metrics`, class-prior estimation (Elkan–Noto, BBE), `IFPU`, `IFPNU`, `SAREM`, `ElkanNoto`, `PUBagging`. |
| 0.2.0 | `PUHellingerTree` and `PUHellingerForest`, rewritten in NumPy. |
| 0.3.0 | Optional PyTorch extra: `NNPU`/`UPU` networks, `LBE`, `PUe` (with the KM prior estimator), `ALMPU`. |
| Later | `SpyEM`, `RankPruning`, loaders for the benchmark datasets, a documentation site, a benchmark table comparing all methods. |

## Methods and their sources

Paths are relative to `~/workspace`. The SAREM, LBE, PUe and `pu_data` files in `pu_learning/`
are identical to the copies in `pu_monorepo/iFPNU/`, so the monorepo is the single source.

| Model | Source | Dependencies |
| --- | --- | --- |
| `IFPU` | `pu_monorepo/iFPNU/pu/fpu_xgb.py` (`PUFLLossNonNegative`), configured in `main.py` | lightgbm |
| `IFPNU` | `pu_monorepo/iFPNU/main_2step_ad.py` (reliable negatives) and `pu/fpu_xgb.py` (`pnu_focal_loss`) | lightgbm, an anomaly detector (scikit-learn's by default, PyOD optional) |
| `ALMPU` | `pu_monorepo/ALM-PU/tabular.py` | torch, jenkspy |
| `PUHellingerTree`, `PUHellingerForest` | `pu_monorepo/PU_Hellinger_Trees/pu_tree_simplified/` | none after the rewrite |
| `SAREM` | `pu_monorepo/iFPNU/models/SAREM.py`, `models/helper_files/sarpu/` | scikit-learn |
| `LBE` | `pu_monorepo/iFPNU/models/LBE.py`, `models/helper_files/lbe/` | torch |
| `PUe` | `pu_monorepo/iFPNU/models/PUe.py`, `models/helper_files/km.py` | torch, cvxopt |
| `ElkanNoto`, `PUBagging`, `SpyEM`, `RankPruning` | `pu_monorepo/PU_Hellinger_Trees/pos_noisyneg/` | scikit-learn (SpyEM: pomegranate) |
| Focal nnPU losses | `pu_monorepo/iFPNU/pu/losses.py` | torch |

## Data utilities (`pypu.datasets`, 0.1.0): done

Ported:

- Uniform (SCAR) selection of the labeled positives: `label_scar`.
- KNN-biased (SAR) selection with real nearest neighbours and a `bias="near" | "far"`
  option: `label_sar_knn`. `bias="near"` is the mechanism of the old code (positives drawn
  for hiding with probability proportional to their distance from the negatives).
- `legacy=True` on both reproduces `create_pu_data` exactly, KNN shortcut included (see
  porting note 1); the tests compare against a copy of the old function. The old `ratio` was
  the fraction of positives *hidden*, so `ratio=r` corresponds to `label_frequency=1 - r`.
- Parameterisation by pollution (share of positives among the unlabeled), from
  `make_noisy_negatives`: `label_frequency_from_pollution`.

New:

- `label_sar_propensity`: labeling probability is a logistic function of selected features,
  so the true propensity is known (returned with `return_propensity=True`).
- `label_sar_cluster`: labeled positives come only from some k-means clusters of the
  positives.
- `label_sar_confidence`: easy (or hard) positives, by out-of-fold classifier confidence, are
  more likely to be labeled.
- `make_case_control`: every positive kept, negatives subsampled so the unlabeled set has the
  class prior of the data.
- `make_pu_split`: stratified train/test split with PU labels on the training part only.
- `pu_summary`: true prior, label frequency and pollution of the unlabeled set.

## Metrics (`pypu.metrics`, 0.1.0): done

Ported from `pu/evaluation.py` and `train_and_validation.py`, and checked against copies of
the old functions: R-precision and precision/recall at k (ties at the cut-off now count in
proportion; the old code broke them by `argsort` order), F1 at the prior threshold
(`average="binary"`, `"macro"` or `None` give `f1_pos`, `f1_macro` and `[f1_neg, f1_pos]`),
and the best F1 over thresholds.

New: Lee and Liu's criterion, and estimates of recall, precision, F1 and ROC AUC from PU
labels and the class prior (Jain, White and Radivojac, AAAI 2017), usable as scikit-learn
scorers for model selection without true labels.

## Class-prior estimation (`pypu.prior`, 0.1.0): done

Implemented from the papers (no code to port): `ElkanNotoPrior` (Elkan and Noto, KDD 2008;
`e1`, `e2`, `e3`) and `BBEPrior` (Best Bin Estimation; Garg et al., NeurIPS 2021). Both use
out-of-fold scores of a classifier separating labeled from unlabeled samples. On synthetic
data BBE stays close to the true prior as the classes overlap, while Elkan-Noto's `e1`/`e2`
overestimate it, as expected. Still to come: the KM estimator (with `PUe`, 0.3.0), and
possibly TIcE.

## Classic baselines (`ElkanNoto`, `PUBagging`, 0.1.0): done

Neither was used by the current experiment scripts; the older `expsetup.py` of the Hellinger
trees ran both with linear SVMs.

- `ElkanNoto` is implemented from the paper, not ported. The copy in `pos_noisyneg/` (from
  pulearn) estimated `c` on a single hold-out split drawn from the global NumPy RNG; PyPU uses
  out-of-fold scores, as `ElkanNotoPrior` does, or `c = P(s=1) / prior` when `prior` is
  given. Its `WeightedElkanotoPuClassifier` was not the paper's weighted method: it scaled
  `g(x)` by an estimate of `P(y=1)` computed on the samples being predicted, so a sample's
  score depended on the rest of the batch, and the ranking was that of `g(x)`. PyPU's
  `weighted=True` trains a classifier on weighted copies of the unlabeled samples.
- `PUBagging` reproduces the bags of `BaggingPuClassifier` (same seeds, the same order of
  random draws, sample weights when the classifier accepts them). Against outputs of the
  original under scikit-learn 1.0 (`tests/models/data/bagging_reference.npz`), the
  experiments' setting (linear SVMs, balanced bags), shallow trees and k-NN match exactly.
  Fully grown trees and logistic regression differ slightly because scikit-learn changed
  those models since, not the bags. The original's out-of-bag estimates redrew the bags
  without first setting the classifier's `random_state`, so for classifiers that have one
  they came from other bags than the training ones; PyPU's `oob_proba_` uses the actual bags.
- Defaults: `n_estimators=100` and bags as large as the labeled set (the experiments'
  setting) instead of the original 10 bags of all unlabeled samples; decision trees as the
  base classifier, as in the original.

## Porting notes

1. **SAR-KNN selection (resolved).** `pu_data.py` (lines 44-45) averages the distance to the
   first `n_neighbors` columns of an unsorted distance matrix, i.e. to the first k negatives
   in row order rather than the k nearest. It also builds the full n×n distance matrix. The
   copy in `pos_noisyneg/utils.py` sorts the distances first and is correct. PyPU uses real
   nearest neighbours; `legacy=True` reproduces the old labels.
2. **iFPU (resolved).** `IFPU` ports `PUFLLossNonNegative` as run in the experiments.
   - The commented-out clip (`fpu_xgb.py` lines 48-49) acts on `(1 - pi) * grad_n`, which is
     never positive, so enabling it would zero the negative-class term of every unlabeled
     sample. It is not an nnPU correction and is not ported. Unlabeled samples get
     `pi * l+ + (1 - pi) * l-`: soft positives with probability `pi`.
   - `main.py` passes `num_boost_round=10` next to `n_estimators=100`; LightGBM treats it as
     an alias and builds 10 trees. The saved results therefore use 10 trees, and so does
     `IFPU` by default.
   - PyPU (`label_scar`/`label_sar_knn` with `legacy=True` and `random_state=123`, the split
     of `CSVDataset.get_splits`, then `IFPU`) reproduces every saved `scores_*.npz` in
     `iFPNU/output/*_iFPU_lgb` exactly. `tests/models/data/ifpu_reference.npz` holds outputs
     of the original code on synthetic data for the test suite.
   - TODO: add the iFPU and iFPNU papers' references to the docstrings and the README once
     available.
3. **Hellinger trees.** The Cython code is a fork of scikit-learn ~0.24 tree internals: it
   imports `numpy.math` (removed in NumPy 2) and private scikit-learn modules, and only builds
   under Python 3.8. It will be rewritten in NumPy (the split criterion is short) and checked
   against outputs of the old implementation. A compiled version can follow if speed requires.
4. **Licensing.** Upstream code under MIT/BSD (PU_Hellinger_Trees, SAR-PU, pulearn,
   rankpruning) can be ported with attribution in `THIRD_PARTY_NOTICES.md`. No license was
   found for the ALM-PU, LBE, PUe, OptIForest and KM-MPE code: check, or reimplement from
   the papers. `IFPNU` defaults to scikit-learn's `IsolationForest` and accepts any PyOD
   detector instead of vendoring OptIForest.
5. **Label conventions.** PyPU uses `s = 1` for labeled positives and `s = 0` for unlabeled.
   ALM-PU uses positive = 0 internally, and iFPNU's PNU step uses -1 for unlabeled; convert
   at the boundary.
6. **iFPNU (resolved).** `IFPNU` ports the two-step method as run for the saved results in
   `iFPNU/output_2step/*_iFPNU_lgb` (DIF, EIF and OptIForest as detectors).
   - First step: the unlabeled samples among the `int(pi * n)` most anomalous training
     samples (labeled positives count towards that number) stay unlabeled; the other
     unlabeled samples become reliable negatives.
   - Second step: `pnu_focal_loss` with LightGBM, 100 rounds. Its non-negative correction
     applies the unlabeled term only while the mean negative-class focal loss over the
     unlabeled samples is positive, which it always is, so the objective is a weighted focal
     loss: positives `pi * l+`, reliable negatives `eta * (1 - pi) * l-`, unlabeled samples
     `(1 - eta) * l-`. With `eta = 0.5` the first step changes little: across the 840 saved
     runs common to the three detectors, the median ROC AUC difference between OptIForest and
     DIF is 0.004.
   - `nnpnu_focal_loss`, a non-negative PNU risk, was only tried (commented out) and is not
     ported; decided with the author on 2026-11-06.
   - The saved runs' detectors were not seeded (DIF uses PyTorch, EIF and OptIForest the
     global NumPy RNG), so their scores cannot be regenerated. With PyOD's `IForest` seeded
     as the detector, the original two steps and `IFPNU` give identical PNU labels and test
     scores (`tests/models/data/ifpnu_reference.npz`), also with the default
     `IsolationForest`.

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
- Core dependencies stay small: numpy, scipy, scikit-learn and lightgbm (since `IFPU`).
  LightGBM's macOS wheels need Homebrew's `libomp`; models are imported lazily, so this only
  affects the LightGBM-based models.
  PyTorch, XGBoost and PyOD are optional extras.
