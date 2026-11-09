# Roadmap

PyPU gathers PU learning methods that currently live in separate research repositories and
exposes them through one scikit-learn style interface (`pypu.base.BasePUClassifier`). This
file tracks what is planned, where the code comes from, and what to watch for when porting.

## Milestones

| Version | Contents |
| --- | --- |
| 0.0.x | Scaffold: packaging, base class, `NaivePU`, CI, release workflow, project documents. Trial release to TestPyPI. |
| 0.1.0 (released 2026-11-06) | `pypu.datasets` (PU data from labeled datasets), `pypu.metrics`, class-prior estimation (Elkan–Noto, BBE), `IFPU`, `IFPNU`, `SAREM`, `ElkanNoto`, `PUBagging`. |
| 0.2.0 (released 2026-11-07) | `PUHellingerTree` and `PUHellingerForest`, rewritten in NumPy. |
| 0.3.0 (released 2026-11-07) | Optional XGBoost extra: `UPUBoost`, `NNPUBoost`, `ImbalancedNNPUBoost`, the boosting PU risks of the Hellinger trees experiments. |
| 0.4.0 (released 2026-11-07) | Optional PyTorch extra: `LBE`, `PUe`, `ALMPU`; the KM prior estimator (`KMPrior`, optional cvxopt extra). uPU and nnPU are covered by the boosting models of 0.3.0, so no network versions are planned. |
| 0.5.0 (released 2026-11-08) | Classic methods of Bekker and Davis's survey, after the notebooks of github.com/hkiyomaru/pu-learning (MIT): `RebalancedPU` (Elkan's rebalancing), `PropensityWeightedPU` (risk minimization with a known propensity), `BiasedSVM`, `SpyPU` (spy two-step, optional S-EM refinement), `PEBL` (iterative SVM after spy or 1-DNF negatives). |
| 0.6.0 (released 2026-11-08) | `RankPruning`, from the PU Hellinger trees code (the rankpruning package, MIT). |
| 0.6.1 (released 2026-11-08) | `examples/`: getting-started examples, one example per model, and a comparison of all the models on one dataset. |
| 0.6.2 (released 2026-11-09) | The documentation site, [pypu.readthedocs.io](https://pypu.readthedocs.io): a user guide, the examples gallery and the API reference, built by Read the Docs. |
| Later | Loaders for the benchmark datasets, a benchmark table comparing all methods on many datasets. |

## Methods and their sources

Paths are relative to `~/workspace`. The SAREM, LBE, PUe and `pu_data` files in `pu_learning/`
are identical to the copies in `pu_monorepo/iFPNU/`, so the monorepo is the single source.

| Model | Source | Dependencies |
| --- | --- | --- |
| `IFPU` | `pu_monorepo/iFPNU/pu/fpu_xgb.py` (`PUFLLossNonNegative`), configured in `main.py` | lightgbm |
| `IFPNU` | `pu_monorepo/iFPNU/main_2step_ad.py` (reliable negatives) and `pu/fpu_xgb.py` (`pnu_focal_loss`) | lightgbm, an anomaly detector (scikit-learn's by default, PyOD optional) |
| `ALMPU` | `pu_monorepo/ALM-PU/tabular.py` | torch, jenkspy |
| `PUHellingerTree`, `PUHellingerForest` | `pu_monorepo/PU_Hellinger_Trees/pu_tree_simplified/` | none after the rewrite |
| `UPUBoost`, `NNPUBoost`, `ImbalancedNNPUBoost` | `pu_monorepo/PU_Hellinger_Trees/ecsmodels/methodologies/cs_boost.py` (`CSBoost`, objectives `puwce`, `nnpu`, `imbnnpu`), configured in `uPU_clean.py`, `nnPU_clean.py`, `imbalancedNNPU_clean.py` | xgboost (optional extra) |
| `SAREM` | `pu_monorepo/iFPNU/models/SAREM.py`, `models/helper_files/sarpu/` | scikit-learn |
| `LBE` | `pu_monorepo/iFPNU/models/LBE.py`, `models/helper_files/lbe/` | torch |
| `PUe` | `pu_monorepo/iFPNU/models/PUe.py`, `models/helper_files/km.py` | torch, cvxopt |
| `ElkanNoto`, `PUBagging`, `RankPruning` | `pu_monorepo/PU_Hellinger_Trees/pos_noisyneg/` | scikit-learn |
| `RebalancedPU`, `PropensityWeightedPU`, `BiasedSVM`, `SpyPU`, `PEBL` | The papers, checked against the notebooks of github.com/hkiyomaru/pu-learning (MIT); `SpyPU` replaces the planned `SpyEM` of `pos_noisyneg/spyEM.py` (which needed pomegranate and imbalanced-learn) | scikit-learn |
| Focal nnPU losses (not planned: network nnPU dropped) | `pu_monorepo/iFPNU/pu/losses.py` | torch |

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
overestimate it, as expected. `KMPrior` (KM1 and KM2; Ramaswamy, Scott and Tewari, ICML
2016) came with `PUe` in 0.4.0, using the authors' code (porting note 10). Possibly later:
TIcE.

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
3. **Hellinger trees (resolved).** The Cython code is a fork of scikit-learn 1.0.2's tree
   internals: it imports `numpy.math` (removed in NumPy 2) and private scikit-learn modules,
   and only builds under Python 3.8. `PUHellingerTree` and `PUHellingerForest` rewrite it in
   NumPy.
   - The criterion estimates the positives of a child as `min(n, L / c)`, with `L` its
     (weighted) labeled count and `c = P(s = 1) / prior` computed once on all training
     samples, and scores a split by the squared Hellinger distance between those estimated
     positives and negatives over the two children. Nodes without labeled samples become
     leaves; nodes of labeled samples only keep splitting (every split scores 1).
   - Many splits tie under this criterion, and the first one found wins, so the rewrite
     replicates scikit-learn's best splitter exactly: the `our_rand_r` xorshift draws of
     features, the bookkeeping of constant features across nodes, the float32
     `FEATURE_THRESHOLD`, float64 midpoint thresholds, and the depth-first order of nodes.
     Against outputs of the original under Python 3.8 (`tests/models/data/hellinger_reference.npz`:
     six trees and two forests, including constant, duplicated and discrete features) the
     trees are identical, node for node, and so are the predictions.
   - `predict_proba` is the fraction of labeled samples in the leaf, as the experiments used;
     `predict_pu_proba` is the original's corrected probability. The forest's `max_samples`
     defaults to the number of unlabeled samples, the experiments' setting (the original
     class defaulted to the number of samples).
   - A forest of 100 trees on 5000 samples fits in about 2 s, so no compiled version is
     needed for now.
   - The saved results (`output/results_*.csv`) cannot be regenerated: `expsetup_clean.py`
     shuffles each dataset with an unseeded `np.random.shuffle`, and saves metrics only. Its
     threshold-tuning tree is fitted with `p_y = mean(noisy_y)`, which makes `1 / c = 1`, so
     that tree has no PU correction; the final tree and the forest get the true prior.
4. **Licensing.** Upstream code under MIT/BSD (PU_Hellinger_Trees, SAR-PU, pulearn,
   rankpruning) can be ported with attribution in `THIRD_PARTY_NOTICES.md`. `IFPNU` defaults
   to scikit-learn's `IsolationForest` and accepts any PyOD detector instead of vendoring
   OptIForest. For 0.4.0 the author decided (2026-11-07) to port the implementations in the
   local repositories rather than reimplement the papers. Checked the same day, two of them
   contain other people's code without a license:
   - `iFPNU/models/helper_files/km.py` (used by `PUe`) is the kernel mixture proportion
     estimation code of Ramaswamy, Scott and Tewari, downloaded from C. Scott's website and
     lightly modified. The author decided to port it unchanged, credited in
     `THIRD_PARTY_NOTICES.md`, with its solver CVXOPT (GPL-3.0) as the optional extra
     `pypu-learn[cvxopt]`, not distributed with PyPU (2026-11-07).
   - `ALM-PU/` is a modified clone of the authors' repository (github.com/JiazheWei/ALM-PU),
     which has no license; its EMA model and cosine schedule come from FixMatch-pytorch
     (MIT). The author decided to port its tabular code as run, crediting both in
     `THIRD_PARTY_NOTICES.md`, with FixMatch-pytorch's license text (2026-11-07).
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
7. **SAREM (resolved).** `SAREM` ports SAR-PU's `pu_learn_sar_em` (MIT, KU Leuven DTAI), the
   function the experiments' `SAREM` wrapper called with all features as propensity
   features. The same sequence of liblinear logistic regressions, clipping, log-likelihood
   and convergence test is kept.
   - Against outputs of the original (`tests/models/data/sarem_reference.npz`), PyPU runs
     the same number of iterations and matches to about 1e-8. The gap comes from the
     original's first sample weights, computed in float32 because NumPy 1 kept float32 PU
     labels times a float64 scalar in float32; with those weights the results agree to
     1e-15. PyPU keeps float64.
   - Rebuilding the 840 saved runs (`iFPNU/output_SAREM`, 84 configurations × 10 splits)
     with PyPU only (legacy labels with `random_state=123`, the experiments' splits): in 75
     configurations the rebuilt scores give exactly the saved ROC AUC; in all of them the
     largest ROC AUC difference is 9e-4 and the largest score difference 2e-3. The original code
     regenerates the saved scores exactly, and a single liblinear fit agrees across
     scikit-learn 1.2 and 1.9 to the last bits (6e-15 in the intercept). On datasets where EM
     converges slowly (Satellite, Piechart, Pizzacutter, Segment), hundreds of iterations
     amplify such differences, and EM can stop at a different iteration.
   - SAR-PU's model without propensity features (`NoFeaturesModel`) divided by the number of
     samples instead of the sum of the weights, so it estimated `P(s = 1)` instead of the
     label frequency. The experiments did not use it; PyPU uses the weighted M-step
     estimate. Even so, EM's estimate of a constant propensity is too high when the classes
     overlap (on synthetic data, 0.89 for a true 0.4), so the docstring points to
     `pypu.prior`.
   - `predict_propensity` and `propensity_features` are new. On synthetic SAR data whose
     propensity depends on one feature, restricting the propensity to it recovers the true
     propensity (correlation 0.99); with all features it does not (0.67).
8. **Boosting PU risks (resolved).** `UPUBoost`, `NNPUBoost` and `ImbalancedNNPUBoost` port
   the `puwce`, `nnpu` and `imbnnpu` objectives of `CSBoost`, with the experiments' XGBoost
   settings (100 rounds, `eta` 0.3, `max_depth` 6, `reg_lambda` 0, `tree_method="exact"`,
   `c = mean(s) / prior`, `oversampling=0.5`).
   - `puwce` is the unbiased PU risk. `nnpu` and `imbnnpu` take the non-negative max per
     sample, of a coefficient that is always negative for labeled samples, so the correction
     always removes their negative term: `nnpu` is labeled `l+ / c` and unlabeled `l-`;
     `imbnnpu` is labeled `rho l+` and unlabeled `(1 - rho) / (1 - prior) l-`. Neither is the
     published nnPU (Kiryo et al.) or imbalanced nnPU (Su et al.), which clip the total
     negative risk. Ported as run, documented in the docstrings; decided with the author on
     2026-11-07.
   - The original ran under NumPy 1, where float32 arrays times float64 scalars stay
     float32, so most of its gradient arithmetic was float32; the port reproduces those
     dtypes. In the original environment (XGBoost 2.1.3, x86), PyPU's objectives give the
     same trees bit for bit. On arm64 the float32 sigmoid differs in the last bit (scores
     within 2e-6); XGBoost 3.4 changes them by up to 3e-3. uPU's risk is unbounded below,
     so its scores reach about 100 and amplify those differences to about 2, with the same
     test ranking (ROC AUC equal to 4 decimals). The tests compare the gradients at fixed
     scores (`tests/models/data/risk_boost_reference.npz`) and the scores within 1e-2.
   - The saved results (`output_nnpu`, `output_upu`, `output_imbnnpu`) cannot be
     regenerated: the scripts shuffle each dataset with an unseeded `np.random.shuffle` and
     save metrics only. They also compute `prior` and `mean(s)` on the whole dataset, test
     part included; PyPU uses the training data.

9. **LBE (resolved).** `LBE` ports the experiments' wrapper (`iFPNU/models/LBE.py`) and
   `lbe_train` (`models/helper_files/lbe/LBE.py`).
   - The wrapper passes `kind='LR'` and `epochs=250`, which `lbe_train` ignores: it always
     uses logistic models, 1000 pre-training steps, and 100 EM iterations of 100 Adam steps,
     all with learning rate 0.01 on the full batch. Those are PyPU's defaults. The helper's
     MLP option is not ported (its zero-initialized propensity network could only learn its
     bias), nor are the gradients `loss` computes by hand and never uses.
   - The expected log-likelihood gives unlabeled samples a `log(1 - eta)` term for `y = 0`
     too, so the "propensity" `eta` is fitted as `P(s = 1 | x)`, a logistic regression of `s`
     on `x` (the logits agree with scikit-learn's to a correlation of 0.9999999), and the
     E-step overestimates `P(y = 1 | x, s = 0)`. Ported as run and documented.
   - Under the same PyTorch, PyPU fits the original's weights to the last bit. Predictions
     are computed from them in float64 with NumPy: in float32, as the original did, the
     rounding of the matrix product depends on how many rows are multiplied at once on some
     platforms, which failed scikit-learn's subset-invariance check on Linux. The
     reference (`tests/models/data/lbe_reference.npz`) comes from PyTorch 2.8, the
     experiments' version; under 2.15 the original itself moves by up to 1.5e-4, so the tests
     compare whole fits within 2e-3, and one E-step and the M-step gradients at fixed weights
     within 1e-5.
   - `main_LBE.py` did not seed PyTorch, so each saved run's initial weights depend on the
     runs before it in the same process; the saved scores are not regenerated here.
   - On macOS, PyTorch's bundled OpenMP runtime and Homebrew's (used by LightGBM and
     XGBoost) crash each other in one process: LightGBM fails if PyTorch was loaded first,
     and PyTorch fails when it runs on several threads. `pypu.models._torch` imports
     LightGBM before PyTorch, and the PyTorch models run on one thread, which also makes
     their results independent of the number of cores.

10. **PUe and KM (resolved).** `PUe` ports the experiments' wrapper (`iFPNU/models/PUe.py`),
    and `KMPrior` the KM code it calls (`models/helper_files/km.py`, by Ramaswamy, Scott and
    Tewari; see note 4).
    - As run: the prior is the KM1 estimate combined with the labeled fraction of `s`
      (computed in float32, as the float32 labels made it); the propensity model and the
      classifier are logistic regressions trained for 20 full-batch Adam steps each, with
      learning rate 0.001, from `nn.Linear`'s initial weights after `seed(i)`. Those are the
      defaults, and `random_state=i` reproduces `seed(i)`.
    - 20 steps at that learning rate barely move the models: the scores stay close to those
      of the random initial weights (on synthetic data the ROC AUC ranges from 0.24 to 0.83
      over seeds; across the saved settings the mean is 0.54). Decided with the author to keep
      these defaults and document them (2026-11-07); `n_steps=2000, learning_rate=0.01` gives
      a trained model.
    - The classifier's loss sums its unlabeled term while the labeled weights sum to one, so
      unlabeled samples weigh `n_u` times more than in a risk estimate, and the negative term
      is not clipped. Ported as run, documented.
    - Against the reference (`tests/models/data/pue_reference.npz`, PyTorch 2.8 and cvxopt
      1.3.3), the KM estimates, the prior and both models' weights agree to the last bit,
      under PyTorch 2.15 too; the test probabilities, computed in float64 as for `LBE`, agree
      to float32 precision.
    - `main_PUe.py` seeded each run, so the saved scores (`iFPNU/output_PUe`, 84 settings ×
      10 runs) can be regenerated with PyPU alone (legacy labels with `random_state=123`, the
      experiments' splits, `PUe(random_state=i)`). Checked on 2026-11-07 for all 84
      settings: every run of the Cardio, Climate and Letter settings (180 runs, predictions
      computed in float32 as the original did) and the first run of each of the other 66
      settings (with PyPU's float64 predictions). Every regenerated score agrees with the saved
      one within float32 rounding (at most 1.7e-7), and every ROC AUC is the same (to 6e-17).
      All 840 runs would take hours: KM on the 5000-sample datasets is the slow part.
    - KM builds the full kernel matrix and solves about 20 dense quadratic programs of the
      sample size: about a minute per fit on 3500 training samples, on one core.

11. **ALM-PU (resolved).** `ALMPU` ports `ALM-PU/tabular.py` as the benchmark ran it
    (`pu_benchmark.py`: `set_seed(i)`, one thread per worker), with its settings as defaults.
    - Everything random (initial weights, the batches of the two data loaders, dropout) came
      from PyTorch's global generator, so `fit` seeds it with `random_state` inside
      `torch.random.fork_rng` and builds the same data loaders, in the same order: under the
      same PyTorch the trends, pseudo-labels and network are the same to the last bit
      (`tests/models/data/almpu_reference.npz`, from PyTorch 2.14).
    - SGD amplifies rounding differences: under PyTorch 2.15 the original's own test scores in
      the benchmark setting differ from 2.14's by up to 0.99 (phase 1's trends by 0.01, with the
      same pseudo-labels; on CI's Linux runners 4 of 367 pseudo-labels flip), so the tests
      compare short trainings closely and the full one's pseudo-labels loosely.
    - Not ported: the `three_sigma` option (tuned for images, off in the tabular runs) and the
      per-epoch test metrics (logging only). Predictions are computed in float64 from the
      trained network, as for `LBE` and `PUe`. When the trends take fewer than two distinct
      values, jenkspy fails; PyPU raises a clear error.
    - In 159 of the benchmark's 840 runs (52 Piechart, 40 Climate, 24 Pizzacutter, ...; most
      with 75% of the positives hidden) the test ROC AUC is below 0.5. In those runs about 59%
      of the unlabeled samples were pseudo-labeled positive, with precision 0.025, where about
      3% were positive: the true positives' trends fell mostly below the Jenks break, and phase
      2 learned the inverted labels. Documented; `pollution_` shows it.
    - All 840 saved runs (`ALM-PU/output_ALMPU`, 84 settings × 10) regenerated with PyPU alone
      under PyTorch 2.14 (legacy labels with `random_state=123`, the benchmark's splits,
      `ALMPU(random_state=i)`), 2026-11-07: the estimated share of positives among the
      unlabeled samples (`est_prior`, PyPU's `pollution_`) is the same in every run, and so is
      the test ROC AUC when the scores are computed in float32, as the original did. With
      PyPU's float64 predictions the ROC AUC is the same in 769 runs and differs by at most
      2.6e-3 in the other 71, where float32 had rounded several test scores to the same
      value (11 per run in the median) and float64 tells them apart.

12. **Classic methods (0.5.0).** Implemented from the papers and Bekker and Davis's survey,
    and checked against the notebooks of github.com/hkiyomaru/pu-learning (MIT) on their data
    (two Gaussian classes in two dimensions), 2026-11-08:
    - `SpyPU` with the notebook's step-1 classifier (logistic regression) and seed: the same
      spies, threshold and reliable negatives, and the same probabilities. The notebook also
      counted labeled samples below the threshold as negatives (none were, there); PyPU only
      takes unlabeled ones. S-EM's EM refinement (`n_em_iter`) is new; its ΔE model selection
      is not ported.
    - `BiasedSVM` with the notebook's train/validation split as `cv`: the same Lee and Liu
      criterion for every weight and the same chosen weight. The notebook's grid
      `np.arange(0.7, 1.0, 0.05)` also yields 0.9999999999999998, on which SVC fails; PyPU's
      default grid stops at 0.95.
    - `PropensityWeightedPU` on the notebook's SAR data: the same probabilities when the
      weights are computed in float64 (the notebook's were float32) and its one zero-weight
      copy (e(x) = 1) is dropped, as PyPU does. The negative weights make the objective
      non-convex: solved to a tight tolerance, float32 and float64 weights end 5e-3 apart, so
      results depend on the solver; documented.
    - `RebalancedPU` uses the survey's weights, w+ = 1 - c tau and w- = (1 - tau) c. The
      notebook's, w+ = tau / (c tau) and w- = 1 - (1 - tau) / (1 - c tau), differ from them by
      a factor 1 - c and are not used.
    - `PEBL`: the notebook's 1-DNF for continuous features (a threshold per feature near the
      1st percentile of the labeled samples, reliable negatives below every threshold)
      reproduced its thresholds, reliable negatives, 17 iterations and selected SVM, but finds
      no reliable negatives on ordinary tabular data (make_classification with 2 to 20
      features), where some features do not separate the classes. Decided with the author:
      reliable negatives come from spies by default (Spy-SVM; Liu et al., 2003) or from the
      original 1-DNF for binary or count features (`negatives="1dnf"`); the iterative SVM and
      its selection by Lee and Liu's criterion match the notebook's.

13. **RankPruning (resolved).** `RankPruning` ports `pos_noisyneg/rankpruning.py`, which is
    C. G. Northcutt's rankpruning package (MIT) without one unused helper. The experiment
    scripts imported it but never ran it, so the defaults are the package's, in its PU
    learning mode (`fit(..., pulearning=True)`: labeled samples are clean, rho0 = 0); by
    default the package also estimates a noise rate for the labeled samples. `prior` replaces
    `frac_pos2neg` (rho1 = 1 - P(s = 1) / prior). The original shuffled its 3 folds with
    NumPy's global random state; `random_state=i` gives the folds of `np.random.seed(i)`.
    Against the original under the same scikit-learn (`tests/models/data/rank_pruning_reference.npz`,
    with rho1 estimated or given), the noise rate, the estimated prior, the pollution, the
    refitted coefficients and the test probabilities agree to the last bit.

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
  PyTorch (`[torch]`), XGBoost (`[xgboost]`) and PyOD are optional.
