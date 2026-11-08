# Changelog

All notable changes to PyPU are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is `0.y.z`
the public API may change between minor versions; such changes are always listed under
**Changed** or **Removed**.

## [Unreleased]

## [0.6.0] - 2026-11-08

Rank Pruning, which learns from the confident examples of PU data.

### Added

- `pypu.models.RankPruning`: Rank Pruning (Northcutt et al.) in its PU learning mode. It
  estimates the label noise by confident counts on out-of-fold probabilities (or takes the
  class prior), prunes the unlabeled samples most likely to be positive, and refits with the
  labeled samples reweighted. It ports the rankpruning package as copied into the PU
  Hellinger trees code, and gives its results to the last bit; `random_state=i` gives the
  folds it drew after `np.random.seed(i)`. The estimated class prior, the share of positives
  among the unlabeled samples and the pruned samples are attributes.

## [0.5.0] - 2026-11-08

Five classic PU methods of Bekker and Davis's survey, which need only scikit-learn:
rebalancing, propensity weighting, the biased SVM and two two-step methods.

### Added

- Classic methods of Bekker and Davis's survey, checked against the notebooks of
  github.com/hkiyomaru/pu-learning:
  - `pypu.models.RebalancedPU`: Elkan's rebalancing of a classifier of labeled versus
    unlabeled samples, with weights from the label frequency, so that it classifies at a
    target threshold like a classifier of the true labels.
  - `pypu.models.PropensityWeightedPU`: risk minimization on SAR data with a known propensity
    (a number, or a function such as `SAREM.predict_propensity`), labeled samples counting as
    positives with weight `1 / e(x)` and as negatives with weight `1 - 1 / e(x)`.
  - `pypu.models.BiasedSVM`: an SVM on PU labels whose class weights are chosen by
    cross-validation with Lee and Liu's criterion.
  - `pypu.models.SpyPU`: the spy technique, reliable negatives below hidden labeled samples,
    then a classifier, optionally refined by EM as in S-EM.
  - `pypu.models.PEBL`: reliable negatives from spies (Spy-SVM) or from 1-DNF (binary or
    count features), then an iterative SVM, keeping the iteration with the best Lee and Liu
    criterion or the last one.

## [0.4.0] - 2026-11-07

Three neural PU methods from the iFPNU and ALM-PU experiments, ported as they were run, and
the KM estimator of the class prior. They come with the optional extras `pypu-learn[torch]`
and `pypu-learn[cvxopt]`.

### Added

- `pypu.models.LBE`: LBE (Gong et al.), which learns a logistic classifier together with a
  logistic model of the labeling bias by EM. It ports the LBE code of the iFPNU experiments,
  whose settings are its defaults, and fits the original's weights to the last bit under the
  same PyTorch. As in that code, the labeling-bias model is fitted as a classifier of labeled
  versus unlabeled samples; the docstring explains how this departs from the paper.
- `pypu.models.PUe`: PUe (Wang et al.), a logistic classifier trained with normalized inverse
  propensity weights. It ports the PUe code of the iFPNU experiments, whose settings are its
  defaults, fits the original's weights to the last bit and regenerates the experiments'
  saved test scores. Those settings train each model for only 20 Adam steps, which barely
  moves it from its random initial weights; the docstring says so, and longer training is a
  parameter away.
- `pypu.models.ALMPU`: ALM-PU (Wei et al.), an MLP trained with an augmented-Lagrangian
  constraint on the unlabeled samples, then retrained towards pseudo-labels from the trends
  of its predictions. It ports the ALM-PU tabular benchmark code, whose settings are its
  defaults: under the same PyTorch it trains the same network to the last bit, and under the
  benchmark's PyTorch it regenerates all of its saved runs. The pseudo-labels, their trends
  and the estimated share of positives among the unlabeled samples are attributes.
- `pypu.prior.KMPrior`: the KM1 and KM2 class-prior estimators of Ramaswamy, Scott and
  Tewari, using the authors' code as the PUe experiments did, unchanged. `PUe` uses KM1 when
  no prior is given.
- The optional extra `pypu-learn[torch]` for the PyTorch-based models, with jenkspy for
  `ALMPU`. They train on a single CPU thread, so their results do not depend on the number
  of cores, and predict in float64, so that how the samples are batched only affects the last
  bits of their predictions. On macOS, PyPU loads LightGBM before PyTorch, as their OpenMP
  runtimes otherwise crash each other.
- The optional extra `pypu-learn[cvxopt]` for the KM estimator. CVXOPT is licensed under the
  GPL-3.0; PyPU stays under the BSD 3-Clause License.

## [0.3.0] - 2026-11-07

Gradient boosting on PU risks: uPU, and the nnPU and imbalanced nnPU objectives of the
original experiments, with XGBoost as an optional extra.

### Added

- `pypu.models.UPUBoost`: XGBoost trained on the unbiased PU risk (uPU).
- `pypu.models.NNPUBoost` and `pypu.models.ImbalancedNNPUBoost`: XGBoost trained on the
  objectives of the original nnPU and imbalanced nnPU experiments. Those apply the
  non-negative correction per sample, which always removes the labeled samples' negative
  term, so they are weighted cross-entropies; the docstrings say so.
- All three port the custom objectives of the PU Hellinger trees experiments (`CSBoost`),
  with their settings. Their gradients match the original to float32 rounding, and their
  test scores match within 1e-2 (exactly under the original's XGBoost 2.1 and platform).
- The optional extra `pypu-learn[xgboost]` for these models.

## [0.2.0] - 2026-11-07

PU Hellinger-distance decision trees and forests, rewritten in NumPy from the original
Cython code and checked to build the same trees.

### Added

- `pypu.models.PUHellingerTree`: a decision tree whose splits maximize the Hellinger
  distance between the estimated positives and negatives, using the class prior. A NumPy
  rewrite of the original PU Hellinger tree code (a Cython fork of scikit-learn 1.0's
  trees, which only built under Python 3.8): it builds the same trees for the same data and
  `random_state`. `predict_pu_proba` gives the original code's prior-corrected
  probabilities.
- `pypu.models.PUHellingerForest`: a random forest of those trees, each grown on all labeled
  positives and a bootstrap sample of the unlabeled samples. It builds the same trees as
  the original `PURandomForestClassifier`, and defaults to the settings of its experiments.

### Changed

- `predict` now counts a positive-class probability equal to `threshold_` as negative, as
  scikit-learn does (the `argmax` of `predict_proba`, the sign of `decision_function`); it
  counted it as positive. Only exact ties change, which trees produce often.

## [0.1.0] - 2026-11-06

The first release with PU learning methods: five new models, four of them ported from research
code and checked against outputs of the originals, plus PU data from labeled datasets,
evaluation metrics (including estimates from PU labels alone) and class-prior estimation.

### Added

- Models:
  - `pypu.models.IFPU`: LightGBM with a focal-loss PU objective. Its defaults are the
    settings of the original iFPU experiments, whose saved test scores it reproduces exactly.
  - `pypu.models.IFPNU`: an anomaly detector (scikit-learn's `IsolationForest` by default, or
    any PyOD detector) marks the least anomalous unlabeled samples as reliable negatives,
    then LightGBM is trained on the focal-loss objective of the original iFPNU experiments.
    Its defaults are the settings of those experiments, and from the same anomaly scores it
    reproduces the original code exactly.
  - `pypu.models.SAREM`: SAR-EM (Bekker, Robberechts and Davis), which learns a classifier
    and a propensity model by expectation-maximization when labels are selected at random
    (SAR). `predict_propensity` estimates the probability that a positive gets labeled, and
    `propensity_features` restricts it to chosen columns. Ported from the SAR-PU code used
    in the iFPNU experiments, whose results it reproduces.
  - `pypu.models.ElkanNoto`: the classifier of Elkan and Noto, either a labeled-vs-unlabeled
    classifier divided by the label frequency or (`weighted=True`) a classifier trained on
    weighted copies of the unlabeled samples. The label frequency is estimated from
    out-of-fold scores, or derived from `prior`.
  - `pypu.models.PUBagging`: PU bagging (Mordelet and Vert), with any scikit-learn
    classifier and out-of-bag probabilities for the training samples (`oob_proba=True`). It
    builds the same bags as the `BaggingPuClassifier` used in earlier experiments.
- `pypu.prior`, for estimating the class prior from PU data: `BBEPrior` (Best Bin
  Estimation) and `ElkanNotoPrior` (Elkan and Noto's `e1`, `e2` and `e3`). Both also estimate
  the label frequency and the share of positives among the unlabeled samples.
- `pypu.datasets`, for making PU data from fully labeled binary datasets:
  - Labeling mechanisms, each returning PU labels `s`:
    - `label_scar`: uniform (SCAR).
    - `label_sar_knn`: by the distance from each positive to its nearest negatives, with
      `bias="near"` or `"far"`.
    - `label_sar_propensity`: a known logistic propensity on chosen features.
    - `label_sar_cluster`: labels from only some k-means clusters of the positives.
    - `label_sar_confidence`: by out-of-fold classifier confidence, with `bias="easy"` or
      `"hard"`.
  - `make_pu_split`: a stratified train/test split with PU labels on the training part.
  - `make_case_control`: data for the case-control scenario.
  - `pu_summary`, `pollution_from_label_frequency` and `label_frequency_from_pollution`.
  - `legacy=True` on `label_scar` and `label_sar_knn` reproduces the labels of the
    `create_pu_data` function used in the iFPU and iFPNU experiments.
- `pypu.metrics`:
  - With true labels: `r_precision`, `precision_at_k`, `recall_at_k`, `f1_at_prior` and
    `max_f1_score`, matching the metrics of the iFPU and iFPNU experiments. Ties at the
    cut-off of `precision_at_k` and `recall_at_k` count in proportion instead of being
    broken arbitrarily.
  - From PU labels alone, for model selection: `lee_liu_score`, and `pu_recall_score`,
    `pu_precision_score`, `pu_f1_score` and `pu_roc_auc_score`, which estimate the usual
    metrics given the class prior. All work with `sklearn.metrics.make_scorer`.
- `pypu.utils.check_true_labels`.
- `THIRD_PARTY_NOTICES.md`, listing the code PyPU derives from, shipped with the wheel and
  the sdist.
- The docstring examples run as part of the test suite.

### Changed

- LightGBM (>= 4.0) is now a dependency. On macOS it needs the OpenMP runtime
  (`brew install libomp`).

## [0.0.1] - 2026-11-06

### Added

- Project scaffold: packaging with hatchling and version numbers taken from git tags
  (hatch-vcs), CI on Linux, macOS and Windows for Python 3.11–3.14, and a tag-triggered
  release workflow that publishes to TestPyPI and PyPI.
- `pypu.base.BasePUClassifier`, the interface shared by all models: `fit(X, s)`,
  `predict_proba`, `decision_function` and `predict`.
- `pypu.models.NaivePU`, the baseline that treats unlabeled samples as negatives.
- `pypu.utils.check_pu_labels`, `pypu.utils.check_prior`, `pypu.utils.threshold_from_prior`
  and `pypu.show_versions()`.

[Unreleased]: https://github.com/izavits/PyPU/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/izavits/PyPU/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/izavits/PyPU/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/izavits/PyPU/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/izavits/PyPU/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/izavits/PyPU/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/izavits/PyPU/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/izavits/PyPU/releases/tag/v0.0.1
