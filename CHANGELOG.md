# Changelog

All notable changes to PyPU are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is `0.y.z`
the public API may change between minor versions; such changes are always listed under
**Changed** or **Removed**.

## [Unreleased]

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

[Unreleased]: https://github.com/izavits/PyPU/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/izavits/PyPU/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/izavits/PyPU/releases/tag/v0.0.1
