# Changelog

All notable changes to PyPU are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is `0.y.z`
the public API may change between minor versions; such changes are always listed under
**Changed** or **Removed**.

## [Unreleased]

### Added

- `pypu.models.IFPU`: LightGBM with a focal-loss PU objective. Its defaults are the settings
  of the original iFPU experiments, whose saved test scores it reproduces exactly.
- LightGBM (>= 4.0) is now a dependency. On macOS it needs the OpenMP runtime
  (`brew install libomp`).
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
- The docstring examples run as part of the test suite.

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

[Unreleased]: https://github.com/izavits/PyPU/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/izavits/PyPU/releases/tag/v0.0.1
