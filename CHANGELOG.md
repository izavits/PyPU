# Changelog

All notable changes to PyPU are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is `0.y.z`
the public API may change between minor versions; such changes are always listed under
**Changed** or **Removed**.

## [Unreleased]

### Added

- Project scaffold: packaging with hatchling and version numbers taken from git tags
  (hatch-vcs), CI on Linux, macOS and Windows for Python 3.11–3.14, and a tag-triggered
  release workflow that publishes to TestPyPI and PyPI.
- `pypu.base.BasePUClassifier`, the interface shared by all models: `fit(X, s)`,
  `predict_proba`, `decision_function` and `predict`.
- `pypu.models.NaivePU`, the baseline that treats unlabeled samples as negatives.
- `pypu.utils.check_pu_labels`, `pypu.utils.check_prior`, `pypu.utils.threshold_from_prior`
  and `pypu.show_versions()`.

[Unreleased]: https://github.com/izavits/PyPU/commits/main
