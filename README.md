# PyPU

**Positive-unlabeled (PU) learning in Python, with a scikit-learn style API.**

[![CI](https://github.com/izavits/PyPU/actions/workflows/ci.yml/badge.svg)](https://github.com/izavits/PyPU/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pypu-learn?label=PyPI)](https://pypi.org/project/pypu-learn/)
[![Python versions](https://img.shields.io/pypi/pyversions/pypu-learn?label=Python)](https://pypi.org/project/pypu-learn/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/izavits/PyPU/blob/main/LICENSE)

> **Status: pre-alpha.** The common interface is in place and the methods are being ported.
> See the [roadmap](https://github.com/izavits/PyPU/blob/main/ROADMAP.md).

In PU learning the training data hold some examples labeled positive and many unlabeled
examples, which are a mix of positives and negatives. No example is labeled negative. This
happens whenever only one class gets recorded: confirmed frauds, known disease genes, reported
misstatements, clicked items.

PyPU collects PU learning methods behind one interface, in the spirit of
[PyOD](https://github.com/yzhao062/pyod) for outlier detection: import a model, `fit` it on
PU data, and predict on new data.

## Installation

```bash
pip install pypu-learn
```

The package is imported as `pypu` and needs Python 3.11 or newer. (The PyPI name `pypu`
belongs to an unrelated project, hence `pypu-learn`.)

To try features that are not released yet, install from GitHub:

```bash
pip install git+https://github.com/izavits/PyPU.git
```

## Quick start

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pypu.models import NaivePU

# A fully labeled dataset...
X, y = make_classification(n_samples=2000, weights=[0.8], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# ...turned into PU data: label half of the training positives, leave the rest unlabeled.
rng = np.random.default_rng(0)
s_train = ((y_train == 1) & (rng.random(len(y_train)) < 0.5)).astype(int)

clf = NaivePU(prior=y_train.mean())
clf.fit(X_train, s_train)  # s = 1: labeled positive, s = 0: unlabeled

scores = clf.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, scores))
```

## PU data from labeled datasets

> New in the unreleased 0.1.0: install from GitHub to try it.

`pypu.datasets` turns a fully labeled binary dataset into PU data, so methods can be compared
against the true labels. `make_pu_split` does it in one call: a stratified train/test split,
then PU labels on the training part.

```python
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from pypu.datasets import make_pu_split, pu_summary
from pypu.models import NaivePU

X, y = make_classification(n_samples=2000, weights=[0.8], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(
    X, y, label_frequency=0.3, mechanism="sar_knn", random_state=0
)
print(pu_summary(y_train, s_train))  # prior, label frequency, positives among the unlabeled

clf = NaivePU(prior=y_train.mean()).fit(X_train, s_train)
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

The labeling mechanism decides which positives get labeled:

| Function (`mechanism=`) | Labeled positives |
| --- | --- |
| `label_scar` (`"scar"`) | Chosen uniformly at random (SCAR) |
| `label_sar_knn` (`"sar_knn"`) | Mostly near the negatives (`bias="near"`) or far from them (`bias="far"`), by distance to the nearest negatives |
| `label_sar_propensity` (`"sar_propensity"`) | Each labeled with a known logistic propensity on chosen features, which can be returned |
| `label_sar_cluster` (`"sar_cluster"`) | Taken from only some k-means clusters of the positives |
| `label_sar_confidence` (`"sar_confidence"`) | Mostly the ones a classifier recognises easily (`bias="easy"`) or poorly (`bias="hard"`) |

`label_frequency` is the fraction of positives that get labeled. To set the share of
positives among the unlabeled samples instead, convert it with
`label_frequency_from_pollution`. `make_case_control` makes data for the case-control
scenario, where the unlabeled set is a separate sample with the population's class prior.

## The interface

Every model derives from `pypu.base.BasePUClassifier` and is a scikit-learn estimator, so it
works with `clone`, `Pipeline` and `GridSearchCV`.

| Method | Returns |
| --- | --- |
| `fit(X, s)` | The fitted model. `s[i] = 1` marks a labeled positive, `s[i] = 0` an unlabeled sample. |
| `predict_proba(X)` | An `(n_samples, 2)` array; column 1 is the positive-class probability. |
| `decision_function(X)` | One score per sample; higher means more likely positive. |
| `predict(X)` | Labels in `{0, 1}`, thresholding the positive-class probability at `threshold_`. |

Models that need the class prior π = P(y = 1) take it as a `prior` parameter.

## Methods

| Model | Description | Status |
| --- | --- | --- |
| `NaivePU` | Treats unlabeled samples as negatives; the usual lower bound | available |
| `IFPU` | Focal-loss PU risk with gradient boosting | planned (0.1.0) |
| `IFPNU` | Anomaly detector picks reliable negatives, then a focal PNU risk with gradient boosting | planned (0.1.0) |
| `SAREM` | EM for labels selected at random (SAR) | planned (0.1.0) |
| `ElkanNoto`, `PUBagging` | Classic PU baselines | planned (0.1.0) |
| `PUHellingerTree`, `PUHellingerForest` | Hellinger-distance decision trees and forests for PU data | planned (0.2.0) |
| `NNPU`, `LBE`, `PUe`, `ALMPU` | Neural PU methods (optional PyTorch dependency) | planned (0.3.0) |

Also planned: class-prior estimation and PU-specific metrics.

## Contributing

Contributions are welcome. See
[CONTRIBUTING.md](https://github.com/izavits/PyPU/blob/main/CONTRIBUTING.md) for the
development setup, how to add a model, and the release process. Changes are listed in
[CHANGELOG.md](https://github.com/izavits/PyPU/blob/main/CHANGELOG.md).

## Citing

If you use PyPU in your research, please cite it; GitHub's "Cite this repository" button uses
[CITATION.cff](https://github.com/izavits/PyPU/blob/main/CITATION.cff).

## License

PyPU is distributed under the
[BSD 3-Clause License](https://github.com/izavits/PyPU/blob/main/LICENSE).
