# PyPU

**Positive-unlabeled (PU) learning in Python, with a scikit-learn style API.**

[![CI](https://github.com/izavits/PyPU/actions/workflows/ci.yml/badge.svg)](https://github.com/izavits/PyPU/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pypu-learn.svg)](https://pypi.org/project/pypu-learn/)
[![Python versions](https://img.shields.io/pypi/pyversions/pypu-learn.svg)](https://pypi.org/project/pypu-learn/)
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

Until the first release is on PyPI, install from GitHub:

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

Data utilities that turn labeled datasets into PU data in one call are coming in 0.1.0.

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

Also planned: creating PU data from labeled datasets (uniform SCAR selection, KNN-biased and
other SAR selection mechanisms), class-prior estimation, and PU-specific metrics.

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
