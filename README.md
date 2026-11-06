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

On macOS, the LightGBM-based models need the OpenMP runtime, which LightGBM's wheels do not
include: `brew install libomp`.

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

## Evaluation

> New in the unreleased 0.1.0: install from GitHub to try it.

`pypu.metrics` has two kinds of metrics.

- **With the true labels of a test set:** `r_precision`, `precision_at_k`, `recall_at_k`,
  `f1_at_prior` (F1 with the top `prior` fraction predicted positive) and `max_f1_score`.
  They rank the scores, so they suit PU methods whose scores are not calibrated
  probabilities. scikit-learn's ROC AUC and average precision apply as they are.
- **From PU labels alone**, for model selection when no true labels exist: `lee_liu_score`
  needs nothing else, while `pu_recall_score`, `pu_precision_score`, `pu_f1_score` and
  `pu_roc_auc_score` estimate the usual metrics given the class prior. They assume SCAR
  labels and work as scikit-learn scorers:

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from pypu.datasets import make_pu_split
from pypu.metrics import pu_roc_auc_score, r_precision
from pypu.models import NaivePU

X, y = make_classification(n_samples=2000, weights=[0.8], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()  # in practice, an estimate

search = GridSearchCV(
    NaivePU(prior=prior),
    {"estimator": [LogisticRegression(C=C) for C in (0.01, 0.1, 1.0)]},
    scoring=make_scorer(pu_roc_auc_score, response_method="predict_proba", prior=prior),
).fit(X_train, s_train)  # only the PU labels are used

print("R-precision:", r_precision(y_test, search.predict_proba(X_test)[:, 1]))
```

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

## Estimating the class prior

> New in the unreleased 0.1.0: install from GitHub to try it.

When the prior is unknown, estimate it from the PU training data with `pypu.prior`, assuming
SCAR labels:

```python
from pypu.models import IFPU
from pypu.prior import BBEPrior

estimator = BBEPrior().fit(X_train, s_train)
clf = IFPU(prior=estimator.prior_).fit(X_train, s_train)
```

- `BBEPrior`: Best Bin Estimation (Garg et al., NeurIPS 2021). It needs a region where
  (almost) all samples are positive, and copes with overlapping classes.
- `ElkanNotoPrior`: the estimators of Elkan and Noto (KDD 2008), `method="e1"`, `"e2"` or
  `"e3"`. They are accurate when the classes barely overlap and overestimate the prior when
  they overlap more.

Both also estimate the label frequency (`label_frequency_`) and the share of positives among
the unlabeled samples (`pollution_`). Estimates are biased upwards when the classifier
cannot isolate a purely positive region, and noisy with few labeled positives; a more
flexible classifier (`estimator=`) can help BBE on non-linear data.

## Methods

| Model | Description | Status |
| --- | --- | --- |
| `NaivePU` | Treats unlabeled samples as negatives; the usual lower bound | available |
| `IFPU` | LightGBM with a focal-loss objective that treats unlabeled samples as positive with probability `prior` | on main (0.1.0) |
| `ElkanNoto` | Elkan and Noto: a labeled-vs-unlabeled classifier corrected by the label frequency, or retrained on weighted unlabeled samples | on main (0.1.0) |
| `PUBagging` | Mordelet and Vert: classifiers trained on the labeled positives against bootstrap samples of the unlabeled ones | on main (0.1.0) |
| `IFPNU` | An anomaly detector picks reliable negatives, then LightGBM with a focal loss that weights positives, reliable negatives and the remaining unlabeled samples | on main (0.1.0) |
| `SAREM` | Bekker et al.: EM that learns the classifier together with the propensity, the probability that a positive gets labeled (SAR) | on main (0.1.0) |
| `PUHellingerTree`, `PUHellingerForest` | Hellinger-distance decision trees and forests for PU data | planned (0.2.0) |
| `NNPU`, `LBE`, `PUe`, `ALMPU` | Neural PU methods (optional PyTorch dependency) | planned (0.3.0) |

"On main" models are not released yet; install from GitHub to try them. `IFPU`'s defaults
are the settings of the original iFPU experiments (10 trees, learning rate 0.1, focal
`gamma=3`), and it reproduces their test scores exactly:

```python
from pypu.models import IFPU

clf = IFPU(prior=0.2).fit(X_train, s_train)  # prior: fraction of positives in the training data
```

`IFPNU`'s defaults are those of the original iFPNU experiments (100 trees), and from the
same anomaly scores it reproduces the original code exactly. Its first step uses
scikit-learn's `IsolationForest` unless given another detector, such as any PyOD detector:

```python
from pypu.models import IFPNU

clf = IFPNU(prior=0.2, random_state=0).fit(X_train, s_train)
clf.pnu_labels_  # 1: labeled positive, 0: reliable negative, -1: left unlabeled
```

`SAREM` needs no prior. It works best when the propensity is restricted to the features
that drive labeling:

```python
from pypu.models import SAREM

clf = SAREM(propensity_features=[0, 3]).fit(X_train, s_train)
clf.predict_propensity(X_test)  # P(labeled | positive, x)
```

`ElkanNoto` estimates the label frequency itself, or takes the `prior`. `PUBagging` builds
the same bags as the `BaggingPuClassifier` of earlier experiments, and its out-of-bag
probabilities score the unlabeled training samples:

```python
from sklearn.svm import SVC

from pypu.models import ElkanNoto, PUBagging

clf = ElkanNoto(weighted=True).fit(X_train, s_train)
bagging = PUBagging(SVC(kernel="linear"), oob_proba=True).fit(X_train, s_train)
unlabeled_scores = bagging.oob_proba_[s_train == 0, 1]
```

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
