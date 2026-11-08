# PyPU

**Positive-unlabeled (PU) learning in Python, with a scikit-learn style API.**

[![CI](https://github.com/izavits/PyPU/actions/workflows/ci.yml/badge.svg)](https://github.com/izavits/PyPU/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pypu-learn?label=PyPI)](https://pypi.org/project/pypu-learn/)
[![Python versions](https://img.shields.io/pypi/pyversions/pypu-learn?label=Python)](https://pypi.org/project/pypu-learn/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/izavits/PyPU/blob/main/LICENSE)

> **Status: alpha.** Nineteen models, PU data generation, evaluation and class-prior estimation
> are available; the API may still change before 1.0. More methods are planned: see the
> [roadmap](https://github.com/izavits/PyPU/blob/main/ROADMAP.md).

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

The XGBoost-based models (`UPUBoost`, `NNPUBoost`, `ImbalancedNNPUBoost`), the
PyTorch-based models (`LBE`, `PUe`, `ALMPU`) and the KM estimator of the class prior (`KMPrior`, also
used by `PUe` when no prior is given) need optional extras:

```bash
pip install "pypu-learn[xgboost]"
pip install "pypu-learn[torch]"
pip install "pypu-learn[cvxopt]"
```

The `cvxopt` extra installs [CVXOPT](https://cvxopt.org), which is licensed under the GPL-3.0;
PyPU itself stays under the BSD 3-Clause License.

On Linux, PyPI's PyTorch wheels include CUDA; for the smaller CPU build, install PyTorch
first as described on [pytorch.org](https://pytorch.org). On macOS, PyTorch bundles its own
OpenMP runtime, which conflicts with the one LightGBM uses: PyPU loads LightGBM first and
runs its PyTorch models on one thread, so they work together, but importing `torch` yourself
before using a LightGBM-based model can crash LightGBM.

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

from pypu.models import ElkanNoto

# A fully labeled dataset...
X, y = make_classification(n_samples=2000, weights=[0.8], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# ...turned into PU data: label half of the training positives, leave the rest unlabeled.
rng = np.random.default_rng(0)
s_train = ((y_train == 1) & (rng.random(len(y_train)) < 0.5)).astype(int)

clf = ElkanNoto(random_state=0)
clf.fit(X_train, s_train)  # s = 1: labeled positive, s = 0: unlabeled

scores = clf.decision_function(X_test)  # higher means more likely positive
print("ROC AUC:", roc_auc_score(y_test, scores))
print("Accuracy:", (clf.predict(X_test) == y_test).mean())
```

## PU data from labeled datasets

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
- `KMPrior`: kernel mean embedding (Ramaswamy, Scott and Tewari, ICML 2016), `method="km1"`
  or `"km2"`, using the authors' code as the PUe experiments did. It works on the features
  directly, without a classifier, and its cost grows quickly with the number of samples
  (`pip install "pypu-learn[cvxopt]"`).

All three also estimate the label frequency (`label_frequency_`) and the share of positives among
the unlabeled samples (`pollution_`). Estimates are biased upwards when the classifier
cannot isolate a purely positive region, and noisy with few labeled positives; a more
flexible classifier (`estimator=`) can help BBE on non-linear data.

## Methods

| Model | Description | Status |
| --- | --- | --- |
| `NaivePU` | Treats unlabeled samples as negatives; the usual lower bound | available |
| `IFPU` | LightGBM with a focal-loss objective that treats unlabeled samples as positive with probability `prior` | available |
| `IFPNU` | An anomaly detector picks reliable negatives, then LightGBM with a focal loss that weights positives, reliable negatives and the remaining unlabeled samples | available |
| `SAREM` | Bekker et al.: EM that learns the classifier together with the propensity, the probability that a positive gets labeled (SAR) | available |
| `ElkanNoto` | Elkan and Noto: a labeled-vs-unlabeled classifier corrected by the label frequency, or retrained on weighted unlabeled samples | available |
| `PUBagging` | Mordelet and Vert: classifiers trained on the labeled positives against bootstrap samples of the unlabeled ones | available |
| `PUHellingerTree` | Decision tree whose splits maximize a Hellinger distance between estimated positives and negatives | available |
| `PUHellingerForest` | Random forest of those trees, each on all labeled positives and a bootstrap sample of the unlabeled ones | available |
| `UPUBoost` | XGBoost on the unbiased PU risk (uPU) | available (`[xgboost]` extra) |
| `NNPUBoost`, `ImbalancedNNPUBoost` | XGBoost on the objectives of the original nnPU and imbalanced nnPU experiments (weighted cross-entropies) | available (`[xgboost]` extra) |
| `LBE` | Gong et al.: EM that learns a logistic classifier together with a logistic model of the labeling bias, as in the original experiments | available (`[torch]` extra) |
| `PUe` | Wang et al.: a logistic classifier trained with normalized inverse propensity weights, as in the original experiments | available (`[torch]` extra) |
| `ALMPU` | Wei et al.: an MLP trained with an augmented-Lagrangian constraint on the unlabeled samples, then retrained towards pseudo-labels from the trends of its predictions | available (`[torch]` extra) |
| `RebalancedPU` | Elkan's rebalancing: a labeled-vs-unlabeled classifier with sample weights from the label frequency, so that it classifies like a classifier of the true labels | on main (0.5.0) |
| `PropensityWeightedPU` | Bekker et al.: risk minimization on SAR data with a known (or estimated) propensity, weighting labeled samples by its inverse | on main (0.5.0) |
| `BiasedSVM` | Liu et al.: an SVM on PU labels with class weights chosen by Lee and Liu's criterion | on main (0.5.0) |
| `SpyPU` | Liu et al.: reliable negatives below hidden "spies", then a classifier, optionally refined by EM (S-EM) | on main (0.5.0) |
| `PEBL` | Reliable negatives from spies or from 1-DNF (binary features), then an iterative SVM (Spy-SVM, PEBL) | on main (0.5.0) |

`IFPU`'s defaults are the settings of the original iFPU experiments (10 trees, learning rate
0.1, focal `gamma=3`), and it reproduces their test scores exactly:

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

`PUHellingerTree` and `PUHellingerForest` build the same trees as the original PU
Hellinger tree code, now in NumPy; their defaults are the settings of its experiments:

```python
from pypu.models import PUHellingerForest, PUHellingerTree

tree = PUHellingerTree(prior=0.2, max_depth=5, random_state=0).fit(X_train, s_train)
forest = PUHellingerForest(prior=0.2, random_state=0).fit(X_train, s_train)
scores = forest.predict_proba(X_test)[:, 1]  # rank samples; not calibrated
```

The boosting models (`pip install "pypu-learn[xgboost]"`) port the XGBoost objectives of the
PU Hellinger trees experiments, with their settings (100 rounds, learning rate 0.3, no L2
regularization). The original nnPU and imbalanced nnPU
objectives apply the non-negative correction per sample, which makes them weighted
cross-entropies rather than Kiryo et al.'s nnPU; PyPU reproduces them as run:

```python
from pypu.models import ImbalancedNNPUBoost, NNPUBoost, UPUBoost

upu = UPUBoost(prior=0.2, random_state=0).fit(X_train, s_train)
nnpu = NNPUBoost(prior=0.2, random_state=0).fit(X_train, s_train)
imbalanced = ImbalancedNNPUBoost(prior=0.2, oversampled_prior=0.5).fit(X_train, s_train)
```

`LBE` (`pip install "pypu-learn[torch]"`) ports the LBE code of the iFPNU experiments, with
their settings: logistic models trained by Adam, 1000 pre-training steps, then 100 EM
iterations of 100 steps. Under the same PyTorch it fits the original's weights to the last
bit (and predicts from them in float64). As in that code, its labeling-bias model ends up as a classifier of labeled versus
unlabeled samples rather than the propensity (see the docstring):

```python
from pypu.models import LBE

clf = LBE(random_state=0).fit(X_train, s_train)  # random_state=i: torch.manual_seed(i)
scores = clf.decision_function(X_test)
```

`PUe` ports the PUe code of the same experiments: it fits the original's weights to the last
bit, and regenerates the experiments' saved test scores. Its defaults are their settings, including only 20 Adam steps per model, which barely
train it: its scores stay close to those of its random initial weights. Raise `n_steps` and
`learning_rate` for a trained model. Without a `prior`, it estimates one with KM1:

```python
from pypu.models import PUe

as_run = PUe(random_state=0).fit(X_train, s_train)  # KM1 prior, needs the cvxopt extra
trained = PUe(prior=0.2, n_steps=2000, learning_rate=0.01, random_state=0).fit(X_train, s_train)
```

`ALMPU` ports the ALM-PU code of the tabular benchmark, with its settings (a small MLP, 20
epochs of the constrained phase, then 30 towards the pseudo-labels). Under the same PyTorch
it trains the same network to the last bit, and under the benchmark's PyTorch it regenerates
all of its saved runs.
On the smallest datasets its pseudo-labels sometimes go wrong, which `pollution_` (the share
of unlabeled samples pseudo-labeled positive) shows:

```python
from pypu.models import ALMPU

clf = ALMPU(random_state=0).fit(X_train, s_train)  # random_state=i: set_seed(i)
clf.pollution_  # far above the expected share of positives: distrust the model
```

The classic methods of Bekker and Davis's survey ("on main" models are not released yet;
install from GitHub to try them) need only scikit-learn. Two need the class prior or the
propensity; the two-step methods find reliable negatives first:

```python
from pypu.models import PEBL, BiasedSVM, PropensityWeightedPU, RebalancedPU, SpyPU

rebalanced = RebalancedPU(prior=0.2).fit(X_train, s_train)  # classifies at threshold 0.5
weighted = PropensityWeightedPU(propensity=0.4).fit(X_train, s_train)  # or a function e(x)
biased = BiasedSVM(random_state=0).fit(X_train, s_train)  # weight chosen by cross-validation
spy = SpyPU(n_em_iter=5, random_state=0).fit(X_train, s_train)  # spies, then S-EM
pebl = PEBL(random_state=0).fit(X_train, s_train)  # spies, then an iterative SVM
spy.reliable_negatives_  # which training samples were taken as negatives
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
