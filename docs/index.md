---
html_theme.sidebar_secondary.remove: true
---

# PyPU

**Positive-unlabeled (PU) learning in Python, with a scikit-learn style API.**

In PU learning the training data hold some examples labeled positive and many unlabeled
examples, which are a mix of positives and negatives. No example is labeled negative. This
happens whenever only one class gets recorded: confirmed frauds, known disease genes,
reported misstatements, clicked items.

PyPU collects PU learning methods behind one interface, in the spirit of
[PyOD](https://github.com/yzhao062/pyod) for outlier detection: import a model, `fit` it on
PU data, and predict on new data. It also makes PU data from fully labeled datasets,
evaluates models with and without true labels, and estimates the class prior.

:::{note}
PyPU is in alpha: the API may still change before 1.0. More methods are planned; see the
[roadmap](https://github.com/izavits/PyPU/blob/main/ROADMAP.md).
:::

## Install

```bash
pip install pypu-learn
```

The package is imported as `pypu`. Some models need an optional extra; see
[Installation](install.md).

## Quick start

```python
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from pypu.datasets import make_pu_split
from pypu.models import ElkanNoto

# A fully labeled dataset, turned into PU data: half of the
# training positives are labeled, the rest stay unlabeled.
X, y = make_classification(n_samples=2000, weights=[0.8], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.5, random_state=0)

clf = ElkanNoto(random_state=0)
clf.fit(X_train, s_train)  # s = 1: labeled positive, s = 0: unlabeled

scores = clf.decision_function(X_test)  # higher means more likely positive
print("ROC AUC:", roc_auc_score(y_test, scores))
```

Every model has the same methods:

| Method | Returns |
| --- | --- |
| `fit(X, s)` | The fitted model. `s[i] = 1` marks a labeled positive, `s[i] = 0` an unlabeled sample. |
| `predict_proba(X)` | An `(n_samples, 2)` array; column 1 is the positive-class probability. |
| `decision_function(X)` | One score per sample; higher means more likely positive. |
| `predict(X)` | Labels in `{0, 1}`, thresholding the positive-class probability at `threshold_`. |

Models that need the class prior π = P(y = 1) take it as a `prior` parameter.

## Where to go next

- [Installation](install.md): the optional extras and platform notes.
- [Examples](auto_examples/index.rst): runnable examples with figures, one per model.
- [API reference](api/index.rst): every model, dataset function, metric and prior
  estimator.

```{toctree}
:hidden:

install
auto_examples/index
api/index
changelog
contributing
citing
license
```
