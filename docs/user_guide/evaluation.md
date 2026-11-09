# Evaluation and model selection

How to evaluate a PU model depends on what is known. In a benchmark, made with
[Making PU data](datasets.md), the test set keeps its true labels and the usual metrics
apply. With real PU data there are no true labels, but some metrics can still be
estimated from the PU labels. {mod}`pypu.metrics` has both kinds.

## With the true labels

Many PU models rank samples well without giving calibrated probabilities (see
[Choosing a model](models.md)), so prefer metrics of the ranking, which need no
threshold:

- {func}`~sklearn.metrics.roc_auc_score` and {func}`~sklearn.metrics.average_precision_score`
  from scikit-learn;
- {func}`~pypu.metrics.r_precision`: the precision among the top $R$ samples, where $R$ is
  the number of positives;
- {func}`~pypu.metrics.precision_at_k` and {func}`~pypu.metrics.recall_at_k`, for a
  top-$k$ list.

To judge labels, compare the F1 of the model's own `predict` with
{func}`~pypu.metrics.f1_at_prior`, which predicts the top $\pi$ fraction of the samples
positive. {func}`~pypu.metrics.max_f1_score` gives the best F1 over all thresholds: an
optimistic bound, since it chooses the threshold with the test labels.

```pycon
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> from pypu.datasets import make_pu_split
>>> from pypu.metrics import f1_at_prior, max_f1_score, r_precision
>>> from pypu.models import ElkanNoto
>>> data = load_breast_cancer()
>>> X, y = data.data, 1 - data.target
>>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
>>> prior = float(y_train.mean())
>>> clf = make_pipeline(StandardScaler(), ElkanNoto(random_state=0)).fit(X_train, s_train)
>>> scores, y_pred = clf.predict_proba(X_test)[:, 1], clf.predict(X_test)
>>> round(roc_auc_score(y_test, scores), 2)
0.98
>>> round(average_precision_score(y_test, scores), 2)
0.97
>>> round(r_precision(y_test, scores), 2)
0.89
>>> round(f1_score(y_test, y_pred), 2), round(f1_at_prior(y_test, scores, prior), 2)
(0.89, 0.88)
>>> round(max_f1_score(y_test, scores), 2)
0.91
```

## From the PU labels alone

Without true labels, the PU labels of held-out data still carry information. Under SCAR
the labeled positives are a random sample of the positives, so:

- the **recall** on the labeled positives estimates the recall on all positives:
  {func}`~pypu.metrics.pu_recall_score`;
- with the class prior, it gives estimates of the **precision**, the **F1 score** and the
  **ROC AUC**: {func}`~pypu.metrics.pu_precision_score`, {func}`~pypu.metrics.pu_f1_score`
  and {func}`~pypu.metrics.pu_roc_auc_score`. For example, the precision is
  $\pi r / P(\hat y = 1)$, where $r$ is the recall on the labeled positives and
  $P(\hat y = 1)$ the share of samples predicted positive;
- without the prior, {func}`~pypu.metrics.lee_liu_score`, $r^2 / P(\hat y = 1)$, equals
  $\text{precision} \cdot \text{recall} / \pi$ [^leeliu]. For a fixed dataset it ranks
  classifiers like the product of precision and recall. It is not bounded by 1, so
  compare it only between models on the same data.

Here the test set gets PU labels too, to compare each estimate with the true value:

```pycon
>>> from sklearn.metrics import precision_score, recall_score
>>> from pypu.datasets import label_scar
>>> from pypu.metrics import pu_f1_score, pu_precision_score
>>> from pypu.metrics import pu_recall_score, pu_roc_auc_score
>>> s_test = label_scar(y_test, label_frequency=0.5, random_state=1)
>>> true = {
...     "recall": recall_score(y_test, y_pred),
...     "precision": precision_score(y_test, y_pred),
...     "F1": f1_score(y_test, y_pred),
...     "ROC AUC": roc_auc_score(y_test, scores),
... }
>>> estimated = {
...     "recall": pu_recall_score(s_test, y_pred),
...     "precision": pu_precision_score(s_test, y_pred, prior),
...     "F1": pu_f1_score(s_test, y_pred, prior),
...     "ROC AUC": pu_roc_auc_score(s_test, scores, prior),
... }
>>> for name in true:
...     print(f"{name:9} true {true[name]:.2f}  estimated {estimated[name]:.2f}")
recall    true 0.83  estimated 0.85
precision true 0.96  estimated 0.99
F1        true 0.89  estimated 0.92
ROC AUC   true 0.98  estimated 0.97
```

With only 27 labeled positives in the test set the estimates are noisy, but close. Note
that the plain AUC of separating labeled from unlabeled samples would be far lower, as
the unlabeled set contains positives; {func}`~pypu.metrics.pu_roc_auc_score` corrects for
them.

## Tuning a model from PU labels

The estimates work as scikit-learn scorers, so {class}`~sklearn.model_selection.GridSearchCV`
can tune a model on PU data: it splits the training samples and scores each fold's
held-out part from its PU labels. Here it tunes the regularization of ElkanNoto's
logistic regression by Lee and Liu's criterion:

```pycon
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import make_scorer
>>> from sklearn.model_selection import GridSearchCV
>>> from pypu.metrics import lee_liu_score
>>> model = ElkanNoto(LogisticRegression(max_iter=1000), random_state=0)
>>> search = GridSearchCV(
...     make_pipeline(StandardScaler(), model),
...     {"elkannoto__estimator__C": [0.001, 0.01, 0.1, 1, 10]},
...     scoring=make_scorer(lee_liu_score),
... ).fit(X_train, s_train)  # only the PU labels are used
>>> search.best_params_
{'elkannoto__estimator__C': 0.01}
>>> round(f1_score(y_test, search.predict(X_test)), 2)
0.94
```

The chosen value also gives the best test F1 of the five. Scorers that need probabilities
and the prior are made the same way:

```pycon
>>> from sklearn.model_selection import cross_val_score
>>> pu_auc = make_scorer(pu_roc_auc_score, response_method="predict_proba", prior=prior)
>>> cv_scores = cross_val_score(search.best_estimator_, X_train, s_train, scoring=pu_auc)
>>> round(float(cv_scores.mean()), 2)
0.96
```

Choose the criterion by what the model is for: the ROC AUC estimate for a ranking, Lee
and Liu's criterion or the F1 estimate for labels. The example
[Tuning a model without true labels](../auto_examples/getting_started/plot_model_selection.rst)
compares the estimates with the true scores over a whole grid.

## Caveats

- All these estimates assume SCAR labels. Under SAR the labeled positives are not a
  random sample of the positives, and the estimates are biased.
- Except for the recall and Lee and Liu's criterion, they need the class prior. An
  estimate from {mod}`pypu.prior` works, but its error carries over.
- They are noisy with few labeled positives in the held-out data: prefer
  cross-validation over a single split.

## References

[^leeliu]: W. S. Lee and B. Liu. "Learning with positive and unlabeled examples using
    weighted logistic regression." ICML, 2003.
