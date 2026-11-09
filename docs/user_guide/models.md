# Choosing a model

PyPU has twenty models, and none is best on every dataset. They all have the same
interface, so trying several is cheap: fit each on the same PU data and compare them.
This page helps decide where to start.

## Five questions

**Are the labels selected completely at random?** Most methods assume SCAR labels (see
[SCAR and SAR](pu_learning.md#which-positives-get-labeled-scar-and-sar)). If the labeled
positives are likely a biased sample, use a model of the propensity:
{class}`~pypu.models.SAREM` (no prior needed, no extra), {class}`~pypu.models.LBE`,
{class}`~pypu.models.PUe`, or {class}`~pypu.models.PropensityWeightedPU` with a known
propensity or SAREM's estimate. The other models were designed for SCAR labels.

**Do you know the class prior?** Eight models require it, four use it when given, and the
others do without:

- **Required:** {class}`~pypu.models.RebalancedPU`, {class}`~pypu.models.IFPU`,
  {class}`~pypu.models.IFPNU`, {class}`~pypu.models.PUHellingerTree`,
  {class}`~pypu.models.PUHellingerForest`, {class}`~pypu.models.UPUBoost`,
  {class}`~pypu.models.NNPUBoost`, {class}`~pypu.models.ImbalancedNNPUBoost`.
- **Used when given:** {class}`~pypu.models.NaivePU` (to set its threshold),
  {class}`~pypu.models.ElkanNoto` and {class}`~pypu.models.RankPruning` (instead of
  estimating the label frequency), {class}`~pypu.models.PUe` (instead of estimating the
  prior with KM1, which needs cvxopt).
- **Not needed:** {class}`~pypu.models.SpyPU`, {class}`~pypu.models.PEBL`,
  {class}`~pypu.models.BiasedSVM`, {class}`~pypu.models.PUBagging`,
  {class}`~pypu.models.ALMPU`, {class}`~pypu.models.SAREM`, {class}`~pypu.models.LBE`.
  {class}`~pypu.models.PropensityWeightedPU` needs the propensity instead.

When the prior is unknown, [estimate it](prior.md) and pass the estimate on, or start
with a model that needs none.

**Do you need labels, probabilities or a ranking?** Every model ranks samples with
`decision_function`, which is all that ROC AUC or a top-$k$ list needs. For labels, the
models that estimate $P(y = 1 \mid x)$ or are built to classify, such as
{class}`~pypu.models.ElkanNoto`, {class}`~pypu.models.RebalancedPU`,
{class}`~pypu.models.SpyPU` and {class}`~pypu.models.RankPruning`, can be used with
`predict` as they are. Others rank well but their probabilities are not calibrated, as
their docstrings say: {class}`~pypu.models.IFPU`, {class}`~pypu.models.PUBagging`, the
Hellinger trees and forests, and the probabilities of {class}`~pypu.models.BiasedSVM` and
{class}`~pypu.models.PEBL`. When the prior is known, predicting the top $\pi$ fraction
positive is a safe default for any model (see
[What a fitted model returns](pu_learning.md#what-a-fitted-model-returns)).

**Are the positives rare?** {class}`~pypu.models.IFPU` and {class}`~pypu.models.IFPNU`
use a focal loss, which focuses training on the hard samples, and
{class}`~pypu.models.ImbalancedNNPUBoost` trains as if the positives were more frequent.
IFPNU also assumes that the positives look anomalous: it takes the samples an anomaly
detector flags as candidate positives.

**Which dependencies can you install?** Fourteen models need only scikit-learn and
LightGBM. The boosting models of the PU risks need XGBoost, and the neural models need
PyTorch (see [Installation](../install.md)).

## All the models

Grouped as in the [API reference](../api/index.rst) and the
[examples](../auto_examples/index.rst):

| Model | Approach | Labels | Prior | Extra |
| --- | --- | --- | --- | --- |
| {class}`~pypu.models.NaivePU` | Unlabeled samples as negatives: the baseline | SCAR | optional | |
| {class}`~pypu.models.ElkanNoto` | A classifier of labeled vs. unlabeled samples, divided by the label frequency, or retrained on weighted samples | SCAR | optional | |
| {class}`~pypu.models.RebalancedPU` | A classifier of labeled vs. unlabeled samples, weighted so that it classifies like one of the true labels | SCAR | required | |
| {class}`~pypu.models.SpyPU` | Reliable negatives below hidden "spies", then a classifier, optionally refined by EM | SCAR | no | |
| {class}`~pypu.models.PEBL` | Reliable negatives from spies or 1-DNF, then an iterative SVM | SCAR | no | |
| {class}`~pypu.models.RankPruning` | Prunes the unlabeled samples that look positive, then refits | SCAR | optional | |
| {class}`~pypu.models.IFPNU` | Reliable negatives from an anomaly detector, then focal-loss boosting | SCAR | required | |
| {class}`~pypu.models.PUBagging` | Classifiers of the labeled samples vs. bootstrap samples of the unlabeled ones | SCAR | no | |
| {class}`~pypu.models.PUHellingerTree` | A decision tree with a Hellinger-distance split criterion for PU data | SCAR | required | |
| {class}`~pypu.models.PUHellingerForest` | A random forest of those trees | SCAR | required | |
| {class}`~pypu.models.BiasedSVM` | An SVM with class weights chosen by Lee and Liu's criterion | SCAR | no | |
| {class}`~pypu.models.UPUBoost` | Gradient boosting on the unbiased PU risk | SCAR | required | XGBoost |
| {class}`~pypu.models.NNPUBoost` | Gradient boosting on the objective of the nnPU experiments | SCAR | required | XGBoost |
| {class}`~pypu.models.ImbalancedNNPUBoost` | The same, trained for a larger class prior | SCAR | required | XGBoost |
| {class}`~pypu.models.IFPU` | Gradient boosting with a focal loss for PU data | SCAR | required | |
| {class}`~pypu.models.ALMPU` | A neural network trained with an augmented Lagrangian, then towards pseudo-labels | SCAR | no | PyTorch |
| {class}`~pypu.models.SAREM` | EM over a classifier and a propensity model | SAR | no | |
| {class}`~pypu.models.PropensityWeightedPU` | A classifier on samples weighted by the inverse propensity | SAR | propensity | |
| {class}`~pypu.models.LBE` | EM over a classifier and a model of the labeling bias | SAR | no | PyTorch |
| {class}`~pypu.models.PUe` | A classifier with normalized inverse propensity weights | SAR | optional | PyTorch |

Several models port the code of published experiments and behave as that code did, which
sometimes differs from the paper; [Implementation notes](implementation_notes.md) lists
where.

## Trying several models

Because the interface is shared, comparing models is a loop. Scale the features in a
pipeline, since many models default to a logistic regression or an SVM, and compare the
ranking (ROC AUC), the labels of `predict` (F1), and the labels from the prior
(F1 at the prior):

```pycon
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.metrics import f1_score, roc_auc_score
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> from pypu.datasets import make_pu_split
>>> from pypu.metrics import f1_at_prior
>>> from pypu.models import ElkanNoto, NaivePU, RankPruning, RebalancedPU, SpyPU
>>> data = load_breast_cancer()
>>> X, y = data.data, 1 - data.target
>>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
>>> prior = float(y_train.mean())  # known here; in practice, an estimate
>>> models = {
...     "NaivePU": NaivePU(),
...     "ElkanNoto": ElkanNoto(random_state=0),
...     "SpyPU": SpyPU(random_state=0),
...     "RankPruning": RankPruning(random_state=0),
...     "RebalancedPU": RebalancedPU(prior=prior),
... }
>>> for name, model in models.items():
...     clf = make_pipeline(StandardScaler(), model).fit(X_train, s_train)
...     scores = clf.predict_proba(X_test)[:, 1]
...     auc = roc_auc_score(y_test, scores)
...     f1 = f1_score(y_test, clf.predict(X_test))
...     print(f"{name:13} {auc:.2f} {f1:.2f} {f1_at_prior(y_test, scores, prior):.2f}")
NaivePU       0.98 0.57 0.88
ElkanNoto     0.98 0.89 0.88
SpyPU         0.98 0.89 0.91
RankPruning   0.99 0.95 0.94
RebalancedPU  0.98 0.85 0.92
```

All five rank the test samples well, but the naive classifier's own labels are poor:
it predicts too few positives. Its scores thresholded at the prior do far better. The
example [Comparing the methods](../auto_examples/comparison/plot_compare_methods.rst)
runs every model this way, averaged over several splits.

## Where to start

- Fit {class}`~pypu.models.NaivePU` first: the baseline every method should beat.
- Without a prior, try {class}`~pypu.models.ElkanNoto`, {class}`~pypu.models.SpyPU`,
  {class}`~pypu.models.RankPruning` and {class}`~pypu.models.BiasedSVM`: they are quick,
  need only scikit-learn, and estimate what they need from the data.
- With a prior, add {class}`~pypu.models.RebalancedPU`, and the boosting models when the
  data are non-linear: {class}`~pypu.models.IFPU`, or
  {class}`~pypu.models.UPUBoost` and {class}`~pypu.models.NNPUBoost` with XGBoost.
- If the labels may be biased, start with {class}`~pypu.models.SAREM`, restricting its
  propensity to the features that drive labeling (`propensity_features=`).
- Tune each model's parameters from the PU labels alone, as
  [Evaluation and model selection](evaluation.md) shows.
