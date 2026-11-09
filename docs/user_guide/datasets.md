# Making PU data

Real PU data have no true labels for the unlabeled samples, so there is nothing to check
a model against. To compare methods, start instead from a dataset whose labels are all
known, hide the labels of some positives, and keep the true labels for evaluation.
{mod}`pypu.datasets` does this.

## A train/test split with PU labels

{func}`~pypu.datasets.make_pu_split` does everything in one call: a stratified train/test
split, then PU labels on the training part. The test part keeps its true labels. Here the
positives are the malignant tumours of scikit-learn's breast cancer data, and half of the
training positives are labeled:

```pycon
>>> from sklearn.datasets import load_breast_cancer
>>> from pypu.datasets import make_pu_split, pu_summary
>>> data = load_breast_cancer()
>>> X, y = data.data, 1 - data.target  # 1: malignant
>>> split = make_pu_split(X, y, label_frequency=0.5, random_state=0)
>>> split.X_train.shape, split.X_test.shape
((426, 30), (143, 30))
>>> pu_summary(split.y_train, split.s_train)
{'n_samples': 426, 'n_positive': 159, 'n_labeled': 80, 'n_unlabeled': 346,
 'prior': 0.3732394366197183, 'label_frequency': 0.5031446540880503,
 'pollution': 0.22832369942196531}
```

The result is a named tuple, {class}`~pypu.datasets.PUSplit`, which also unpacks as
`X_train, X_test, s_train, y_train, y_test`. A model is fitted on `X_train` and
`s_train` only; `y_train` is there to describe the data, as {func}`~pypu.datasets.pu_summary`
does above, and `X_test` with `y_test` to evaluate the model. Lists, arrays, sparse
matrices and pandas DataFrames are accepted, and the parts come back in the same type.

To label data without splitting it, call a labeling function directly, such as
`label_scar(y)` or `label_sar_knn(X, y)`: it returns the PU labels `s`.

## Labeling mechanisms

The labeling mechanism decides *which* positives get labeled (see
[SCAR and SAR](pu_learning.md#which-positives-get-labeled-scar-and-sar)). Choose it with
`mechanism=`, and pass its parameters to {func}`~pypu.datasets.make_pu_split` as keyword
arguments:

| Function (`mechanism=`) | The labeled positives are | Parameters |
| --- | --- | --- |
| {func}`~pypu.datasets.label_scar` (`"scar"`) | chosen uniformly at random (SCAR) | |
| {func}`~pypu.datasets.label_sar_knn` (`"sar_knn"`) | mostly near the negatives, or far from them | `bias="near"` or `"far"`, `n_neighbors` |
| {func}`~pypu.datasets.label_sar_propensity` (`"sar_propensity"`) | each labeled with a known logistic propensity of chosen features | `features`, `coef`, `strength` |
| {func}`~pypu.datasets.label_sar_cluster` (`"sar_cluster"`) | taken from only some k-means clusters of the positives | `n_clusters` |
| {func}`~pypu.datasets.label_sar_confidence` (`"sar_confidence"`) | mostly the ones a classifier recognises easily, or poorly | `bias="easy"` or `"hard"`, `estimator` |

SCAR labels test a method under its usual assumption. The SAR mechanisms test it on
biased labels, each in its own way: labels near or far from the class boundary, labels
driven by a few features, whole kinds of positives never labeled, or only the obvious (or
only the ambiguous) cases labeled.

Under SAR, the labeled positives differ from the positives that stay hidden. Here the
propensity grows with the first feature, the mean radius of the tumour, so the labeled
tumours are larger than the hidden ones; under SCAR the two groups look alike:

```pycon
>>> str(data.feature_names[0])
'mean radius'
>>> settings = {"scar": {}, "sar_propensity": {"features": [0], "strength": 2.0}}
>>> for mechanism, params in settings.items():
...     part = make_pu_split(X, y, mechanism=mechanism, random_state=0, **params)
...     labeled = part.X_train[part.s_train == 1, 0].mean()
...     hidden = part.X_train[(part.y_train == 1) & (part.s_train == 0), 0].mean()
...     print(mechanism, round(float(labeled), 1), round(float(hidden), 1))
scar 17.3 17.4
sar_propensity 19.6 15.2
```

{func}`~pypu.datasets.label_sar_propensity` can also return the true propensity $e(x)$
of every sample, to check how well a SAR method such as {class}`~pypu.models.SAREM`
recovers it. Its mean over the positives is the label frequency:

```pycon
>>> from pypu.datasets import label_sar_propensity
>>> s, propensity = label_sar_propensity(
...     X, y, features=[0], strength=2.0, return_propensity=True, random_state=0
... )
>>> round(float(propensity[y == 1].mean()), 2)
0.5
>>> round(float(propensity[y == 1].min()), 2), round(float(propensity[y == 1].max()), 2)
(0.02, 1.0)
```

{func}`~pypu.datasets.label_sar_knn` and {func}`~pypu.datasets.label_sar_cluster` compute
distances on the features as given, so standardize features with different units first.
`label_scar` and `label_sar_knn` take `legacy=True` to reproduce the labels of the
original iFPU and iFPNU experiments.

The example
[Which positives get labeled](../auto_examples/getting_started/plot_labeling_mechanisms.rst)
plots every mechanism.

## Label frequency or pollution

`label_frequency` sets the share $c$ of the positives that get labeled. Some studies
specify the data by the share of positives among the unlabeled samples instead, the
*pollution* $P(y = 1 \mid s = 0)$. {func}`~pypu.datasets.label_frequency_from_pollution`
converts one into the other, given the class prior:

```pycon
>>> from pypu.datasets import label_frequency_from_pollution
>>> prior = float(y.mean())
>>> c = label_frequency_from_pollution(0.1, prior=prior)  # 10% of the unlabeled
>>> round(c, 3)
0.813
>>> split = make_pu_split(X, y, label_frequency=c, random_state=0)
>>> round(pu_summary(split.y_train, split.s_train)["pollution"], 3)
0.101
```

{func}`~pypu.datasets.pollution_from_label_frequency` is the inverse. Both give expected
values; with finite data the realised shares differ slightly, because whole samples are
labeled.

## Case-control data

The labeling functions follow the single-training-set scenario: the labeled positives are
taken out of one sample, so the unlabeled samples have fewer positives than the
population. In the case-control scenario the unlabeled samples are a random sample of the
whole population, with its class prior (see
[Single training set and case-control](pu_learning.md#single-training-set-and-case-control)).
{func}`~pypu.datasets.make_case_control` makes such data. It keeps every positive and
drops some negatives, so that the unlabeled part has the prior of `y`:

```pycon
>>> from pypu.datasets import make_case_control
>>> X_cc, s_cc, y_cc = make_case_control(X, y, label_frequency=0.5, random_state=0)
>>> X_cc.shape  # some negatives were dropped
(391, 30)
>>> int(s_cc.sum()), int((s_cc == 0).sum())  # labeled, unlabeled
(106, 285)
>>> round(float(y_cc[s_cc == 0].mean()), 3), round(float(y.mean()), 3)  # prior of each
(0.372, 0.373)
```

It does not split the data: to keep a test set, split first, for example with
{func}`~sklearn.model_selection.train_test_split`, and make case-control data from the
training part.
