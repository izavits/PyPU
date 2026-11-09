% test-requires: cvxopt

# Class-prior estimation

Many PU models need the class prior $\pi = P(y = 1)$ (see
[Choosing a model](models.md)). Sometimes it is known from the domain, such as the
prevalence of a disease. Otherwise it can be estimated from the PU data, with
{mod}`pypu.prior`.

In the single-training-set scenario the prior and the label frequency $c$ are tied: the
share of labeled samples is their product, $P(s = 1) = c \, \pi$. Estimating one gives
the other, and also the pollution, the share of positives among the unlabeled samples.
Every estimator in {mod}`pypu.prior` estimates all three.

## What the estimates assume

The PU labels alone cannot tell how many unlabeled samples are positive: the unlabeled
set could be any mixture. The estimators therefore assume:

- **SCAR labels.** The labeled positives are a random sample of the positives, so they
  show where the positives lie.
- **A region of (almost) only positives.** Somewhere in the feature space the positives
  are not mixed with negatives. Where the classes overlap, an estimator cannot tell how
  many unlabeled samples there are positive, and the estimate is biased upwards.

## The estimators

| Estimator | How it works | Notes |
| --- | --- | --- |
| {class}`~pypu.prior.BBEPrior` | Ranks the samples with a classifier of labeled vs. unlabeled samples, and finds a top bin of mostly positive samples [^bbe] | Copes with overlapping classes; the scores need only rank well |
| {class}`~pypu.prior.ElkanNotoPrior` | Estimates $c$ from the scores of the same kind of classifier on the labeled samples: their mean (`"e1"`), a ratio of sums (`"e2"`) or the maximum (`"e3"`) [^elkan] | Accurate when the classes barely overlap; `"e1"` and `"e2"` overestimate the prior when they overlap more |
| {class}`~pypu.prior.KMPrior` | Embeds the labeled and the unlabeled distributions with a kernel, without a classifier (`"km1"` or `"km2"`) [^km] | Needs cvxopt; memory grows with the square of the number of samples, so subsample beyond a few thousand |

All three are fitted like a model, `fit(X, s)`, and then hold `prior_`,
`label_frequency_` and `pollution_`. They use a logistic regression or distances by
default, so scale the features first:

```pycon
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.preprocessing import StandardScaler
>>> from pypu.datasets import make_pu_split
>>> from pypu.prior import BBEPrior, ElkanNotoPrior, KMPrior
>>> data = load_breast_cancer()
>>> X, y = data.data, 1 - data.target
>>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, random_state=0)
>>> scaler = StandardScaler().fit(X_train)
>>> X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
>>> round(float(y_train.mean()), 2)  # the true prior, hidden from the estimators
0.37
>>> bbe = BBEPrior(random_state=0).fit(X_train, s_train)
>>> round(bbe.prior_, 2), round(bbe.label_frequency_, 2), round(bbe.pollution_, 2)
(0.4, 0.46, 0.27)
>>> round(ElkanNotoPrior(random_state=0).fit(X_train, s_train).prior_, 2)
0.47
>>> round(KMPrior().fit(X_train, s_train).prior_, 2)
0.41
```

All three overestimate the prior a little here, as the classes overlap somewhat; the
estimate of Elkan and Noto most.

BBE is only as good as the ranking of its classifier. When the positives are not linearly
separable from the rest, a more flexible classifier reduces its bias, at a higher cost:

```pycon
>>> from sklearn.ensemble import HistGradientBoostingClassifier
>>> hgb = HistGradientBoostingClassifier(random_state=0)
>>> flexible = BBEPrior(estimator=hgb, random_state=0).fit(X_train, s_train)
>>> round(flexible.prior_, 2)
0.38
```

## Using the estimate

Pass the estimate to a model as its `prior`. A small error matters little: here
{class}`~pypu.models.RebalancedPU` classifies the test set about as well with the
estimate as with the true prior:

```pycon
>>> from sklearn.metrics import f1_score
>>> from pypu.models import RebalancedPU
>>> for prior in [float(y_train.mean()), bbe.prior_]:  # true, estimated
...     clf = RebalancedPU(prior=prior).fit(X_train, s_train)
...     print(round(prior, 2), round(f1_score(y_test, clf.predict(X_test)), 2))
0.37 0.85
0.4 0.88
```

Some models estimate what they need themselves when no prior is given:
{class}`~pypu.models.ElkanNoto` and {class}`~pypu.models.RankPruning` estimate the label
frequency, and {class}`~pypu.models.PUe` estimates the prior with KM1.

## When the labels are not SCAR

Under SAR the labeled positives show only part of where the positives lie, and the
estimates go wrong. With labels that favour the larger tumours, both estimates fall far
below the true prior:

```pycon
>>> X_sar, _, s_sar, y_sar, _ = make_pu_split(
...     X, y, mechanism="sar_propensity", features=[0], strength=2.0, random_state=0
... )
>>> X_sar = StandardScaler().fit_transform(X_sar)
>>> round(float(y_sar.mean()), 2)
0.37
>>> round(BBEPrior(random_state=0).fit(X_sar, s_sar).prior_, 2)
0.22
>>> round(ElkanNotoPrior(random_state=0).fit(X_sar, s_sar).prior_, 2)
0.24
```

If the labels may be biased, prefer a prior from the domain, or a SAR model that needs
none, such as {class}`~pypu.models.SAREM`.

## Tips

- With few labeled positives the estimates are noisy. Refit with other `random_state`
  values, or on bootstrap samples, to see how much they vary.
- A known prevalence from the domain usually beats an estimate.
- The example
  [Estimating the class prior](../auto_examples/getting_started/plot_prior_estimation.rst)
  compares the estimators on separated and overlapping classes, for several label
  frequencies.

## References

[^bbe]: S. Garg, Y. Wu, A. J. Smola, S. Balakrishnan and Z. C. Lipton. "Mixture
    proportion estimation and PU learning: a modern approach." NeurIPS, 2021.

[^elkan]: C. Elkan and K. Noto. "Learning classifiers from only positive and unlabeled
    data." KDD, 2008.

[^km]: H. Ramaswamy, C. Scott and A. Tewari. "Mixture proportion estimation via kernel
    embedding of distributions." ICML, 2016.
