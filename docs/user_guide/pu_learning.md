# PU learning

## The setting

In ordinary binary classification every training sample has a label $y$: 1 for positive,
0 for negative. In *positive-unlabeled* (PU) learning only some positives are labeled.
Each sample has a PU label $s$ instead:

- $s = 1$: a **labeled** sample, which is always positive;
- $s = 0$: an **unlabeled** sample, which may be positive or negative.

No sample is labeled negative. The goal is still a classifier for $y$, learned from the
features $x$ and the PU labels $s$.

PU data arise whenever only one class gets recorded: confirmed frauds among all
transactions, known disease genes among all genes, diagnosed patients among everyone,
pages a user bookmarked among all pages they saw. A transaction that was never flagged
may still be fraudulent; it is unlabeled, not negative.

Every PyPU model is fitted on the features and the PU labels, `fit(X, s)`. To try methods
out, {mod}`pypu.datasets` turns a fully labeled dataset into PU data, so that the results
can be checked against the hidden true labels:

```pycon
>>> from sklearn.datasets import make_classification
>>> from pypu.datasets import label_scar, pu_summary
>>> X, y = make_classification(n_samples=1000, weights=[0.7], random_state=0)
>>> s = label_scar(y, label_frequency=0.4, random_state=0)  # label 40% of the positives
>>> pu_summary(y, s)
{'n_samples': 1000, 'n_positive': 302, 'n_labeled': 121, 'n_unlabeled': 879,
 'prior': 0.302, 'label_frequency': 0.40066225165562913,
 'pollution': 0.20591581342434584}
```

[Making PU data](datasets.md) covers these tools.

## The class prior and the label frequency

Two quantities describe PU data, and many methods need one of them:

- the **class prior** $\pi = P(y = 1)$, the share of positives among all samples;
- the **label frequency** $c = P(s = 1 \mid y = 1)$, the share of positives that are
  labeled.

When the labeled samples are taken from one sample of the population (the
*single-training-set* scenario, as above), a fraction $c\pi$ of the samples is labeled,
and the unlabeled samples still contain positives, a share

$$
P(y = 1 \mid s = 0) = \frac{\pi (1 - c)}{1 - \pi c}
$$

of them (the `pollution` above: 0.21). Neither $\pi$ nor $c$ can be read off the PU
labels: the number of labeled samples only gives their product. When neither is known,
[estimate the class prior](prior.md) from the data.

## Which positives get labeled: SCAR and SAR

Most PU methods assume that the labeled positives are a random sample of all positives:
the labels are **selected completely at random** (SCAR). Every positive then has the same
chance $c$ of being labeled, whatever its features:

$$
e(x) = P(s = 1 \mid x, y = 1) = c .
$$

$e(x)$ is the **propensity** of a positive with features $x$. Often it is not constant:
severe cases are diagnosed more often than mild ones, and well-known genes are studied
more than obscure ones. The labels are then **selected at random** given the features
(SAR), and the labeled positives are a biased sample of the positives. Methods that
assume SCAR can go wrong on such data; {class}`~pypu.models.SAREM`,
{class}`~pypu.models.PropensityWeightedPU`, {class}`~pypu.models.LBE` and
{class}`~pypu.models.PUe` model the propensity instead. The example
[Which positives get labeled](../auto_examples/getting_started/plot_labeling_mechanisms.rst)
shows both kinds of labels.

## Why the naive classifier is biased

The simplest approach treats every unlabeled sample as negative and trains an ordinary
classifier: {class}`~pypu.models.NaivePU`. It learns $P(s = 1 \mid x)$, the probability of
being *labeled*, rather than $P(y = 1 \mid x)$. Under SCAR the two differ only by the
label frequency [^elkan]:

$$
P(s = 1 \mid x) = P(y = 1 \mid x) \, P(s = 1 \mid x, y = 1) = c \, P(y = 1 \mid x) .
$$

So the naive classifier ranks samples in the right order, but its probabilities are too
small by a factor $c$, and with the usual threshold of 0.5 it predicts far too few
positives. {class}`~pypu.models.ElkanNoto` divides by an estimate of $c$, which keeps the
ranking and fixes the predictions:

```pycon
>>> from sklearn.metrics import roc_auc_score
>>> from pypu.datasets import make_pu_split
>>> from pypu.models import ElkanNoto, NaivePU
>>> X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.4, random_state=0)
>>> naive = NaivePU().fit(X_train, s_train)
>>> corrected = ElkanNoto(random_state=0).fit(X_train, s_train)
>>> round(float(y_test.mean()), 2)  # share of positives in the test set
0.3
>>> round(float(naive.predict(X_test).mean()), 2)  # share predicted positive
0.05
>>> round(float(corrected.predict(X_test).mean()), 2)
0.25
>>> round(roc_auc_score(y_test, naive.decision_function(X_test)), 3)  # same ranking
0.975
>>> round(roc_auc_score(y_test, corrected.decision_function(X_test)), 3)
0.975
```

Other methods correct the bias in other ways: by reweighting the samples
({class}`~pypu.models.RebalancedPU`), by first finding unlabeled samples that are reliably
negative ({class}`~pypu.models.SpyPU`), or by minimizing an unbiased estimate of the risk
on the true labels ({class}`~pypu.models.UPUBoost`). Under SAR the factor is $e(x)$
instead of $c$, so even the ranking of the naive classifier is distorted.
[Choosing a model](models.md) compares the methods.

## Single training set and case-control

PU data come in two scenarios [^bekker]:

- **Single training set:** one sample of the population, in which some positives are
  labeled. The unlabeled samples have fewer positives than the population.
  {func}`~pypu.datasets.make_pu_split` and the labeling functions make such data.
- **Case-control:** the labeled positives and the unlabeled samples are drawn separately,
  and the unlabeled samples are a random sample of the population, with its class prior
  $\pi$. {func}`~pypu.datasets.make_case_control` makes such data.

The two relate the class prior and the label frequency differently. In the
single-training-set scenario a fraction $c\pi$ of all samples is labeled; in the
case-control scenario the number of labeled samples is chosen separately, and the
unlabeled samples alone have the population's class prior. The rest of this guide, like
the examples, uses the single-training-set scenario.

## What a fitted model returns

Every model has the same methods:

- `predict_proba(X)` returns the probability of each class; column 1 is the probability
  of the positive class. Some methods give calibrated probabilities, others only scores
  that rank well; [Choosing a model](models.md) says which.
- `decision_function(X)` returns one score per sample: higher means more likely positive.
- `predict(X)` returns labels in $\{0, 1\}$: a sample is positive when its positive-class
  probability is above the model's `threshold_`, which is 0.5 for most models.

When a model's scores rank well but are not calibrated, a threshold from the class prior
is often better than 0.5: predict the top $\pi$ fraction of the samples positive, with
{func}`~pypu.utils.threshold_from_prior`. {class}`~pypu.models.NaivePU` does this itself
when given a `prior`. [Evaluation and model selection](evaluation.md)
covers this and other ways to evaluate PU models.

## References

[^elkan]: C. Elkan and K. Noto. "Learning classifiers from only positive and unlabeled
    data." KDD, 2008.

[^bekker]: J. Bekker and J. Davis. "Learning from positive and unlabeled data: a survey."
    Machine Learning, 2020.
