"""
Getting started with PU learning
================================

In positive-unlabeled (PU) learning, the training data hold some samples labeled positive
and many unlabeled samples, which are a mix of positives and negatives. No sample is labeled
negative.

This example makes PU data from a fully labeled dataset, shows what goes wrong when the
unlabeled samples are simply taken as negatives, and fixes it with a PU learning method.
Every model in :mod:`pypu.models` is used the same way: ``fit(X, s)``, then
``predict_proba``, ``decision_function`` or ``predict``.
"""

# %%
# PU data from a labeled dataset
# ------------------------------
# The examples start from a fully labeled dataset, so that the models can be checked
# against the true labels ``y``. :func:`~pypu.datasets.make_pu_split` splits it into a
# training set and a test set, and hides the labels of most training positives: here 30% of
# them stay labeled, chosen completely at random (the SCAR assumption). The PU labels ``s``
# are 1 for a labeled positive and 0 for an unlabeled sample.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score

from pypu.datasets import make_pu_split, pu_summary
from pypu.metrics import f1_at_prior
from pypu.models import ElkanNoto, NaivePU

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
for key, value in pu_summary(y_train, s_train).items():
    print(f"{key:>16}: {value:.3g}")

# %%
# The class prior :math:`\pi` is the share of positives (40%), the label frequency
# :math:`c` the share of positives that are labeled (30%), and the pollution the share of
# positives among the unlabeled samples (32%). A model sees only ``X_train`` and
# ``s_train``:

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True, layout="constrained")
axes[0].scatter(*X_train[y_train == 0].T, s=10, c="tab:blue", label="negative")
axes[0].scatter(*X_train[y_train == 1].T, s=10, c="tab:orange", label="positive")
axes[0].set_title("True labels y (hidden)")
axes[1].scatter(*X_train[s_train == 0].T, s=10, c="0.75", label="unlabeled")
axes[1].scatter(*X_train[s_train == 1].T, s=10, c="tab:orange", label="labeled positive")
axes[1].set_title("PU labels s (what a model sees)")
for ax in axes:
    ax.legend(loc="lower right")
plt.show()

# %%
# The naive classifier
# --------------------
# :class:`~pypu.models.NaivePU` trains a logistic regression with the unlabeled samples as
# negatives. It learns :math:`P(s = 1 \mid x)` instead of :math:`P(y = 1 \mid x)`: under
# SCAR the two differ by the factor :math:`c`, so its probabilities are about 0.3 times too
# small and it predicts hardly any positive.

naive = NaivePU().fit(X_train, s_train)
print(f"Share predicted positive: {naive.predict(X_test).mean():.1%} (true: 40%)")

# %%
# A PU learning method
# --------------------
# :class:`~pypu.models.ElkanNoto` corrects this. It trains the same classifier, estimates
# the label frequency :math:`c` from the labeled samples' scores, and divides the
# probabilities by it.

clf = ElkanNoto(random_state=0).fit(X_train, s_train)
print(f"Estimated label frequency: {clf.label_frequency_:.3f} (true: 0.300)")
print(f"Share predicted positive: {clf.predict(X_test).mean():.1%} (true: 40%)")

# %%
# Evaluation
# ----------
# The examples report three metrics on the test set:
#
# - **ROC AUC**, which only depends on how the scores rank the samples.
# - **F1** of the model's own predictions, ``predict``.
# - **F1 at the prior**, :func:`~pypu.metrics.f1_at_prior`: F1 when the top :math:`\pi`
#   fraction of the scores is predicted positive. It shows how good the scores would be
#   with a threshold set from the class prior.
#
# Under SCAR the naive classifier ranks the samples as well as the PU method, since its
# probabilities are only scaled down. Its problem is the threshold, which ElkanNoto fixes:

prior = y_train.mean()
for name, model in {"NaivePU": naive, "ElkanNoto": clf}.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<10}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# The decision boundaries show the same thing: the naive one sits deep inside the positive
# class.

fig, ax = plt.subplots(figsize=(5.5, 4.6), layout="constrained")
ax.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
for name, model, style in [("NaivePU", naive, "--"), ("ElkanNoto", clf, "-")]:
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict_proba",
        plot_method="contour",
        levels=[model.threshold_],
        colors="k",
        linestyles=style,
        ax=ax,
    )
    ax.plot([], [], "k" + style, label=f"{name} boundary")
ax.set_title("Test data")
ax.legend(loc="lower right")
plt.show()

# %%
# Next steps
# ----------
# - The other getting-started examples show the labeling mechanisms, how to estimate the
#   class prior, and how to tune a model without true labels.
# - The examples in ``examples/models`` show each model of :mod:`pypu.models` on data like
#   these, and ``examples/comparison/plot_compare_methods.py`` compares them all on a real
#   dataset.
