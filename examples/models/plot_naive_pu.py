"""
NaivePU: the baseline
=====================

:class:`~pypu.models.NaivePU` treats every unlabeled sample as a negative and trains an
ordinary classifier, a logistic regression by default. It is the baseline that PU learning
methods should beat.

The classifier learns :math:`P(s = 1 \\mid x)`, the probability of being *labeled*, instead
of :math:`P(y = 1 \\mid x)`. When the labeled positives are chosen completely at random
(SCAR), the two differ by the constant label frequency :math:`c`: the classifier ranks the
samples well, but its probabilities are too small, and the usual threshold of 0.5 predicts
hardly any positive. Given the class prior, NaivePU moves the threshold instead.
"""

# %%
# Data
# ----
# Two Gaussian blobs with 40% positives. 30% of the training positives are labeled,
# chosen completely at random; the rest of the samples are unlabeled.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()  # the class prior, 0.4

# %%
# Fit
# ---
# Without a prior, NaivePU predicts positive above a probability of 0.5. With the prior, it
# sets the threshold so that a ``prior`` fraction of the training samples is predicted
# positive.

naive = NaivePU().fit(X_train, s_train)
with_prior = NaivePU(prior=prior).fit(X_train, s_train)
print(f"Threshold without the prior: {naive.threshold_:.3f}")
print(f"Threshold with the prior:    {with_prior.threshold_:.3f}")

# %%
# Both models share the same classifier, so they rank the samples identically and have the
# same ROC AUC. Only the threshold differs:

models = {"NaivePU": naive, "NaivePU(prior)": with_prior}
for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<14}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: the predicted probabilities of the training samples. The positives' probabilities
# stay far below 0.5, because only 30% of the positives are labeled. Right: the two
# decision boundaries on the test data.

fig, (ax_hist, ax_test) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
proba = naive.predict_proba(X_train)[:, 1]
ax_hist.hist(proba[y_train == 0], bins=30, range=(0, 1), alpha=0.7, label="negatives")
ax_hist.hist(proba[y_train == 1], bins=30, range=(0, 1), alpha=0.7, label="positives")
for name, model, style in [("NaivePU", naive, "--"), ("NaivePU(prior)", with_prior, "-")]:
    ax_hist.axvline(model.threshold_, color="k", linestyle=style, label=f"{name} threshold")
ax_hist.set_xlabel("Predicted probability P(s = 1 | x)")
ax_hist.set_ylabel("Training samples")
ax_hist.set_title("The probabilities are too small")
ax_hist.legend()

ax_test.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax_test.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
for name, model, style in [("NaivePU", naive, "--"), ("NaivePU(prior)", with_prior, "-")]:
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        response_method="predict_proba",
        plot_method="contour",
        levels=[model.threshold_],
        colors="k",
        linestyles=style,
        ax=ax_test,
    )
    ax_test.plot([], [], "k" + style, label=f"{name} boundary")
ax_test.set_title("Test data and decision boundaries")
ax_test.legend(loc="lower right")
plt.show()
