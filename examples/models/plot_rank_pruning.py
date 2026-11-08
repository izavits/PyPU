"""
RankPruning: setting aside the unlabeled samples that look positive
===================================================================

:class:`~pypu.models.RankPruning` (Northcutt et al.) sees the unlabeled samples as negatives
with label noise, and removes the noise before training:

1. Out-of-fold probabilities of a classifier of labeled versus unlabeled samples estimate
   the noise: how many positives the unlabeled samples hold.
2. That many unlabeled samples, the ones with the highest probabilities, are *pruned*.
3. The classifier is refitted on the rest, with the labeled samples weighted up so that
   they count for all the positives.

It needs no class prior: it estimates the prior along the way.
"""

# %%
# Data
# ----
# Two Gaussian blobs with 40% positives; 30% of the training positives are labeled,
# chosen completely at random.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score

from pypu.datasets import make_pu_split, pu_summary
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU, RankPruning

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---

naive = NaivePU().fit(X_train, s_train)
clf = RankPruning(random_state=0).fit(X_train, s_train)

truth = pu_summary(y_train, s_train)
print(f"Class prior:              {clf.prior_:.3f} (true: {truth['prior']:.3f})")
print(f"Positives among unlabeled: {clf.pollution_:.3f} (true: {truth['pollution']:.3f})")
pruned = clf.pruned_
print(f"Pruned: {pruned.sum()}, of which truly positive: {y_train[pruned].sum()}")

# %%
# Evaluate
# --------

models = {"NaivePU": naive, "RankPruning": clf}
for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<11}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: the pruned unlabeled samples. Right: the decision boundaries on the test data.

fig, (ax_train, ax_test) = plt.subplots(
    1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
)
kept = (s_train == 0) & ~pruned
ax_train.scatter(*X_train[kept].T, s=10, c="0.75", label="unlabeled, kept")
ax_train.scatter(*X_train[pruned].T, s=10, c="tab:red", label="unlabeled, pruned")
ax_train.scatter(*X_train[s_train == 1].T, s=10, c="tab:orange", label="labeled positive")
ax_train.set_title("Training data")
ax_train.legend(loc="lower right")

ax_test.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax_test.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
for (name, model), style in zip(models.items(), ["--", "-"], strict=True):
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
