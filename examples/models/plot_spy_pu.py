"""
SpyPU: reliable negatives from spies
====================================

:class:`~pypu.models.SpyPU` (Liu et al.) is a two-step method. It first finds unlabeled
samples that are almost surely negative:

1. A share of the labeled positives, the *spies*, is hidden among the unlabeled samples.
2. A classifier separates the remaining labeled samples from the unlabeled ones.
3. Unlabeled samples that score below (almost) every spy are taken as *reliable
   negatives*: the spies show how low a positive can score.

A final classifier, Gaussian naive Bayes by default, is then trained on the labeled
positives against the reliable negatives. With ``n_em_iter``, EM refines it on all the
unlabeled samples (S-EM).
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

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU, SpyPU

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# 10% of the labeled positives become spies (``spy_ratio``); the reliable negatives score
# below the lowest spy (``spy_quantile=0``).

models = {
    "NaivePU": NaivePU(),
    "SpyPU": SpyPU(random_state=0),
    "S-EM": SpyPU(n_em_iter=5, random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)

spy = models["SpyPU"]
reliable = spy.reliable_negatives_
print(f"Spies: {len(spy.spies_)}")
n_negative = (y_train[reliable] == 0).sum()
print(f"Reliable negatives: {reliable.sum()}, of which truly negative: {n_negative}")

# %%
# Evaluate
# --------

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<8}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: the spies and the reliable negatives they found. Right: the decision boundaries on
# the test data.

fig, (ax_train, ax_test) = plt.subplots(
    1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
)
others = (s_train == 0) & ~reliable
ax_train.scatter(*X_train[others].T, s=10, c="0.75", label="unlabeled")
ax_train.scatter(*X_train[reliable].T, s=10, c="tab:blue", label="reliable negative")
ax_train.scatter(*X_train[s_train == 1].T, s=10, c="tab:orange", label="labeled positive")
ax_train.scatter(*X_train[spy.spies_].T, s=40, c="k", marker="x", label="spy")
ax_train.set_title("Training data")
ax_train.legend(loc="lower right")

ax_test.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax_test.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
for (name, model), style in zip(models.items(), ["--", "-", ":"], strict=True):
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
