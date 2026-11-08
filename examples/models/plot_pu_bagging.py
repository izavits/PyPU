"""
PUBagging: finding the hidden positives
=======================================

:class:`~pypu.models.PUBagging` (Mordelet and Vert) trains many classifiers, each on all the
labeled positives against a small bootstrap sample of the unlabeled samples taken as
negatives. A small sample holds few hidden positives, and averaging over the bags smooths
out the ones it does hold.

With ``oob_proba=True`` it also scores each unlabeled *training* sample with the
classifiers whose bag left it out. Ranking the unlabeled samples by these out-of-bag scores
is a common use: it points out the hidden positives, for example the genes most likely to
be disease genes.
"""

# %%
# Data
# ----
# Two Gaussian blobs with 40% positives; 30% of the training positives are labeled,
# chosen completely at random.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU, PUBagging

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# 100 decision trees by default; each bag holds as many unlabeled samples as there are
# labeled positives.

naive = NaivePU().fit(X_train, s_train)
clf = PUBagging(oob_proba=True, random_state=0).fit(X_train, s_train)

# %%
# The hidden positives
# --------------------
# Among the unlabeled training samples, the ones with the highest out-of-bag scores are
# mostly hidden positives:

unlabeled = np.flatnonzero(s_train == 0)
oob_scores = clf.oob_proba_[unlabeled, 1]
n_hidden = y_train[unlabeled].sum()
top = unlabeled[np.argsort(-oob_scores)[:n_hidden]]
print(f"Hidden positives among the unlabeled samples: {n_hidden}")
print(f"Positives among the top {n_hidden} out-of-bag scores: {y_train[top].sum()}")

# %%
# Evaluate
# --------
# The averaged probabilities rank the samples, but are not calibrated estimates of
# :math:`P(y = 1 \mid x)`.

models = {"NaivePU": naive, "PUBagging": clf}
for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<10}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: the unlabeled training samples, colored by their out-of-bag scores. Right: the
# decision boundaries on the test data.

fig, (ax_train, ax_test) = plt.subplots(
    1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
)
points = ax_train.scatter(*X_train[unlabeled].T, s=10, c=oob_scores, cmap="viridis")
ax_train.scatter(
    *X_train[s_train == 1].T, s=12, c="tab:orange", edgecolors="k", label="labeled positive"
)
fig.colorbar(points, ax=ax_train, label="out-of-bag score")
ax_train.set_title("Unlabeled training samples")
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
