"""
PUHellingerTree: a decision tree for PU data
============================================

:class:`~pypu.models.PUHellingerTree` is a decision tree whose splits maximize the
Hellinger distance between the positive and the negative class, a criterion that is
insensitive to class imbalance. The classes are not observed in PU data, so each node's
positives are estimated from its labeled samples and the label frequency
:math:`c = P(s = 1) / \\pi`. It needs the class prior :math:`\\pi`.

A sample's score is the fraction of labeled samples in its leaf. It ranks the samples, but
it estimates the probability of being labeled, not of being positive, so it stays below 0.5:
use the scores to rank, or set a threshold from the prior.
"""

# %%
# Data
# ----
# Two interleaving half circles with 40% positives; 30% of the training positives are
# labeled, chosen completely at random.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU, PUHellingerTree

X, y = make_moons(n_samples=(600, 400), noise=0.25, random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# The baseline is scikit-learn's decision tree of the same depth, trained with the
# unlabeled samples as negatives.

models = {
    "naive tree": NaivePU(DecisionTreeClassifier(max_depth=5, random_state=0)),
    "PUHellingerTree": PUHellingerTree(prior=prior, max_depth=5, random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)

# %%
# Evaluate
# --------
# The Hellinger tree ranks the test samples better. Neither tree predicts many positives at
# a threshold of 0.5, but with a threshold set from the prior the Hellinger tree's scores do
# well:

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<15}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# The threshold can be set on the training data: predict positive the samples scoring at
# least as high as the top :math:`\pi` fraction of the training samples.

clf = models["PUHellingerTree"]
threshold = np.quantile(clf.predict_proba(X_train)[:, 1], 1 - prior)
y_pred = clf.predict_proba(X_test)[:, 1] >= threshold
print(f"Threshold {threshold:.3f}: F1 {f1_score(y_test, y_pred):.3f}")

# %%
# Depth matters: grown to full depth (the default), the tree splits until its leaves are
# pure in labeled samples, and overfits the PU labels.

for max_depth in [3, 5, 8, None]:
    tree = PUHellingerTree(prior=prior, max_depth=max_depth, random_state=0).fit(X_train, s_train)
    auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
    print(f"max_depth={max_depth}: ROC AUC {auc:.3f}")

# %%
# Plot
# ----
# Left: the training data. Right: the tree's scores, with the threshold set from the prior.

fig, (ax_train, ax_test) = plt.subplots(
    1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
)
ax_train.scatter(*X_train[s_train == 0].T, s=10, c="0.75", label="unlabeled")
ax_train.scatter(*X_train[s_train == 1].T, s=10, c="tab:orange", label="labeled positive")
ax_train.set_title("Training data")
ax_train.legend(loc="lower right")

display = DecisionBoundaryDisplay.from_estimator(
    clf, X, response_method="predict_proba", cmap="Greys", alpha=0.5, ax=ax_test
)
fig.colorbar(display.surface_, ax=ax_test, label="score (fraction labeled in the leaf)")
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict_proba",
    plot_method="contour",
    levels=[threshold],
    colors="k",
    ax=ax_test,
)
ax_test.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax_test.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
ax_test.plot([], [], "k", label="threshold from the prior")
ax_test.set_title("Test data and the tree's scores")
ax_test.legend(loc="lower right")
plt.show()
