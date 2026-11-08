"""
IFPNU: reliable negatives from an anomaly detector
==================================================

:class:`~pypu.models.IFPNU` turns PU data into PNU data (positive, negative and unlabeled)
before training, assuming that positives are rare and look anomalous:

1. An anomaly detector, scikit-learn's ``IsolationForest`` by default, scores every
   training sample. The :math:`\\pi n` most anomalous samples might be positive: the
   unlabeled ones among them stay unlabeled. Every other unlabeled sample becomes a
   *reliable negative*.
2. LightGBM trees are trained on a focal loss over the labeled positives, the reliable
   negatives and the remaining unlabeled samples; ``eta`` sets how much the reliable
   negatives count against the unlabeled samples.

Any detector with ``score_samples`` works, and so do the detectors of PyOD. Its defaults
are the settings of the original iFPNU experiments (100 trees). It needs the class prior.
"""

# %%
# Data
# ----
# Rare positives, like frauds among transactions: 10% of the samples, scattered around a
# dense cluster of negatives. Half of the training positives are labeled, chosen
# completely at random.

import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import IFPNU, NaivePU

X, y = make_blobs(
    n_samples=[1800, 200], centers=[[0, 0], [2.5, 2.5]], cluster_std=[0.8, 1.5], random_state=0
)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.5, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# The baseline is LightGBM with the same 100 trees, trained with the unlabeled samples as
# negatives. With the default ``eta=0.5`` the reliable negatives count about as much as the
# other unlabeled samples; ``eta=0.9`` leans on them.

models = {
    "naive LightGBM": NaivePU(LGBMClassifier(verbose=-1)),
    "IFPNU": IFPNU(prior=prior, random_state=0),
    "IFPNU(eta=0.9)": IFPNU(prior=prior, eta=0.9, random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)

clf = models["IFPNU"]
reliable = clf.pnu_labels_ == 0
n_negative = (y_train[reliable] == 0).sum()
print(f"Reliable negatives: {reliable.sum()}, of which truly negative: {n_negative}")
print(f"Left unlabeled: {(clf.pnu_labels_ == -1).sum()}")

# %%
# Evaluate
# --------
# All three rank the samples almost perfectly. At a threshold of 0.5 they predict too few
# positives, IFPNU with the default ``eta`` the fewest, since the focal loss pushes the
# unlabeled samples' probabilities down; leaning on the reliable negatives helps, and a
# threshold from the prior helps most (F1 at prior).

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<15}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: the PNU labels of the first step. The anomalies that stay unlabeled are mostly the
# hidden positives, with some negatives at the edge of their cluster. Right: the
# probabilities of IFPNU with ``eta=0.9`` on the test data.

fig, (ax_train, ax_test) = plt.subplots(
    1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
)
labels = clf.pnu_labels_
ax_train.scatter(*X_train[labels == 0].T, s=10, c="tab:blue", label="reliable negative")
ax_train.scatter(*X_train[labels == -1].T, s=10, c="0.6", label="anomalous, left unlabeled")
ax_train.scatter(*X_train[labels == 1].T, s=10, c="tab:orange", label="labeled positive")
ax_train.set_title("Training data after the first step")
ax_train.legend(loc="lower right")

display = DecisionBoundaryDisplay.from_estimator(
    models["IFPNU(eta=0.9)"],
    X,
    response_method="predict_proba",
    cmap="Greys",
    alpha=0.5,
    ax=ax_test,
)
fig.colorbar(display.surface_, ax=ax_test, label="IFPNU(eta=0.9) probability")
ax_test.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax_test.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
ax_test.set_title("Test data and IFPNU's probabilities")
ax_test.legend(loc="lower right")
plt.show()
