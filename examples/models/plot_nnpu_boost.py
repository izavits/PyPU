"""
NNPUBoost: the objective of the original nnPU experiments
=========================================================

The non-negative PU risk (nnPU, Kiryo et al.) fixes the overfitting of the unbiased risk of
:class:`~pypu.models.UPUBoost`, whose negative part can go below zero, by clipping that part
at zero. :class:`~pypu.models.NNPUBoost` ports the XGBoost objective of the original
experiments, which applies the clipping to each labeled sample. That always removes the
labeled samples' negative weight, so the objective is a weighted cross-entropy: each
labeled sample counts as a positive with weight :math:`1 / c`, and each unlabeled sample as
a negative with weight 1. PyPU reproduces it as run. It needs XGBoost:
``pip install "pypu-learn[xgboost]"``.
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
from xgboost import XGBClassifier

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU, NNPUBoost, UPUBoost

X, y = make_moons(n_samples=(600, 400), noise=0.25, random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# NNPUBoost against the naive XGBoost (unlabeled samples as negatives, weight 1 for all)
# and against UPUBoost.

models = {
    "naive XGBoost": NaivePU(XGBClassifier(n_estimators=100, learning_rate=0.3, reg_lambda=0)),
    "NNPUBoost": NNPUBoost(prior=prior, random_state=0),
    "UPUBoost": UPUBoost(prior=prior, random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)
c = s_train.mean() / prior
print(f"Weight of each labeled sample: 1 / c = {1 / c:.2f}")

# %%
# Evaluate
# --------
# Weighting the labeled samples by :math:`1 / c` lifts the probabilities only part of the
# way: without the negative weights of the unbiased risk, NNPUBoost's probabilities stay
# below :math:`P(y = 1 \mid x)`, and a threshold of 0.5 finds too few positives. On these
# data it does about as well as the naive XGBoost, while UPUBoost does better; a threshold
# from the prior turns its scores into much better predictions.

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<13}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

clf = models["NNPUBoost"]
threshold = np.quantile(clf.predict_proba(X_train)[:, 1], 1 - prior)
y_pred = clf.predict_proba(X_test)[:, 1] >= threshold
f1 = f1_score(y_test, y_pred)
print(f"NNPUBoost with the threshold {threshold:.3f} from the prior: F1 {f1:.3f}")

# %%
# Plot
# ----
# Left: the training data. Right: NNPUBoost's probabilities, its boundary at 0.5, and the
# boundary at the threshold from the prior.

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
fig.colorbar(display.surface_, ax=ax_test, label="NNPUBoost probability")
for level, style, label in [
    (0.5, "--", "threshold 0.5"),
    (threshold, "-", "threshold from the prior"),
]:
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict_proba",
        plot_method="contour",
        levels=[level],
        colors="k",
        linestyles=style,
        ax=ax_test,
    )
    ax_test.plot([], [], "k" + style, label=label)
ax_test.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax_test.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
ax_test.set_title("Test data and NNPUBoost")
ax_test.legend(loc="lower right")
plt.show()
