"""
ImbalancedNNPUBoost: training for a larger class prior
======================================================

When positives are rare, a classifier can do well on average by neglecting them.
Imbalanced nnPU (Su et al.) trains as if the class prior were ``oversampled_prior``
(:math:`\\rho`, 0.5 by default) rather than the true, small prior :math:`\\pi`.
:class:`~pypu.models.ImbalancedNNPUBoost` ports the XGBoost objective of the original
experiments, a weighted cross-entropy: each labeled sample counts as a positive with weight
:math:`\\rho`, and each unlabeled sample as a negative with weight
:math:`(1 - \\rho) / (1 - \\pi)`. A larger :math:`\\rho` predicts more positives. It needs
XGBoost: ``pip install "pypu-learn[xgboost]"``.
"""

# %%
# Data
# ----
# Rare positives, like frauds among transactions: 10% of the samples, scattered around a
# dense cluster of negatives. Half of the training positives are labeled, chosen
# completely at random.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import ImbalancedNNPUBoost, NNPUBoost

X, y = make_blobs(
    n_samples=[1800, 200], centers=[[0, 0], [2.5, 2.5]], cluster_std=[0.8, 1.5], random_state=0
)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.5, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# With the default :math:`\rho = 0.5`, against :class:`~pypu.models.NNPUBoost`:

models = {
    "NNPUBoost": NNPUBoost(prior=prior, random_state=0),
    "ImbalancedNNPUBoost": ImbalancedNNPUBoost(prior=prior, random_state=0),
}
for name, model in models.items():
    model.fit(X_train, s_train)
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<19}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# The oversampled prior
# ---------------------
# :math:`\rho` trades precision for recall: the larger it is, the more weight the labeled
# positives get, and the more positives the model predicts.

rhos = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
precision, recall, f1 = [], [], []
for rho in rhos:
    model = ImbalancedNNPUBoost(prior=prior, oversampled_prior=rho, random_state=0)
    y_pred = model.fit(X_train, s_train).predict(X_test)
    precision.append(precision_score(y_test, y_pred, zero_division=0))
    recall.append(recall_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))
    print(
        f"rho = {rho:.2f}: precision {precision[-1]:.3f}  recall {recall[-1]:.3f}  F1 {f1[-1]:.3f}"
    )

# %%
# Plot
# ----

fig, (ax_train, ax_rho) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
ax_train.scatter(*X_train[s_train == 0].T, s=10, c="0.75", label="unlabeled")
ax_train.scatter(*X_train[s_train == 1].T, s=10, c="tab:orange", label="labeled positive")
ax_train.set_title("Training data")
ax_train.legend(loc="lower right")

ax_rho.plot(rhos, precision, "o-", label="precision")
ax_rho.plot(rhos, recall, "s-", label="recall")
ax_rho.plot(rhos, f1, "^-", label="F1")
ax_rho.axvline(prior, color="0.5", linestyle=":", label="true prior")
ax_rho.set_xlabel("oversampled_prior")
ax_rho.set_ylabel("Test score at threshold 0.5")
ax_rho.set_title("Effect of the oversampled prior")
ax_rho.legend()
plt.show()
