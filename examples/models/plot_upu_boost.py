"""
UPUBoost: gradient boosting on the unbiased PU risk
===================================================

When the labeled positives are chosen completely at random (SCAR), the loss on the true
labels can be estimated without bias from PU data: each labeled sample counts as a positive
with weight :math:`1 / c` and as a negative with weight :math:`1 - 1 / c`, and each
unlabeled sample as a negative (uPU, du Plessis et al.). Here :math:`c = P(s = 1) / \\pi`
is the label frequency, computed from the class prior :math:`\\pi`.
:class:`~pypu.models.UPUBoost` fits XGBoost trees to that risk.

The labeled samples' negative weight makes the risk unbounded below, so flexible models can
overfit it; the non-negative risk of :class:`~pypu.models.NNPUBoost` was designed against
this. It needs XGBoost: ``pip install "pypu-learn[xgboost]"``.
"""

# %%
# Data
# ----
# Two interleaving half circles with 40% positives; 30% of the training positives are
# labeled, chosen completely at random.

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU, UPUBoost

X, y = make_moons(n_samples=(600, 400), noise=0.25, random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# The baseline is XGBoost with the same settings (100 rounds, learning rate 0.3, no L2
# regularization), trained with the unlabeled samples as negatives.

models = {
    "naive XGBoost": NaivePU(XGBClassifier(n_estimators=100, learning_rate=0.3, reg_lambda=0)),
    "UPUBoost": UPUBoost(prior=prior, random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)

# %%
# Evaluate
# --------
# The risk estimates the loss on the true labels, so UPUBoost's probabilities estimate
# :math:`P(y = 1 \mid x)` and its threshold of 0.5 works.

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<13}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: the training data. Right: UPUBoost's probabilities and the decision boundaries.

fig, (ax_train, ax_test) = plt.subplots(
    1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
)
ax_train.scatter(*X_train[s_train == 0].T, s=10, c="0.75", label="unlabeled")
ax_train.scatter(*X_train[s_train == 1].T, s=10, c="tab:orange", label="labeled positive")
ax_train.set_title("Training data")
ax_train.legend(loc="lower right")

clf = models["UPUBoost"]
display = DecisionBoundaryDisplay.from_estimator(
    clf, X, response_method="predict_proba", cmap="Greys", alpha=0.5, ax=ax_test
)
fig.colorbar(display.surface_, ax=ax_test, label="UPUBoost: P(y = 1 | x)")
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
ax_test.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax_test.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
ax_test.set_title("Test data and decision boundaries")
ax_test.legend(loc="lower right")
plt.show()
