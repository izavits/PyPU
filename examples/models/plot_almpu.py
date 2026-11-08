"""
ALMPU: a neural network trained with an augmented Lagrangian
============================================================

:class:`~pypu.models.ALMPU` (Wei et al.) trains a small neural network (an MLP) in two
phases:

1. It learns the labeled positives while keeping the loss of the unlabeled samples, taken
   as negatives, under a bound: an augmented-Lagrangian constraint. Meanwhile it records
   how each unlabeled sample's predicted probability changes from epoch to epoch.
2. Unlabeled samples whose probability keeps rising are pseudo-labeled positive (split by
   Jenks natural breaks), and a new network is trained towards these pseudo-labels.

It needs no class prior. PyPU ports the code of the original tabular experiments, with
their settings; it needs PyTorch: ``pip install "pypu-learn[torch]"``.
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
from sklearn.neural_network import MLPClassifier

from pypu.datasets import make_pu_split, pu_summary
from pypu.metrics import f1_at_prior
from pypu.models import ALMPU, NaivePU

X, y = make_moons(n_samples=(600, 400), noise=0.25, random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# The baseline is scikit-learn's MLP with the same hidden layers, trained with the
# unlabeled samples as negatives. ALMPU trains for 50 epochs of 100 steps.

models = {
    "naive MLP": NaivePU(
        MLPClassifier(hidden_layer_sizes=(16, 18, 20, 24), max_iter=2000, random_state=0)
    ),
    "ALMPU": ALMPU(random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)

# %%
# The share of unlabeled samples pseudo-labeled positive, ``pollution_``, should be close to
# the true share of positives among them. When it is far off, the pseudo-labels went wrong,
# which happens on some small datasets: distrust the model then.

clf = models["ALMPU"]
print(f"Pseudo-labeled positive: {clf.pollution_:.3f} of the unlabeled samples")
print(f"Truly positive:          {pu_summary(y_train, s_train)['pollution']:.3f}")

# %%
# Evaluate
# --------

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<9}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: the pseudo-labels of the unlabeled training samples. Right: ALMPU's probabilities
# and the decision boundaries.

fig, (ax_train, ax_test) = plt.subplots(
    1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
)
unlabeled = X_train[s_train == 0]
pseudo = clf.pseudo_labels_ == 1
ax_train.scatter(*unlabeled[~pseudo].T, s=10, c="0.75", label="pseudo-negative")
ax_train.scatter(*unlabeled[pseudo].T, s=10, c="tab:green", label="pseudo-positive")
ax_train.scatter(*X_train[s_train == 1].T, s=10, c="tab:orange", label="labeled positive")
ax_train.set_title("Pseudo-labels of the unlabeled samples")
ax_train.legend(loc="lower right")

display = DecisionBoundaryDisplay.from_estimator(
    clf, X, response_method="predict_proba", cmap="Greys", alpha=0.5, ax=ax_test
)
fig.colorbar(display.surface_, ax=ax_test, label="ALMPU probability")
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
