"""
IFPU: focal-loss gradient boosting for rare positives
=====================================================

:class:`~pypu.models.IFPU` trains LightGBM trees on a PU focal loss. Each unlabeled sample
is treated as a positive with probability :math:`\\pi`, the class prior, and as a negative
otherwise; the focal loss (``gamma``) puts more weight on the samples the model gets
wrong, which helps when positives are rare. Its defaults are the settings of the original
iFPU experiments (10 trees, learning rate 0.1, ``gamma=3``). It needs the class prior.
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
from pypu.models import IFPU, NaivePU

X, y = make_blobs(
    n_samples=[1800, 200], centers=[[0, 0], [2.5, 2.5]], cluster_std=[0.8, 1.5], random_state=0
)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.5, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# The baseline is LightGBM with the same 10 trees, trained with the unlabeled samples as
# negatives.

models = {
    "naive LightGBM": NaivePU(LGBMClassifier(n_estimators=10, verbose=-1)),
    "IFPU": IFPU(prior=prior, random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)

# %%
# Evaluate
# --------
# Both rank the samples well. At a threshold of 0.5 the naive model predicts almost no
# positive, while IFPU, whose loss accounts for the hidden positives, predicts well without
# any threshold tuning.

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
# Left: the training data. Right: IFPU's probabilities and the decision boundaries.

fig, (ax_train, ax_test) = plt.subplots(
    1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
)
ax_train.scatter(*X_train[s_train == 0].T, s=10, c="0.75", label="unlabeled")
ax_train.scatter(*X_train[s_train == 1].T, s=10, c="tab:orange", label="labeled positive")
ax_train.set_title("Training data")
ax_train.legend(loc="lower right")

clf = models["IFPU"]
display = DecisionBoundaryDisplay.from_estimator(
    clf, X, response_method="predict_proba", cmap="Greys", alpha=0.5, ax=ax_test
)
fig.colorbar(display.surface_, ax=ax_test, label="IFPU probability")
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
