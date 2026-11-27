"""
PUExtraTrees: extra trees grown on a PU risk
=============================================

:class:`~pypu.models.PUExtraTrees` grows extremely randomized trees whose splits minimize a
PU estimate of the classification risk, the non-negative PU risk (nnPU) by default. Each
leaf estimates the share of positives among its samples from the labeled and unlabeled
samples it holds and the class prior :math:`\\pi`. These estimates, averaged over the
trees, are probabilities of the positive class: ``predict`` can use them as they are.
"""

# %%
# Data
# ----
# Two interleaving half circles with 40% positives; 30% of the training positives are
# labeled, chosen completely at random.

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU, PUExtraTrees

X, y = make_moons(n_samples=(600, 400), noise=0.25, random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# The baseline is scikit-learn's extra trees, trained with the unlabeled samples as
# negatives. All three forests have 100 trees. The second PU forest minimizes the unbiased
# PU risk (uPU), which the nnPU risk corrects.

models = {
    "naive extra trees": NaivePU(ExtraTreesClassifier(random_state=0)),
    "PUExtraTrees": PUExtraTrees(prior=prior, random_state=0),
    "PUExtraTrees (uPU)": PUExtraTrees(prior=prior, risk_estimator="uPU", random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)

# %%
# Evaluate
# --------

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<18}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# With the nnPU risk, the forest's own labels are about as good as those from the prior.
# The naive forest predicts too few positives. The uPU risk can go below zero, so its trees
# keep splitting until each leaf holds only labeled or only unlabeled samples: the forest
# learns the labeled samples rather than the positives.
#
# The risk needs the share of positives among the unlabeled samples, which the forest
# derives from the prior and keeps as ``unlabeled_prior_``:

clf = models["PUExtraTrees"]
print(f"Prior {prior:.3f}, among the unlabeled samples {clf.unlabeled_prior_:.3f}")

# %%
# Plot
# ----
# The nnPU forest's probabilities, and their 0.5 contour, which ``predict`` uses.

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
fig.colorbar(display.surface_, ax=ax_test, label="P(y = 1)")
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict_proba",
    plot_method="contour",
    levels=[0.5],
    colors="k",
    ax=ax_test,
)
ax_test.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax_test.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
ax_test.plot([], [], "k", label="probability 0.5")
ax_test.set_title("Test data and PUExtraTrees' probabilities")
ax_test.legend(loc="lower right")
plt.show()
