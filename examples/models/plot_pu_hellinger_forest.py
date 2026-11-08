"""
PUHellingerForest: a random forest for PU data
==============================================

:class:`~pypu.models.PUHellingerForest` is a random forest of
:class:`~pypu.models.PUHellingerTree`. Each tree is grown on all the labeled positives and
a bootstrap sample of the unlabeled samples, drawing a random subset of the features at each
split. A sample's score is the mean, over the trees, of the fraction of labeled samples in
its leaf. Like the single tree's, it ranks the samples but stays below 0.5; set a threshold
from the class prior :math:`\\pi`, which the forest needs anyway.
"""

# %%
# Data
# ----
# Two interleaving half circles with 40% positives; 30% of the training positives are
# labeled, chosen completely at random.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU, PUHellingerForest

X, y = make_moons(n_samples=(600, 400), noise=0.25, random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# The baseline is scikit-learn's random forest, trained with the unlabeled samples as
# negatives. Both have 100 trees.

models = {
    "naive forest": NaivePU(RandomForestClassifier(random_state=0)),
    "PUHellingerForest": PUHellingerForest(prior=prior, random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)

# %%
# Evaluate
# --------

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<17}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# A threshold from the prior, set on the training data, turns the scores into predictions:

clf = models["PUHellingerForest"]
threshold = np.quantile(clf.predict_proba(X_train)[:, 1], 1 - prior)
y_pred = clf.predict_proba(X_test)[:, 1] >= threshold
print(f"Threshold {threshold:.3f}: F1 {f1_score(y_test, y_pred):.3f}")

# %%
# Plot
# ----
# The forest's scores are smoother than a single tree's.

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
fig.colorbar(display.surface_, ax=ax_test, label="score")
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
ax_test.set_title("Test data and the forest's scores")
ax_test.legend(loc="lower right")
plt.show()
