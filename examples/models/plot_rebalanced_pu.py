"""
RebalancedPU: reweighting the labeled and unlabeled samples
===========================================================

:class:`~pypu.models.RebalancedPU` trains a classifier to separate labeled from unlabeled
samples, like the naive baseline, but with sample weights computed from the label frequency
:math:`c` (Elkan, after Bekker and Davis's survey). The weights make the classifier's
threshold of 0.5 fall where a classifier of the true labels would put it, so that ``predict``
needs no further correction. It needs the class prior, which gives :math:`c`.
"""

# %%
# Data
# ----
# Two Gaussian blobs with 40% positives; 30% of the training positives are labeled,
# chosen completely at random.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import NaivePU, RebalancedPU

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# With label frequency :math:`c` and threshold :math:`\tau = 0.5`, labeled samples get
# weight :math:`1 - c \tau` and unlabeled samples :math:`(1 - \tau) c`.

naive = NaivePU().fit(X_train, s_train)
clf = RebalancedPU(prior=prior).fit(X_train, s_train)
c = clf.label_frequency_
labeled_weight, unlabeled_weight = 1 - c * 0.5, (1 - 0.5) * c
print(f"Label frequency: {c:.3f}")
print(f"Weights: {labeled_weight:.3f} per labeled sample, {unlabeled_weight:.3f} per unlabeled")

# %%
# Evaluate
# --------

models = {"NaivePU": naive, "RebalancedPU": clf}
for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<12}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: the training samples, drawn in proportion to their weights. Right: the decision
# boundaries on the test data.

fig, (ax_train, ax_test) = plt.subplots(
    1, 2, figsize=(11, 4.6), sharex=True, sharey=True, layout="constrained"
)
ax_train.scatter(*X_train[s_train == 0].T, s=40 * unlabeled_weight, c="0.6", label="unlabeled")
ax_train.scatter(
    *X_train[s_train == 1].T, s=40 * labeled_weight, c="tab:orange", label="labeled positive"
)
ax_train.set_title("Training data, sized by weight")
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
