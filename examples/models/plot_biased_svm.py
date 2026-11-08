"""
BiasedSVM: an SVM that trusts the labeled samples more
======================================================

:class:`~pypu.models.BiasedSVM` (Liu et al.) trains an SVM to separate labeled from
unlabeled samples, but penalizes errors on the labeled positives (weight :math:`w`) more
than errors on the unlabeled samples (weight :math:`1 - w`), since some unlabeled samples
are positives. The weight :math:`w` is chosen by cross-validation with Lee and Liu's
criterion (:func:`~pypu.metrics.lee_liu_score`), which needs no true labels and no class
prior.
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
from pypu.models import BiasedSVM

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# An SVM with equal weights (:math:`w = 0.5`) is the naive baseline. The biased SVM tries
# the default weights from 0.7 to 0.95, with an RBF kernel.

models = {
    "unbiased SVM": BiasedSVM(positive_weights=[0.5], random_state=0),
    "BiasedSVM": BiasedSVM(random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)

clf = models["BiasedSVM"]
for weight, score in zip(clf.positive_weights, clf.cv_scores_, strict=True):
    print(f"w = {weight:.2f}: Lee-Liu score {score:.3f}")
print(f"Chosen weight: {clf.positive_weight_}")

# %%
# Evaluate
# --------
# ``predict_proba`` is the sigmoid of the SVM's decision function, so it ranks the samples
# but is not a calibrated probability.

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
# Left: Lee and Liu's criterion for each weight. Right: the decision boundaries on the test
# data.

fig, (ax_cv, ax_test) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
ax_cv.plot(clf.positive_weights, clf.cv_scores_, "o-")
ax_cv.axvline(clf.positive_weight_, color="k", linestyle=":", label="chosen")
ax_cv.set_xlabel("Weight w of the labeled samples")
ax_cv.set_ylabel("Lee-Liu score (cross-validated)")
ax_cv.set_title("Choosing the weight without true labels")
ax_cv.legend()

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
