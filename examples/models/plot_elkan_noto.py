"""
ElkanNoto: correcting by the label frequency
============================================

When the labeled positives are chosen completely at random (SCAR), a classifier :math:`g`
that separates labeled from unlabeled samples estimates
:math:`P(s = 1 \\mid x) = c \\, P(y = 1 \\mid x)`, where :math:`c` is the label frequency,
the share of positives that are labeled. :class:`~pypu.models.ElkanNoto` estimates
:math:`c` as the mean score of the labeled samples, then either divides the scores by it
(the default) or, with ``weighted=True``, trains a new classifier in which each unlabeled
sample counts partly as a positive and partly as a negative.

It needs no class prior, but takes one if known (``prior=``), which gives :math:`c`
directly.
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
from pypu.models import ElkanNoto, NaivePU

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Fit
# ---
# Both forms estimate :math:`c` on out-of-fold scores, so that :math:`g` never scores the
# samples it was trained on.

models = {
    "NaivePU": NaivePU(),
    "ElkanNoto": ElkanNoto(random_state=0),
    "weighted": ElkanNoto(weighted=True, random_state=0),
}
for model in models.values():
    model.fit(X_train, s_train)
print(f"Estimated label frequency: {models['ElkanNoto'].label_frequency_:.3f} (true: 0.300)")

# %%
# Evaluate
# --------
# Dividing by :math:`c` keeps the ranking of the naive classifier and only moves its
# threshold. The weighted form trains another classifier, whose ranking can differ.

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<10}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: ElkanNoto's probabilities are the naive classifier's divided by :math:`c`, clipped
# at 1. Right: the decision boundaries on the test data.

fig, (ax_scale, ax_test) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
naive_proba = models["NaivePU"].predict_proba(X_test)[:, 1]
ax_scale.scatter(naive_proba, models["ElkanNoto"].predict_proba(X_test)[:, 1], s=10)
ax_scale.axhline(0.5, color="0.5", linewidth=0.8)
ax_scale.set_xlabel("NaivePU: P(s = 1 | x)")
ax_scale.set_ylabel("ElkanNoto: P(y = 1 | x)")
ax_scale.set_title(f"Scores divided by c = {models['ElkanNoto'].label_frequency_:.2f}")

ax_test.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax_test.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
for (name, model), style in zip(models.items(), ["--", "-", ":"], strict=True):
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
