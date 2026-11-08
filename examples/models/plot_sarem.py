"""
SAREM: when the labels are not chosen at random
===============================================

Most PU methods assume that the labeled positives are chosen completely at random (SCAR).
Often they are not: a patient who sees a doctor often is more likely to be diagnosed, but
not more likely to be ill. The probability that a positive gets labeled, the *propensity*
:math:`e(x) = P(s = 1 \\mid y = 1, x)`, then depends on the sample (SAR).

:class:`~pypu.models.SAREM` (Bekker et al.) learns the classifier and the propensity
together, by expectation-maximization. It needs no class prior, and works best when told
which features the propensity depends on (``propensity_features``).
"""

# %%
# SAR data
# --------
# Two features decide the class: two Gaussian blobs with 40% positives. A third feature,
# unrelated to the class, decides which positives get labeled: think of it as how often
# a person sees a doctor.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pypu.metrics import f1_at_prior
from pypu.models import SAREM, ElkanNoto, NaivePU

rng = np.random.default_rng(0)
X, y = make_blobs(n_samples=[1200, 800], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X = np.column_stack([X, rng.normal(size=len(X))])


def propensity(X):
    """Return the probability that a positive gets labeled: it grows with feature 2."""
    return expit(2.5 * X[:, 2] - 1.2)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
s_train = ((y_train == 1) & (rng.random(len(y_train)) < propensity(X_train))).astype(int)
prior = y_train.mean()
print(f"Labeled positives: {s_train.sum()} of {y_train.sum()}")

# %%
# Fit
# ---
# :class:`~pypu.models.ElkanNoto` assumes SCAR. SAREM is told that the propensity depends
# on feature 2 only. After EM, it refits the classifier with the estimated propensities as
# weights (``refit=True``, as the original SAR-PU code does); ``refit=False`` keeps the
# classifier that EM learned.

models = {
    "NaivePU": NaivePU(),
    "ElkanNoto": ElkanNoto(random_state=0),
    "SAREM": SAREM(propensity_features=[2]),
    "SAREM(refit=False)": SAREM(propensity_features=[2], refit=False),
}
for model in models.values():
    model.fit(X_train, s_train)
print(f"EM iterations: {models['SAREM'].n_iter_}")

# %%
# Evaluate
# --------
# The naive classifier and ElkanNoto learn that feature 2 predicts the label, and rank the
# samples partly by it. SAREM attributes feature 2 to the propensity, and ranks the samples
# by the two features that decide the class.
#
# On these data the final refit costs SAREM part of its F1. Its weighted objective is not
# convex (each labeled sample also counts as a negative with a negative weight), and the
# solver stops at small coefficients: the refitted classifier still ranks the samples well,
# but its probabilities are too flat, and it predicts too few positives.

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<18}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: SAREM's estimated propensity follows the true one. Right: the weight each
# classifier gives to each feature (the logistic regression's coefficients, scaled to unit
# length).

fig, (ax_prop, ax_coef) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
grid = np.zeros((200, 3))
grid[:, 2] = np.linspace(-3, 3, 200)
ax_prop.plot(grid[:, 2], propensity(grid), "k", label="true propensity")
ax_prop.plot(grid[:, 2], models["SAREM"].predict_propensity(grid), "--", label="SAREM's estimate")
ax_prop.set_xlabel("Feature 2")
ax_prop.set_ylabel("P(labeled | positive)")
ax_prop.set_title("Propensity")
ax_prop.legend()

names = ["NaivePU", "SAREM", "SAREM(refit=False)"]
for i, name in enumerate(names):
    coef = models[name].estimator_.coef_.ravel()
    ax_coef.bar(np.arange(3) + (i - 1) * 0.27, coef / np.linalg.norm(coef), width=0.27, label=name)
ax_coef.axhline(0, color="k", linewidth=0.8)
ax_coef.set_xticks(range(3), ["feature 0", "feature 1", "feature 2"])
ax_coef.set_ylabel("Coefficient (scaled)")
ax_coef.set_title("What each classifier relies on")
ax_coef.legend()
plt.show()
