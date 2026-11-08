"""
LBE: a classifier and a model of the labeling bias
==================================================

:class:`~pypu.models.LBE` (Gong et al.) handles labels that are not chosen at random (SAR).
It learns two logistic models by expectation-maximization: a classifier of the true class,
and a model of the labeling bias, the probability of being labeled. The E-step estimates
how likely each unlabeled sample is to be positive, and the M-step refits both models to
those estimates. LBE needs no class prior.

PyPU ports the code of the original experiments, with their settings (1000 pre-training
steps, then 100 EM iterations of 100 Adam steps). It needs PyTorch:
``pip install "pypu-learn[torch]"``.
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
from sklearn.metrics import RocCurveDisplay, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pypu.metrics import f1_at_prior
from pypu.models import LBE, NaivePU

rng = np.random.default_rng(0)
X, y = make_blobs(n_samples=[1200, 800], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X = np.column_stack([X, rng.normal(size=len(X))])


def propensity(X):
    """Return the probability that a positive gets labeled: it grows with feature 2."""
    return expit(2.5 * X[:, 2] - 1.2)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
s_train = ((y_train == 1) & (rng.random(len(y_train)) < propensity(X_train))).astype(int)
prior = y_train.mean()

# %%
# Fit
# ---
# ``random_state`` seeds the initial weights; the features should be on similar scales,
# as they are here.

naive = NaivePU().fit(X_train, s_train)
clf = LBE(random_state=0).fit(X_train, s_train)
print(f"Classifier coefficients: {np.round(clf.coef_, 2)}")
print(f"Bias model coefficients: {np.round(clf.propensity_coef_, 2)}")

# %%
# The classifier gives feature 2 almost no weight: the bias model takes it. As in the
# original code, the bias model ends up separating labeled from unlabeled samples, so it
# also uses the class features, rather than estimating the propensity alone.
#
# Evaluate
# --------

models = {"NaivePU": naive, "LBE": clf}
for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<8}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Plot
# ----
# Left: the coefficients of the two models of LBE, and of the naive classifier. Right: ROC
# curves on the test data.

fig, (ax_coef, ax_roc) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
coefficients = {
    "NaivePU": naive.estimator_.coef_.ravel(),
    "LBE classifier": clf.coef_,
    "LBE bias model": clf.propensity_coef_,
}
for i, (name, coef) in enumerate(coefficients.items()):
    ax_coef.bar(np.arange(3) + (i - 1) * 0.27, coef / np.linalg.norm(coef), width=0.27, label=name)
ax_coef.axhline(0, color="k", linewidth=0.8)
ax_coef.set_xticks(range(3), ["feature 0", "feature 1", "feature 2"])
ax_coef.set_ylabel("Coefficient (scaled to unit length)")
ax_coef.set_title("What each model relies on")
ax_coef.legend()

for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax_roc)
ax_roc.set_title("ROC curves on the test data")
plt.show()
