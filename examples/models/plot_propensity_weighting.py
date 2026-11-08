"""
PropensityWeightedPU: learning from SAR data with a known propensity
====================================================================

When the probability that a positive gets labeled, the *propensity*
:math:`e(x) = P(s = 1 \\mid y = 1, x)`, depends on the sample (SAR), a labeled sample with a
low propensity stands for many positives that were not labeled. Given the propensity,
:class:`~pypu.models.PropensityWeightedPU` (Bekker et al.) weights each labeled sample by
:math:`1 / e(x)` as a positive (and :math:`1 - 1 / e(x)` as a negative), and each unlabeled
sample by 1 as a negative. This makes the training loss an unbiased estimate of the loss on
the true labels.

The propensity can be known from how the data were collected, or estimated, for example by
:class:`~pypu.models.SAREM`.
"""

# %%
# SAR data
# --------
# Two features decide the class: two Gaussian blobs with 40% positives. A third feature,
# unrelated to the class, decides which positives get labeled.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from sklearn.datasets import make_blobs
from sklearn.metrics import RocCurveDisplay, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pypu.metrics import f1_at_prior
from pypu.models import SAREM, NaivePU, PropensityWeightedPU

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
# With a constant propensity, the label frequency :math:`c`, the weights assume SCAR. With
# the true propensity function, they correct for how the labels were chosen. Without it,
# SAREM's estimate can stand in.

label_frequency = s_train.sum() / y_train.sum()
sarem = SAREM(propensity_features=[2]).fit(X_train, s_train)
models = {
    "NaivePU": NaivePU(),
    "constant c": PropensityWeightedPU(propensity=label_frequency),
    "true e(x)": PropensityWeightedPU(propensity=propensity),
    "SAREM's e(x)": PropensityWeightedPU(propensity=sarem.predict_propensity),
}
for model in models.values():
    model.fit(X_train, s_train)

# %%
# Evaluate
# --------
# With the propensity, the classifier ranks the test samples almost perfectly. As for
# SAREM's refit, the weighted objective is not convex and its solution has flat
# probabilities, so ``predict`` finds too few positives: when the class prior is known, a
# threshold set from it does better (F1 at prior).

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
# Left: the weights of the labeled samples, :math:`1 / e(x)`: rarely labeled positives
# count for more. Right: ROC curves on the test data.

fig, (ax_weight, ax_roc) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
labeled = s_train == 1
ax_weight.scatter(X_train[labeled, 2], 1 / propensity(X_train[labeled]), s=10)
ax_weight.set_yscale("log")
ax_weight.set_xlabel("Feature 2")
ax_weight.set_ylabel("Weight 1 / e(x)")
ax_weight.set_title("Weights of the labeled samples")

for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax_roc)
ax_roc.set_title("ROC curves on the test data")
plt.show()
