"""
Which positives get labeled
===========================

PU data arise when only some positives get labeled. *Which* positives get labeled is the
labeling mechanism, and it matters: most PU methods assume that the labeled positives are
chosen completely at random (SCAR), and their estimates are biased when they are not (SAR,
selected at random given the features).

:mod:`pypu.datasets` makes PU data from a fully labeled dataset with one SCAR and four SAR
mechanisms, so that methods can be tested under each. This example shows what each one
does.
"""

# %%
# The mechanisms
# --------------
# Each function takes the true labels ``y`` (and the features ``X`` for the SAR ones) and
# returns PU labels ``s``. Here each labels 30% of the positives of two Gaussian blobs.

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from pypu.datasets import (
    label_frequency_from_pollution,
    label_sar_cluster,
    label_sar_confidence,
    label_sar_knn,
    label_sar_propensity,
    label_scar,
    make_pu_split,
    pu_summary,
)
from pypu.prior import BBEPrior

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)

mechanisms = {
    "SCAR: uniformly at random": label_scar(y, label_frequency=0.3, random_state=0),
    "SAR kNN: near the negatives": label_sar_knn(X, y, label_frequency=0.3, random_state=0),
    "SAR propensity: large feature 0": label_sar_propensity(
        X, y, label_frequency=0.3, features=[0], strength=2.0, random_state=0
    ),
    "SAR cluster: some clusters only": label_sar_cluster(X, y, label_frequency=0.3, random_state=0),
    "SAR confidence: the easy ones": label_sar_confidence(
        X, y, label_frequency=0.3, random_state=0
    ),
}

# %%
# - :func:`~pypu.datasets.label_scar` picks the labeled positives uniformly at random.
# - :func:`~pypu.datasets.label_sar_knn` labels positives by their distance from the
#   negatives: with ``bias="near"`` (the default) mostly the ones near the class boundary,
#   with ``bias="far"`` mostly the ones far from it.
# - :func:`~pypu.datasets.label_sar_propensity` labels each positive with a known logistic
#   propensity of chosen features; ``return_propensity=True`` returns it.
# - :func:`~pypu.datasets.label_sar_cluster` labels positives from only some k-means
#   clusters of the positive class.
# - :func:`~pypu.datasets.label_sar_confidence` labels mostly the positives that a
#   classifier recognises easily (``bias="easy"``) or poorly (``bias="hard"``).

fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), sharex=True, sharey=True, layout="constrained")
axes = axes.ravel()
axes[0].scatter(*X[y == 0].T, s=8, c="tab:blue", label="negative")
axes[0].scatter(*X[y == 1].T, s=8, c="tab:orange", label="positive")
axes[0].set_title("True labels")
axes[0].legend(loc="lower right")
for ax, (name, s) in zip(axes[1:], mechanisms.items(), strict=True):
    ax.scatter(*X[s == 0].T, s=8, c="0.8", label="unlabeled")
    ax.scatter(*X[s == 1].T, s=8, c="tab:orange", label="labeled positive")
    ax.set_title(name)
axes[1].legend(loc="lower right")
plt.show()

# %%
# Why it matters
# --------------
# The estimators of the class prior assume SCAR, and so do many PU methods. Under SAR the
# labeled positives are not representative of all the positives, and the estimates are
# off. With Best Bin Estimation, for example, the estimate is close to the true prior of 0.4
# under SCAR, but far too low when the labeled positives come from only part of the
# positive class, which the estimator then takes for all of it:

print("Mechanism                         estimated prior (true: 0.400)")
for name, s in mechanisms.items():
    print(f"{name:<32}  {BBEPrior(random_state=0).fit(X, s).prior_:.3f}")

# %%
# In practice
# -----------
# :func:`~pypu.datasets.make_pu_split` splits a dataset into PU training data and a fully
# labeled test set, with any mechanism by name; extra arguments go to the mechanism.

X_train, X_test, s_train, y_train, y_test = make_pu_split(
    X, y, label_frequency=0.3, mechanism="sar_knn", bias="far", random_state=0
)
print(pu_summary(y_train, s_train))

# %%
# The label frequency :math:`c` is the share of positives that are labeled. To set the share
# of positives among the unlabeled samples (the pollution) instead, convert it:

c = label_frequency_from_pollution(0.1, prior=0.4)
print(f"For 10% positives among the unlabeled samples, label {c:.1%} of the positives")

# %%
# :func:`~pypu.datasets.make_case_control` makes data for the case-control scenario, where
# the labeled positives and the unlabeled samples are two separate samples.
