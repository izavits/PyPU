"""
Estimating the class prior
==========================

Many PU methods need the class prior :math:`\\pi = P(y = 1)`, the share of positives in the
data. When it is unknown, :mod:`pypu.prior` estimates it from the PU training data, assuming
that the labeled positives were chosen completely at random (SCAR):

- :class:`~pypu.prior.BBEPrior`: Best Bin Estimation (Garg et al.). It looks for a region
  where (almost) all samples are positive, and copes with overlapping classes.
- :class:`~pypu.prior.ElkanNotoPrior`: the estimators of Elkan and Noto, from the scores of
  a classifier of labeled versus unlabeled samples. They are accurate when the classes
  barely overlap, and overestimate the prior when they overlap more.
- :class:`~pypu.prior.KMPrior`: kernel mean embedding (Ramaswamy et al.), on the features
  directly. It needs ``pip install "pypu-learn[cvxopt]"``, and its cost grows quickly with
  the number of samples.

Every estimator is fitted like a model, ``fit(X, s)``, and gives ``prior_``, the label
frequency ``label_frequency_`` and the share of positives among the unlabeled samples
``pollution_``.
"""

# %%
# Data
# ----
# Two Gaussian blobs with 40% positives, well separated or overlapping, and with 10% to 70%
# of the positives labeled.

import importlib.util

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from pypu.datasets import make_pu_split
from pypu.models import IFPU
from pypu.prior import BBEPrior, ElkanNotoPrior, KMPrior

datasets = {
    "Separated classes": make_blobs(
        n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0
    ),
    "Overlapping classes": make_blobs(
        n_samples=[600, 400], centers=[[-0.7, -0.7], [0.7, 0.7]], random_state=0
    ),
}
label_frequencies = [0.1, 0.3, 0.5, 0.7]

# %%
# Estimate
# --------
# KMPrior runs only when cvxopt is installed.

estimators = {
    "BBEPrior": BBEPrior(random_state=0),
    "ElkanNotoPrior": ElkanNotoPrior(random_state=0),
}
if importlib.util.find_spec("cvxopt") is not None:
    estimators["KMPrior (KM1)"] = KMPrior(method="km1")
    estimators["KMPrior (KM2)"] = KMPrior(method="km2")

estimates = {}
for data_name, (X, y) in datasets.items():
    for c in label_frequencies:
        X_train, _, s_train, y_train, _ = make_pu_split(X, y, label_frequency=c, random_state=0)
        for name, estimator in estimators.items():
            estimates[data_name, name, c] = estimator.fit(X_train, s_train).prior_
        row = "  ".join(f"{name} {estimates[data_name, name, c]:.3f}" for name in estimators)
        print(f"{data_name}, c = {c}: {row}")

# %%
# Plot
# ----
# The estimates improve with more labeled positives. Elkan and Noto's estimator
# overestimates the prior, the more so when the classes overlap.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, layout="constrained")
for ax, data_name in zip(axes, datasets, strict=True):
    for name in estimators:
        ax.plot(
            label_frequencies,
            [estimates[data_name, name, c] for c in label_frequencies],
            "o-",
            label=name,
        )
    ax.axhline(0.4, color="k", linestyle="--", label="true prior")
    ax.set_xlabel("Label frequency c")
    ax.set_title(data_name)
axes[0].set_ylabel("Estimated class prior")
axes[0].legend()
plt.show()

# %%
# Using the estimate
# ------------------
# Pass the estimate to any model that needs the prior:

X, y = datasets["Separated classes"]
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = BBEPrior(random_state=0).fit(X_train, s_train).prior_
clf = IFPU(prior=prior, random_state=0).fit(X_train, s_train)
print(
    f"Estimated prior {prior:.3f}; IFPU test accuracy {(clf.predict(X_test) == y_test).mean():.3f}"
)
