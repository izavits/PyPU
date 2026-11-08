"""
PUe: inverse propensity weights, as in the original experiments
===============================================================

:class:`~pypu.models.PUe` (Wang et al.) targets labels that are not chosen at random (SAR).
It trains a logistic model of the propensity, the probability of being labeled, then a
logistic classifier whose loss weights each labeled sample by the inverse of its propensity.

PyPU ports the code of the original experiments, and keeps their settings so that their
results can be reproduced. This example shows what these settings do, and how to train
the model further. It needs PyTorch: ``pip install "pypu-learn[torch]"``. Without a
``prior``, PUe estimates it with KM1, which also needs ``pip install "pypu-learn[cvxopt]"``;
this example passes the prior instead.
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
from pypu.models import NaivePU, PUe

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
# The experiments' settings
# -------------------------
# The experiments trained each model for 20 Adam steps with learning rate 0.001, from random
# initial weights. That barely moves the weights, so the scores depend on the seed more
# than on the data:

seeds = range(10)
as_run = [
    roc_auc_score(
        y_test, PUe(prior=prior, random_state=seed).fit(X_train, s_train).decision_function(X_test)
    )
    for seed in seeds
]
print("ROC AUC by seed, as run:", np.round(as_run, 2))

# %%
# A trained model
# ---------------
# With more steps and a larger learning rate, the model trains. Its loss sums over the
# unlabeled samples but averages over the labeled ones, so the unlabeled samples weigh
# :math:`n_u` times more: the classifier ranks the samples, but puts every one of them
# below 0.5, and ``predict`` finds no positive. With the class prior, a threshold can be
# set from it instead (F1 at prior).

trained = [
    PUe(prior=prior, n_steps=2000, learning_rate=0.01, random_state=seed).fit(X_train, s_train)
    for seed in seeds
]
print(
    "ROC AUC by seed, trained:",
    np.round([roc_auc_score(y_test, m.decision_function(X_test)) for m in trained], 3),
)

models = {"NaivePU": NaivePU().fit(X_train, s_train), "PUe, trained": trained[0]}
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
# Left: the test ROC AUC of each seed, with the experiments' settings and trained. Right:
# ROC curves of the trained model and of the naive classifier.

fig, (ax_seed, ax_roc) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
ax_seed.plot(seeds, as_run, "o", label="as run (20 steps, lr 0.001)")
ax_seed.plot(
    seeds,
    [roc_auc_score(y_test, m.decision_function(X_test)) for m in trained],
    "s",
    label="trained (2000 steps, lr 0.01)",
)
ax_seed.axhline(0.5, color="0.5", linewidth=0.8)
ax_seed.set_xlabel("random_state")
ax_seed.set_ylabel("Test ROC AUC")
ax_seed.set_title("Settings of the experiments vs. trained")
ax_seed.legend()

for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax_roc)
ax_roc.set_title("ROC curves on the test data")
plt.show()
