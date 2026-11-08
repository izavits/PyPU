"""
Tuning a model without true labels
==================================

Choosing hyperparameters usually needs labeled validation data, which PU data lack: there
is no labeled negative. :mod:`pypu.metrics` estimates metrics from PU labels alone, so that
scikit-learn's model selection works on PU data:

- :func:`~pypu.metrics.pu_roc_auc_score`, :func:`~pypu.metrics.pu_f1_score`,
  :func:`~pypu.metrics.pu_precision_score` and :func:`~pypu.metrics.pu_recall_score`
  estimate the usual metrics, given the class prior.
- :func:`~pypu.metrics.lee_liu_score`, :math:`r^2 / P(\\hat y = 1)`, needs nothing else. It
  is not a metric on its own scale, but ranks models as F1 would.

They assume that the labeled positives were chosen completely at random (SCAR).
"""

# %%
# Data
# ----
# Two interleaving half circles with 40% positives; 30% of the training positives are
# labeled, chosen completely at random.

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from pypu.datasets import make_pu_split
from pypu.metrics import lee_liu_score, pu_roc_auc_score
from pypu.models import NaivePU

X, y = make_moons(n_samples=(600, 400), noise=0.25, random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()  # in practice, an estimate (see the prior estimation example)

# %%
# Search
# ------
# The number of neighbors of a k-nearest-neighbors classifier, used as a naive PU model:
# few neighbors overfit, many underfit. Cross-validation on the training data scores each
# setting with the PU estimate of ROC AUC and with Lee and Liu's criterion. The model is
# given the prior, so that its predictions, which Lee and Liu's criterion scores, are made
# at a sensible threshold.

grid = {"estimator__n_neighbors": [1, 2, 5, 10, 20, 50, 100, 200, 400]}
search = GridSearchCV(
    NaivePU(KNeighborsClassifier(), prior=prior),
    grid,
    scoring={
        "pu_roc_auc": make_scorer(pu_roc_auc_score, response_method="predict_proba", prior=prior),
        "lee_liu": make_scorer(lee_liu_score),
    },
    refit="pu_roc_auc",
).fit(X_train, s_train)  # only the PU labels are used
print(f"Chosen: {search.best_params_}")

# %%
# Check against the true labels
# -----------------------------
# The test set's true labels show how good each setting really is. The PU estimates follow
# the true ROC AUC: both rule out the settings that overfit or underfit, and the chosen
# setting is close to the best one.

neighbors = grid["estimator__n_neighbors"]
true_auc = []
for k in neighbors:
    model = NaivePU(KNeighborsClassifier(n_neighbors=k), prior=prior).fit(X_train, s_train)
    true_auc.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
for k, pu_auc, auc in zip(
    neighbors, search.cv_results_["mean_test_pu_roc_auc"], true_auc, strict=True
):
    print(f"n_neighbors={k:<3}  PU estimate of ROC AUC {pu_auc:.3f}  true test ROC AUC {auc:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4.6), layout="constrained")
ax.plot(neighbors, true_auc, "o-", label="true ROC AUC (test set)")
ax.plot(
    neighbors,
    search.cv_results_["mean_test_pu_roc_auc"],
    "s-",
    label="PU estimate (cross-validated)",
)
ax.set_xscale("log")
ax.set_xlabel("n_neighbors")
ax.set_ylabel("ROC AUC")
lee_liu_ax = ax.twinx()
lee_liu_ax.plot(neighbors, search.cv_results_["mean_test_lee_liu"], "^:", color="tab:green")
lee_liu_ax.set_ylabel("Lee-Liu score (cross-validated)", color="tab:green")
ax.plot([], [], "^:", color="tab:green", label="Lee-Liu score (right axis)")
ax.legend(loc="lower center")
ax.set_title("Model selection from PU labels")
plt.show()
