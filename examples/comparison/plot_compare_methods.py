"""
Comparing the methods
=====================

All the models of :mod:`pypu.models` on one real dataset: scikit-learn's breast cancer data
(569 tumours, 30 features), with the malignant tumours as positives. Half of the training
positives are labeled, chosen completely at random, and the test set keeps its true labels.

Every model is a scikit-learn estimator, so each is put in a pipeline after a
``StandardScaler``. The models that need the class prior get the true one, as an oracle;
in practice it would be estimated (see the prior estimation example). The models that need
XGBoost or PyTorch are skipped when these are not installed.

This is one dataset and one labeling, averaged over five random splits: it shows how to
compare methods, not which method is best in general.
"""

# %%
# Data
# ----

import importlib.util
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pypu.datasets import make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import (
    IFPNU,
    IFPU,
    PEBL,
    SAREM,
    BiasedSVM,
    ElkanNoto,
    NaivePU,
    PropensityWeightedPU,
    PUBagging,
    PUHellingerForest,
    PUHellingerTree,
    RankPruning,
    RebalancedPU,
    SpyPU,
)

data = load_breast_cancer()
X, y = data.data, 1 - data.target  # target 0 is malignant: make it the positive class


# %%
# Models
# ------
# Default settings, except for PUe, whose defaults (the original experiments' 20 training
# steps) leave it untrained.


def make_models(prior, label_frequency):
    """Return the models to compare, given the class prior and the label frequency."""
    models = {
        "NaivePU": NaivePU(),
        "NaivePU(prior)": NaivePU(prior=prior),
        "ElkanNoto": ElkanNoto(random_state=0),
        "RebalancedPU": RebalancedPU(prior=prior),
        "PropensityWeightedPU": PropensityWeightedPU(propensity=label_frequency),
        "PUBagging": PUBagging(random_state=0),
        "SpyPU": SpyPU(random_state=0),
        "PEBL": PEBL(random_state=0),
        "BiasedSVM": BiasedSVM(random_state=0),
        "RankPruning": RankPruning(random_state=0),
        "SAREM": SAREM(),
        "IFPU": IFPU(prior=prior, random_state=0),
        "IFPNU": IFPNU(prior=prior, random_state=0),
        "PUHellingerTree": PUHellingerTree(prior=prior, random_state=0),
        "PUHellingerForest": PUHellingerForest(prior=prior, random_state=0),
    }
    if importlib.util.find_spec("xgboost") is not None:
        from pypu.models import ImbalancedNNPUBoost, NNPUBoost, UPUBoost

        models["UPUBoost"] = UPUBoost(prior=prior, random_state=0)
        models["NNPUBoost"] = NNPUBoost(prior=prior, random_state=0)
        models["ImbalancedNNPUBoost"] = ImbalancedNNPUBoost(prior=prior, random_state=0)
    if importlib.util.find_spec("torch") is not None:
        from pypu.models import ALMPU, LBE, PUe

        models["LBE"] = LBE(random_state=0)
        models["PUe (trained)"] = PUe(prior=prior, n_steps=2000, learning_rate=0.01, random_state=0)
        models["ALMPU"] = ALMPU(random_state=0)
    return models


# %%
# Run
# ---
# Five splits, each with its own PU labels. For each model: test ROC AUC, F1 of
# ``predict``, F1 at the prior, and the training time.

results = {}
for split in range(5):
    X_train, X_test, s_train, y_train, y_test = make_pu_split(
        X, y, label_frequency=0.5, random_state=split
    )
    prior = y_train.mean()
    for name, model in make_models(prior, s_train.mean() / prior).items():
        pipeline = make_pipeline(StandardScaler(), model)
        start = time.perf_counter()
        pipeline.fit(X_train, s_train)
        seconds = time.perf_counter() - start
        scores = pipeline.predict_proba(X_test)[:, 1]
        results.setdefault(name, []).append(
            [
                roc_auc_score(y_test, scores),
                f1_score(y_test, pipeline.predict(X_test)),
                f1_at_prior(y_test, scores, prior),
                seconds,
            ]
        )

results = {name: np.array(rows) for name, rows in results.items()}
print(f"{'model':<21} {'ROC AUC':>13} {'F1':>13} {'F1 at prior':>13} {'seconds':>8}")
for name, rows in sorted(results.items(), key=lambda item: -item[1][:, 0].mean()):
    mean, std = rows.mean(axis=0), rows.std(axis=0)
    print(
        f"{name:<21} {mean[0]:.3f} ± {std[0]:.3f} {mean[1]:.3f} ± {std[1]:.3f}"
        f" {mean[2]:.3f} ± {std[2]:.3f} {mean[3]:8.2f}"
    )

# %%
# Under SCAR the naive classifier already ranks the samples well; its F1 suffers from its
# threshold, which ``NaivePU(prior=...)`` fixes. Two models fall behind for reasons their
# examples show: PUHellingerTree grows to full depth by default and overfits the PU labels
# (``max_depth=5`` helps), and ALMPU's pseudo-labels sometimes go wrong on small datasets,
# which makes its results vary from split to split.
#
# Plot
# ----
# Mean test ROC AUC and F1 at the prior over the five splits, with their standard
# deviations. ROC AUC and F1 at the prior judge the ranking of the samples; the F1 of
# ``predict`` in the table above also judges each model's own threshold, which several
# models get wrong (see their examples).

names = sorted(results, key=lambda name: results[name][:, 0].mean())
positions = np.arange(len(names))
fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
for offset, column, label in [(0.2, 0, "ROC AUC"), (-0.2, 2, "F1 at prior")]:
    ax.barh(
        positions + offset,
        [results[name][:, column].mean() for name in names],
        xerr=[results[name][:, column].std() for name in names],
        height=0.4,
        label=label,
    )
ax.set_yticks(positions, names)
ax.set_xlim(0.5, 1)
ax.set_xlabel("Test score (mean over 5 splits)")
ax.set_title("PU methods on the breast cancer data")
ax.legend(loc="lower right")
plt.show()
