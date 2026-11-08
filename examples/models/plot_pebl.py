"""
PEBL: an iterative SVM from reliable negatives
==============================================

:class:`~pypu.models.PEBL` is a two-step method. It first finds reliable negatives, then
grows them with an iterative SVM: each SVM separates the labeled positives from the
negatives found so far, and the unlabeled samples it classifies as negative join them,
until no new negative is found. One SVM of the sequence is kept, chosen by Lee and Liu's
criterion (:func:`~pypu.metrics.lee_liu_score`).

The reliable negatives come from spies by default (Spy-SVM), or from 1-DNF for binary
features, as in the original PEBL for text: a feature is *positive* if it is more frequent
among the labeled positives than among the unlabeled samples, and unlabeled samples with no
positive feature are reliable negatives.
"""

# %%
# Data
# ----
# Two Gaussian blobs with 40% positives; 30% of the training positives are labeled,
# chosen completely at random.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pypu.datasets import label_scar, make_pu_split
from pypu.metrics import f1_at_prior
from pypu.models import PEBL, NaivePU

X, y = make_blobs(n_samples=[600, 400], centers=[[-1.2, -1.2], [1.2, 1.2]], random_state=0)
X_train, X_test, s_train, y_train, y_test = make_pu_split(X, y, label_frequency=0.3, random_state=0)
prior = y_train.mean()

# %%
# Spy-SVM
# -------
# The SVMs are linear by default.

models = {"NaivePU": NaivePU(), "PEBL": PEBL(random_state=0)}
for model in models.values():
    model.fit(X_train, s_train)

clf = models["PEBL"]
print(f"Reliable negatives from the spies: {clf.reliable_negatives_.sum()}")
print(f"SVMs trained: {clf.n_iter_}, kept: number {clf.selected_iteration_ + 1}")

for name, model in models.items():
    scores = model.predict_proba(X_test)[:, 1]
    print(
        f"{name:<8}  ROC AUC {roc_auc_score(y_test, scores):.3f}"
        f"  F1 {f1_score(y_test, model.predict(X_test)):.3f}"
        f"  F1 at prior {f1_at_prior(y_test, scores, prior):.3f}"
    )

# %%
# Each SVM moves its boundary towards the positives, as more unlabeled samples become
# negatives. Thin lines show the sequence, the thick line the SVM that was kept.

fig, ax = plt.subplots(figsize=(5.5, 4.6), layout="constrained")
ax.scatter(*X_test[y_test == 0].T, s=10, c="tab:blue", label="negative")
ax.scatter(*X_test[y_test == 1].T, s=10, c="tab:orange", label="positive")
for i, svm in enumerate(clf.estimators_):
    DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="decision_function",
        plot_method="contour",
        levels=[0],
        colors="k",
        linewidths=2.5 if i == clf.selected_iteration_ else 0.6,
        alpha=1 if i == clf.selected_iteration_ else 0.5,
        ax=ax,
    )
ax.plot([], [], "k", linewidth=0.6, label="SVM of an iteration")
ax.plot([], [], "k", linewidth=2.5, label="SVM kept")
ax.set_title("Test data and the SVMs of PEBL")
ax.legend(loc="lower right")
plt.show()

# %%
# 1-DNF on binary features
# ------------------------
# PEBL was made for text. Here 2,000 "documents" use 40 words: words 0-4 are common in
# positive documents, words 5-9 in negative ones, and the others in both. 30% of the
# training positives are labeled.

rng = np.random.default_rng(0)
y_docs = (rng.random(2000) < 0.3).astype(int)
word_rates = np.full((2000, 40), 0.1)
word_rates[y_docs == 1, :5] = 0.4
word_rates[y_docs == 0, 5:10] = 0.4
X_docs = (rng.random((2000, 40)) < word_rates).astype(int)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_docs, y_docs, stratify=y_docs, random_state=0
)
sd_train = label_scar(yd_train, label_frequency=0.3, random_state=0)

dnf = PEBL(negatives="1dnf").fit(Xd_train, sd_train)
reliable = dnf.reliable_negatives_
print(f"Positive features (words): {np.flatnonzero(dnf.positive_features_).tolist()}")
n_negative = (yd_train[reliable] == 0).sum()
print(f"Reliable negatives: {reliable.sum()}, of which truly negative: {n_negative}")

docs_prior = yd_train.mean()
for name, model in {"NaivePU": NaivePU().fit(Xd_train, sd_train), "PEBL (1-DNF)": dnf}.items():
    scores = model.predict_proba(Xd_test)[:, 1]
    print(
        f"{name:<12}  ROC AUC {roc_auc_score(yd_test, scores):.3f}"
        f"  F1 {f1_score(yd_test, model.predict(Xd_test)):.3f}"
        f"  F1 at prior {f1_at_prior(yd_test, scores, docs_prior):.3f}"
    )

# %%
# 1-DNF finds words 0-4, but also noise words that happen to be a little more frequent
# among the labeled positives. A reliable negative must contain none of them, so there are
# few reliable negatives, but almost all of them are truly negative.
