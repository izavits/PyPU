"""Metrics for PU learning.

With the true labels of a test set (scores ranked, no fixed threshold needed):

- :func:`r_precision`, :func:`precision_at_k`, :func:`recall_at_k`
- :func:`f1_at_prior`: F1 with the top ``prior`` fraction predicted positive.
- :func:`max_f1_score`: the best F1 over all thresholds.

From PU labels alone, for model selection when the true labels are unknown:

- :func:`lee_liu_score`, which needs no class prior.
- :func:`pu_recall_score`, :func:`pu_precision_score`, :func:`pu_f1_score` and
  :func:`pu_roc_auc_score`, estimates of the usual metrics given the class prior.

Standard metrics such as :func:`sklearn.metrics.roc_auc_score` and
:func:`sklearn.metrics.average_precision_score` apply as they are when true labels are known.
"""

from pypu.metrics._pu_labels import (
    lee_liu_score,
    pu_f1_score,
    pu_precision_score,
    pu_recall_score,
    pu_roc_auc_score,
)
from pypu.metrics._true_labels import (
    f1_at_prior,
    max_f1_score,
    precision_at_k,
    r_precision,
    recall_at_k,
)

__all__ = [
    "f1_at_prior",
    "lee_liu_score",
    "max_f1_score",
    "precision_at_k",
    "pu_f1_score",
    "pu_precision_score",
    "pu_recall_score",
    "pu_roc_auc_score",
    "r_precision",
    "recall_at_k",
]
