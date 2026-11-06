"""Make PU data from fully labeled binary datasets.

The labeling functions hide the labels of some positives, following a chosen labeling
mechanism, and return PU labels ``s`` (1 for a labeled positive, 0 for unlabeled):

- :func:`label_scar`: uniformly at random (SCAR).
- :func:`label_sar_knn`: by distance from the negatives.
- :func:`label_sar_propensity`: with a known logistic propensity on chosen features.
- :func:`label_sar_cluster`: only from some clusters of the positives.
- :func:`label_sar_confidence`: by how confidently a classifier recognises each positive.

:func:`make_pu_split` combines a stratified train/test split with one of these mechanisms,
:func:`make_case_control` makes case-control data, and :func:`pu_summary` describes PU data
whose true labels are known.
"""

from pypu.datasets._case_control import make_case_control
from pypu.datasets._labeling import (
    label_sar_cluster,
    label_sar_confidence,
    label_sar_knn,
    label_sar_propensity,
    label_scar,
)
from pypu.datasets._split import PUSplit, make_pu_split
from pypu.datasets._summary import (
    label_frequency_from_pollution,
    pollution_from_label_frequency,
    pu_summary,
)

__all__ = [
    "PUSplit",
    "label_frequency_from_pollution",
    "label_sar_cluster",
    "label_sar_confidence",
    "label_sar_knn",
    "label_sar_propensity",
    "label_scar",
    "make_case_control",
    "make_pu_split",
    "pollution_from_label_frequency",
    "pu_summary",
]
