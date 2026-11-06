"""Validation helpers shared by the metrics."""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_array, check_consistent_length, column_or_1d


def check_scores(y_score, labels) -> np.ndarray:
    """Validate one finite score per sample, higher meaning more likely positive."""
    y_score = check_array(y_score, ensure_2d=False, dtype=np.float64, input_name="y_score")
    if y_score.ndim != 1:
        raise ValueError(
            f"y_score must be 1-D, got shape {y_score.shape}. For the output of "
            "predict_proba, pass the positive-class column: predict_proba(X)[:, 1]."
        )
    check_consistent_length(labels, y_score)
    return y_score


def check_predictions(y_pred, labels) -> np.ndarray:
    """Validate predicted labels in {0, 1}; a single class is allowed."""
    y_pred = column_or_1d(y_pred, warn=True)
    if y_pred.dtype.kind not in "biuf" or not np.isin(y_pred, (0, 1)).all():
        raise ValueError(
            "y_pred must hold predicted labels: 1 for positive, 0 for negative, got values "
            f"{np.unique(y_pred)[:10].tolist()}. For scores, use a ranking metric or "
            "threshold them first."
        )
    check_consistent_length(labels, y_pred)
    return y_pred.astype(np.int64)
