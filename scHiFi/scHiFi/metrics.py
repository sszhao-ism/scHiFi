from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    n = max(int(y_true.max(initial=0)), int(y_pred.max(initial=0))) + 1
    contingency = np.zeros((n, n), dtype=np.int64)
    np.add.at(contingency, (y_pred, y_true), 1)
    row, col = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[row, col].sum() / max(1, y_true.size))


def evaluate_clustering(
    y_true: Optional[np.ndarray], y_pred: np.ndarray
) -> Dict[str, float]:
    if y_true is None:
        return {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "ACC": clustering_accuracy(y_true, y_pred),
        "NMI": float(metrics.normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(metrics.adjusted_rand_score(y_true, y_pred)),
        "AMI": float(metrics.adjusted_mutual_info_score(y_true, y_pred)),
    }
