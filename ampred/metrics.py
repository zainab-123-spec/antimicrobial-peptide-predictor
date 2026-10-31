"""Binary classification metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, MCC, ROC-AUC, sensitivity and specificity."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def bootstrap_ci(
    y_true, y_prob, metric: str = "auc", n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval for one metric (resampling test items)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(binary_metrics(y_true[idx], y_prob[idx])[metric])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
