from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def compute_metrics_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    fov_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    return compute_metrics_from_probs(probs=probs, targets=targets, fov_mask=fov_mask, threshold=threshold)


def compute_metrics_from_probs(
    probs: np.ndarray,
    targets: np.ndarray,
    fov_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    valid = fov_mask > 0.5
    y_true = targets[valid].astype(np.uint8)
    y_prob = probs[valid]
    y_pred = (y_prob >= threshold).astype(np.uint8)

    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())

    se = _safe_div(tp, tp + fn)
    sp = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5

    return {"Se": se, "Sp": sp, "F1": f1, "Acc": acc, "AUC": auc}
