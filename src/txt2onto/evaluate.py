"""Metrics and threshold selection (PLAN.md sections 4 and 6)."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, roc_auc_score)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Precision / recall / F1 for the human class at `threshold`, plus macro-F1 and confusion counts."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
        "average_precision": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def _top_k_precision_threshold(y: np.ndarray, s: np.ndarray, target: float, min_support: int):
    """Lowest score t such that predicting positive for s >= t keeps precision >= target for
    every cut-off between the top `min_support` items and t. Returns (t, n_selected) or (None, 0)."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])  # last index of each tie block
    k = ends + 1
    precision = np.cumsum(ys)[ends] / k
    ok = k >= min_support
    if not ok.any():
        return None, 0
    running_min = np.minimum.accumulate(np.where(ok, precision, np.inf))
    valid = ok & (running_min >= target)
    if not valid.any():
        return None, 0
    last = np.flatnonzero(valid)[-1]
    return float(ss[ends[last]]), int(k[last])


def precision_target_thresholds(y: np.ndarray, p: np.ndarray, target: float, min_support: int = 100) -> dict:
    """τ⁺: lowest P(human) where human precision stays >= target; τ⁻: highest P(human) where
    not-human precision stays >= target (PLAN.md section 5)."""
    tau_pos, n_pos = _top_k_precision_threshold(y, p, target, min_support)
    tau_neg_s, n_neg = _top_k_precision_threshold(1 - y, 1 - p, target, min_support)
    n = len(y)
    return {
        "target_precision": target,
        "tau_pos": None if tau_pos is None else round(tau_pos, 4),
        "tau_neg": None if tau_neg_s is None else round(1 - tau_neg_s, 4),
        "coverage_pos": round(n_pos / n, 4),
        "coverage_neg": round(n_neg / n, 4),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    return [
        {"bin_lo": round(float(edges[b]), 2), "bin_hi": round(float(edges[b + 1]), 2), "n": int((idx == b).sum()),
         "mean_p": round(float(p[idx == b].mean()), 4), "frac_human": round(float(y[idx == b].mean()), 4)}
        for b in range(n_bins) if (idx == b).any()
    ]
