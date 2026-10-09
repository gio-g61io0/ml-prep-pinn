"""Evaluation metrics for discrimination, calibration and paired fold comparison."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             matthews_corrcoef, roc_auc_score, roc_curve)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        ece += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        xs.append(p[sel].mean()); ys.append(y[sel].mean()); ns.append(int(sel.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def best_threshold(y: np.ndarray, p: np.ndarray, criterion: str = "youden") -> float:
    """Threshold selected on *training-fold* predictions only."""
    fpr, tpr, thr = roc_curve(y, p)
    if criterion == "youden":
        j = np.argmax(tpr - fpr)
    else:
        j = np.argmin(np.abs(tpr - (1 - fpr)))
    t = float(thr[j])
    return float(np.clip(t, 1e-6, 1 - 1e-6))


def classification_metrics(y: np.ndarray, p: np.ndarray,
                           threshold: Optional[float] = None) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    out: Dict[str, float] = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    except ValueError:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    out["brier"] = float(brier_score_loss(y, p))
    try:
        out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    except ValueError:
        out["log_loss"] = float("nan")
    out["ece"] = expected_calibration_error(y, p)
    out["prevalence"] = float(y.mean())
    out["mean_pred"] = float(p.mean())

    if threshold is not None:
        yhat = (p >= threshold).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
        out["threshold"] = float(threshold)
        out["sensitivity"] = tp / max(tp + fn, 1)
        out["specificity"] = tn / max(tn + fp, 1)
        out["precision"] = tp / max(tp + fp, 1)
        out["mcc"] = float(matthews_corrcoef(y, yhat)) if len(np.unique(yhat)) > 1 else 0.0
    return out


def paired_fold_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank and exact binomial sign test on paired fold scores."""
    from scipy.stats import binomtest, wilcoxon
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    d = a - b
    res: Dict[str, float] = {"n_folds": int(len(d)), "mean_diff": float(np.mean(d)) if len(d) else np.nan,
                             "median_diff": float(np.median(d)) if len(d) else np.nan}
    nz = d[d != 0]
    if len(nz) >= 3:
        try:
            res["wilcoxon_p"] = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:
            res["wilcoxon_p"] = float("nan")
        res["sign_p"] = float(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue)
        res["n_wins"] = int((nz > 0).sum())
    else:
        res["wilcoxon_p"] = res["sign_p"] = float("nan")
        res["n_wins"] = int((nz > 0).sum())
    return res
