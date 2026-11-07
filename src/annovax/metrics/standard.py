# Vendored and pinned: this copy is frozen so results cannot drift with upstream edits.
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

#: Reported first because the data is heavily imbalanced; plain accuracy is deliberately
#: absent, since a trivial all-negative classifier would score ~0.91 on it here.
HEADLINE_METRICS = ("balanced_accuracy", "mcc", "roc_auc", "pr_auc")


def classification_metrics(y_true, scores, threshold: float) -> dict[str, float | int]:
    """Fully supervised metrics. Retrospective evaluation only -- never model selection."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        both = np.unique(y_true).size == 2
        roc = float(roc_auc_score(y_true, scores)) if both else float("nan")
        ap = float(average_precision_score(y_true, scores)) if y_true.sum() else float("nan")
        bal = float(balanced_accuracy_score(y_true, pred)) if both else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    result: dict[str, float | int] = {
        "n": int(len(y_true)), "positives": int(y_true.sum()),
        "balanced_accuracy": bal,
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "roc_auc": roc, "pr_auc": ap,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "brier": float(brier_score_loss(y_true, scores)),
        "threshold": float(threshold), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }
    order = np.argsort(-scores)
    for k in (10, 25, 50, 100):
        selected = y_true[order[:min(k, len(order))]]
        result[f"precision_at_{k}"] = float(selected.mean())
        result[f"recall_at_{k}"] = float(selected.sum() / max(1, y_true.sum()))
    return result
