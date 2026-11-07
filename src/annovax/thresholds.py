"""Threshold rules. Only the label-free ones may inform a decision."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef

from .metrics.pu_metrics import prior_threshold

__all__ = ["prior_threshold", "mcc_optimal_threshold", "THRESHOLD_RULES"]

#: Rule -> whether it reads hidden labels. A True here means the number is retrospective
#: and may never be quoted as the headline.
THRESHOLD_RULES = {"prior_transfer": False, "prior_insample": False,
                   "fixed_0.5": False, "mcc_optimal": True}


def mcc_optimal_threshold(y_true, scores, max_candidates: int = 400) -> float:
    """Threshold maximizing MCC. Reads labels, so retrospective reporting only."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(scores)
    if len(candidates) > max_candidates:
        candidates = np.quantile(candidates, np.linspace(0, 1, max_candidates))
    best, best_score = 0.5, -2.0
    for threshold in candidates:
        value = matthews_corrcoef(y_true, (scores >= threshold).astype(int))
        if value > best_score:
            best, best_score = float(threshold), float(value)
    return best
