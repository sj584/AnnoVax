# Vendored and pinned: this copy is frozen so results cannot drift with upstream edits.
"""Metrics computable from observed positives and unlabeled examples alone.

Everything here is safe for model selection: no hidden label is ever read. The
Each estimator carries its derivation and failure mode in its own docstring.

All estimators assume SCAR (observed positives are a uniform random sample of all
positives). SCAR is assumed, not tested, in this release build.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def proxy_auc(observed: np.ndarray, scores: np.ndarray) -> float:
    """AUC of observed positives against unlabeled examples (the P-vs-U proxy)."""
    observed = np.asarray(observed)
    if np.unique(observed).size < 2:
        return float("nan")
    return float(roc_auc_score(observed, scores))


def prior_threshold(scores: np.ndarray, class_prior: float) -> float:
    """Threshold whose predicted-positive rate equals the declared class prior."""
    if not 0 < class_prior < 1:
        raise ValueError("class_prior must lie strictly between zero and one")
    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        raise ValueError("Cannot select a threshold from no scores")
    rank = max(1, min(len(scores), int(np.ceil(class_prior * len(scores)))))
    return float(np.sort(scores)[-rank])


def unlabeled_prior(observed: np.ndarray, class_prior: float) -> tuple[float, float]:
    """Positive rate *within the unlabeled pool*, plus the label frequency c.

    A common slip is to treat the unlabeled pool as carrying the population prior pi.
    It does not: the labelled positives have been removed from it. Under SCAR with
    label frequency c = P(observed=1 | y=1), which is identified by
    c = P(observed=1) / pi, the leftover pool satisfies

        pi_u = P(y=1 | unlabeled) = pi * (1 - c) / (1 - pi * c)

    Using pi in place of pi_u overstates specificity whenever c > 0.
    """
    observed = np.asarray(observed, dtype=int)
    labelled_rate = float((observed == 1).mean())
    c = float(np.clip(labelled_rate / class_prior, 0.0, 1.0))
    denominator = 1.0 - class_prior * c
    pi_u = 0.0 if denominator <= 0 else class_prior * (1.0 - c) / denominator
    return float(np.clip(pi_u, 0.0, 1.0)), c


def pu_bacc(observed: np.ndarray, scores: np.ndarray, threshold: float,
            class_prior: float) -> dict[str, float]:
    """Balanced accuracy estimated without hidden labels.

    Sensitivity is directly observable on the labelled positives. Specificity is
    recovered from the unlabeled predicted-positive rate q_u, since the unlabeled pool
    is a pi_u mixture of positives and negatives (see unlabeled_prior):

        q_u = pi_u * sensitivity + (1 - pi_u) * (1 - specificity)
        =>  specificity = 1 - (q_u - pi_u * sensitivity) / (1 - pi_u)

    When pi is understated the estimate can leave [0, 1]; it is clipped and the raw
    value is returned alongside so the violation stays visible instead of silent.
    """
    if not 0 < class_prior < 1:
        raise ValueError("class_prior must lie strictly between zero and one")
    observed = np.asarray(observed, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = scores >= threshold
    positives, unlabeled = pred[observed == 1], pred[observed == 0]
    if not len(positives) or not len(unlabeled):
        return {"pu_bacc": float("nan"), "pu_sensitivity": float("nan"),
                "pu_specificity": float("nan"), "pu_specificity_raw": float("nan"),
                "pu_specificity_clipped": False, "unlabeled_prior": float("nan"),
                "label_frequency": float("nan")}
    sensitivity = float(positives.mean())
    q_u = float(unlabeled.mean())
    pi_u, c = unlabeled_prior(observed, class_prior)
    if pi_u >= 1.0:
        raw = float("nan")
    else:
        raw = 1.0 - (q_u - pi_u * sensitivity) / (1.0 - pi_u)
    specificity = float(np.clip(raw, 0.0, 1.0))
    return {
        "pu_bacc": 0.5 * (sensitivity + specificity),
        "pu_sensitivity": sensitivity,
        "pu_specificity": specificity,
        "pu_specificity_raw": float(raw),
        "pu_specificity_clipped": bool(raw < 0.0 or raw > 1.0),
        "unlabeled_prior": pi_u,
        "label_frequency": c,
    }


def pu_f1(observed: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Lee & Liu criterion r^2 / P(f(x)=1): ranks models like F1 but needs no labels."""
    observed = np.asarray(observed, dtype=int)
    pred = np.asarray(scores, dtype=float) >= threshold
    if not (observed == 1).any():
        return float("nan")
    recall = float(pred[observed == 1].mean())
    predicted_positive_rate = float(pred.mean())
    if predicted_positive_rate == 0:
        return 0.0
    return recall ** 2 / predicted_positive_rate


def pu_lift(observed: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """P(pred=1 | observed positive) / P(pred=1): 1.0 means no better than chance."""
    observed = np.asarray(observed, dtype=int)
    pred = np.asarray(scores, dtype=float) >= threshold
    rate = float(pred.mean())
    if rate == 0 or not (observed == 1).any():
        return float("nan")
    return float(pred[observed == 1].mean()) / rate


def pu_auc_bounds(observed: np.ndarray, scores: np.ndarray,
                  class_prior: float) -> dict[str, float]:
    """Bound the true positive-vs-negative AUC from the observed P-vs-U AUC.

    The unlabeled pool mixes negatives with a pi_u fraction of positives, so
    A_pu = pi_u * 0.5 + (1 - pi_u) * A_pn under the convention that positives rank
    indistinguishably against hidden positives. Solving gives the point estimate; the
    interval covers the extremes where hidden positives rank worst or best. Note this
    uses pi_u, the prior *within the unlabeled pool*, not the population prior.
    """
    a_pu = proxy_auc(observed, scores)
    pi_u, _ = unlabeled_prior(observed, class_prior)
    if np.isnan(a_pu) or pi_u >= 1.0:
        return {"pu_auc": float(a_pu), "pn_auc_estimate": float("nan"),
                "pn_auc_lower": float("nan"), "pn_auc_upper": float("nan")}
    denominator = 1.0 - pi_u
    return {
        "pu_auc": a_pu,
        "pn_auc_estimate": float(np.clip((a_pu - 0.5 * pi_u) / denominator, 0.0, 1.0)),
        "pn_auc_lower": float(np.clip((a_pu - pi_u) / denominator, 0.0, 1.0)),
        "pn_auc_upper": float(np.clip(a_pu / denominator, 0.0, 1.0)),
    }


def pu_metrics(observed: np.ndarray, scores: np.ndarray, threshold: float,
               class_prior: float) -> dict[str, float]:
    """All label-free PU metrics for one dataset."""
    result: dict[str, float] = {
        "n": int(len(scores)),
        "observed_positives": int(np.asarray(observed, dtype=int).sum()),
        "predicted_positive_rate": float((np.asarray(scores) >= threshold).mean()),
        "proxy_auc": proxy_auc(observed, scores),
        "pu_f1": pu_f1(observed, scores, threshold),
        "pu_lift": pu_lift(observed, scores, threshold),
    }
    result.update(pu_bacc(observed, scores, threshold, class_prior))
    result.update(pu_auc_bounds(observed, scores, class_prior))
    return result
