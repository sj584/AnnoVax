"""Assembles the four metric families into tidy rows."""
from __future__ import annotations

import numpy as np

from .metrics.pu_metrics import pu_metrics
from .metrics.standard import classification_metrics
from .thresholds import THRESHOLD_RULES, mcc_optimal_threshold, prior_threshold


def observed_positive_rate(y) -> float:
    return float((np.asarray(y) == 1).mean())


def thresholds_for(scores, train_pool_scores, y_true, prior: float) -> dict[str, float]:
    """All four rules at one prior.

    prior_transfer is the primary: it fixes the cut on the *training* score distribution
    and carries it over unchanged. On Vaxign, which is 75.7% positive, an in-sample prior
    cut would force 25% predicted-positive on a three-quarters-positive set and cap
    sensitivity by construction -- that would measure the rule, not the model.
    """
    return {
        "prior_transfer": prior_threshold(train_pool_scores, prior),
        "prior_insample": prior_threshold(scores, prior),
        "fixed_0.5": 0.5,
        "mcc_optimal": mcc_optimal_threshold(y_true, scores),
    }


def evaluate_labelled(name: str, y_true, scores, train_pool_scores,
                      priors) -> list[dict]:
    """Supervised + PU metrics for a set whose labels are known (test, Vaxign)."""
    rows = []
    rate = observed_positive_rate(y_true)
    for prior in priors:
        cuts = thresholds_for(scores, train_pool_scores, y_true, prior)
        for rule, threshold in cuts.items():
            supervised = classification_metrics(y_true, scores, threshold)
            pu = pu_metrics(np.asarray(y_true, dtype=int), scores, threshold, prior)
            rows.append({
                "dataset": name, "class_prior": prior, "threshold_rule": rule,
                "uses_labels": THRESHOLD_RULES[rule],
                "observed_positive_rate": rate,
                # Below the observed rate the label frequency saturates at 1, the prior
                # inside the unlabeled pool goes to zero, and every PU correction becomes
                # an identity. Flagged rather than silently reported.
                "degenerate_prior": bool(prior <= rate),
                **supervised,
                # pu_metrics' only key collision with classification_metrics is "n";
                # dropping it (and the redundant observed_positives) keeps every PU column
                # under the same name here as in the train-pool table.
                **{k: v for k, v in pu.items() if k not in ("n", "observed_positives")},
            })
    return rows


def evaluate_train_pool(observed, scores, priors,
                        p_scores_are_inbag: bool = True) -> list[dict]:
    """PU metrics only: the training pool's true labels are, by assumption, unknown."""
    rows = []
    for prior in priors:
        threshold = prior_threshold(scores, prior)
        pu = pu_metrics(np.asarray(observed, dtype=int), scores, threshold, prior)
        rows.append({
            "dataset": "train_oob", "class_prior": prior,
            "threshold_rule": "prior_insample", "uses_labels": False,
            "observed_positive_rate": observed_positive_rate(observed),
            "degenerate_prior": bool(prior <= observed_positive_rate(observed)),
            "threshold": threshold,
            # True only when p_ratio == 1.0 puts every positive in every bag, leaving them
            # with no out-of-bag score so their saturated in-bag mean has to stand in. Then
            # proxy_auc on the training pool is optimistic for P and is not comparable with
            # the test-set value. The caller knows which case it is; this must not be
            # hardcoded, or the artifact claims a caveat the run does not have.
            "p_scores_are_inbag": bool(p_scores_are_inbag),
            **pu,
        })
    return rows
