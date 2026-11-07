"""Metrics that exist only because the full K x n bag matrices were retained."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .metrics.standard import classification_metrics


def per_example_stats(bag_scores: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "score_mean": bag_scores.mean(axis=0),
        "score_std": bag_scores.std(axis=0),
        "score_q05": np.percentile(bag_scores, 5, axis=0),
        "score_q50": np.percentile(bag_scores, 50, axis=0),
        "score_q95": np.percentile(bag_scores, 95, axis=0),
    }


def spread_summary(bag_scores: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    std = bag_scores.std(axis=0)
    y_true = np.asarray(y_true, dtype=int)
    return {
        "mean_score_std": float(std.mean()),
        "median_score_std": float(np.median(std)),
        "score_std_positive": float(std[y_true == 1].mean()) if (y_true == 1).any() else float("nan"),
        "score_std_negative": float(std[y_true == 0].mean()) if (y_true == 0).any() else float("nan"),
    }


def convergence(bag_scores: np.ndarray, y_true: np.ndarray, threshold: float,
                points) -> list[dict]:
    """Headline metrics on the running mean over the first k bags.

    This is the direct evidence for whether K was large enough; a curve still moving at
    the end means the ensemble has not converged, whatever the final number looks like.
    """
    rows = []
    for k in points:
        if k > len(bag_scores):
            continue
        scores = bag_scores[:k].mean(axis=0)
        metrics = classification_metrics(y_true, scores, threshold)
        rows.append({"k": int(k), "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
                     "mcc": metrics["mcc"], "balanced_accuracy": metrics["balanced_accuracy"]})
    return rows


def per_bag_metrics(bag_scores: np.ndarray, y_true: np.ndarray,
                    threshold: float) -> list[dict]:
    """Each individual bag scored on its own. The gap to the ensemble is the bagging gain."""
    return [{"bag": int(k), **{key: classification_metrics(y_true, bag_scores[k], threshold)[key]
                               for key in ("roc_auc", "pr_auc", "mcc", "balanced_accuracy")}}
            for k in range(len(bag_scores))]


def rank_stability(bag_scores: np.ndarray, ks=(10, 25, 50, 100),
                   pairs: int = 200, seed: int = 0) -> dict:
    """Spearman agreement of single bags with the ensemble, and top-k set overlap.

    The half-split overlap is the useful one: if the first half and the second half of the
    bags disagree about the top candidates, the ranking is noise.
    """
    ensemble = bag_scores.mean(axis=0)
    rhos = [spearmanr(bag_scores[k], ensemble).statistic for k in range(len(bag_scores))]
    result = {"spearman_bag_vs_ensemble_mean": float(np.mean(rhos)),
              "spearman_bag_vs_ensemble_sd": float(np.std(rhos))}

    half = len(bag_scores) // 2
    if half >= 1:
        first, second = bag_scores[:half].mean(axis=0), bag_scores[half:].mean(axis=0)
        result["spearman_half_split"] = float(spearmanr(first, second).statistic)
        for k in ks:
            k = min(k, len(ensemble))
            top_a = set(np.argsort(-first)[:k])
            top_b = set(np.argsort(-second)[:k])
            result[f"half_split_jaccard_top{k}"] = len(top_a & top_b) / len(top_a | top_b)

    rng = np.random.default_rng(seed)
    if len(bag_scores) >= 2:
        for k in ks:
            k = min(k, len(ensemble))
            overlaps = []
            for _ in range(min(pairs, len(bag_scores) * (len(bag_scores) - 1) // 2)):
                i, j = rng.choice(len(bag_scores), size=2, replace=False)
                top_a = set(np.argsort(-bag_scores[i])[:k])
                top_b = set(np.argsort(-bag_scores[j])[:k])
                overlaps.append(len(top_a & top_b) / len(top_a | top_b))
            result[f"pairwise_jaccard_top{k}"] = float(np.mean(overlaps))
    return result


def bagging_gain(bag_scores: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    single = [roc_auc_score(y_true, bag_scores[k]) for k in range(len(bag_scores))]
    ensemble = roc_auc_score(y_true, bag_scores.mean(axis=0))
    return {"single_bag_roc_auc_mean": float(np.mean(single)),
            "single_bag_roc_auc_sd": float(np.std(single)),
            "ensemble_roc_auc": float(ensemble),
            "bagging_gain": float(ensemble - np.mean(single))}
