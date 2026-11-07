"""Bootstrap intervals over examples, and every artifact writer."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics.standard import classification_metrics

HEADLINE = ("roc_auc", "pr_auc", "balanced_accuracy", "mcc", "f1", "precision",
            "sensitivity", "specificity")


def bootstrap_metric_ci(y_true, scores, threshold: float, n_resamples: int = 2000,
                        alpha: float = 0.05, seed: int = 0,
                        metrics=HEADLINE) -> list[dict]:
    """Stratified bootstrap over evaluation examples.

    Resampling within the positive and negative strata keeps n_pos and n_neg fixed; an
    unstratified resample occasionally draws a single class and roc_auc comes back NaN.
    The threshold is frozen, not refitted per resample: refitting would fold threshold
    variance into what is meant to be the metric's sampling variance.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        values = classification_metrics(y_true[idx], scores[idx], threshold)
        for metric in metrics:
            draws[metric].append(values[metric])
    point = classification_metrics(y_true, scores, threshold)
    rows = []
    for metric in metrics:
        sample = np.asarray([v for v in draws[metric] if np.isfinite(v)])
        rows.append({
            "metric": metric, "point": point[metric],
            "ci_low": float(np.percentile(sample, 100 * alpha / 2)) if len(sample) else float("nan"),
            "ci_high": float(np.percentile(sample, 100 * (1 - alpha / 2))) if len(sample) else float("nan"),
            "B": int(len(sample)),
        })
    return rows


def bag_variability(per_bag: list[dict]) -> list[dict]:
    """Spread of the individual bags.

    This is a property of the estimator, not an interval for the ensemble: the ensemble is
    a deterministic function of all K bags, and its uncertainty is the example-level
    bootstrap above. Conflating the two is the standard bagging-paper error.
    """
    frame = pd.DataFrame(per_bag)
    rows = []
    for metric in ("roc_auc", "pr_auc", "mcc", "balanced_accuracy"):
        values = frame[metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append({"metric": metric, "mean": float(values.mean()),
                     "sd": float(values.std()), "p5": float(np.percentile(values, 5)),
                     "p50": float(np.percentile(values, 50)),
                     "p95": float(np.percentile(values, 95)),
                     "note": "across-bag spread, NOT a CI for the ensemble"})
    return rows


def environment_record(cfg, digests: dict) -> dict:
    import sklearn
    import xgboost
    upstream = Path(__file__).with_name("UPSTREAM_SHA1.txt")
    try:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as error:                                   # noqa: BLE001
        gpus = f"unavailable: {error}"
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__, "host": platform.node(),
        "gpus_requested": list(cfg.gpus), "nvidia_smi": gpus,
        "vendored_metrics_upstream_sha1": upstream.read_text() if upstream.exists() else None,
        "input_sha256": digests,
    }


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")
