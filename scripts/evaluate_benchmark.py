#!/usr/bin/env python
"""Score a saved AnnoVax model on the paper's benchmarks and print the metric table.

    python scripts/evaluate_benchmark.py -m models/annovax_K1000 -e embeddings

Covers the PLGDL test split (Table 1), its taxon strata (Table 3) and the Vaxign
independent benchmark (Table 4). The decision threshold is the model's own shipped
prior_transfer cut; it reads no labels, so nothing here tunes on the test set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annovax import config as cfgmod                                   # noqa: E402
from annovax.data import EmbeddingRepo                                 # noqa: E402
from annovax.metrics.pu_metrics import pu_f1, proxy_auc                # noqa: E402
from annovax.metrics.standard import classification_metrics            # noqa: E402
from annovax.model import AnnoVaxEnsemble                              # noqa: E402

#: Top-K% cutoffs for the early-enrichment measure reported in the paper.
RECALL_PERCENTS = (5, 10, 20)

REPORT_COLUMNS = ["dataset", "n", "positives", "roc_auc", "pr_auc",
                  "balanced_accuracy", "f1", "precision", "sensitivity", "mcc",
                  *[f"recall_at_{k}pct" for k in RECALL_PERCENTS], "pu_f1"]


def recall_at_percent(y_true, scores, percent: float) -> float:
    """Fraction of positives inside the top `percent`% of the ranking."""
    y_true = np.asarray(y_true, dtype=int)
    if not y_true.sum():
        return float("nan")
    k = max(1, int(np.ceil(len(scores) * percent / 100)))
    top = y_true[np.argsort(-np.asarray(scores, dtype=float))[:k]]
    return float(top.sum() / y_true.sum())


def evaluate(name, y_true, scores, threshold) -> dict:
    row = {"dataset": name, **classification_metrics(y_true, scores, threshold)}
    for percent in RECALL_PERCENTS:
        row[f"recall_at_{percent}pct"] = recall_at_percent(y_true, scores, percent)
    # PU-adjusted F1 (Lee & Liu): recall^2 / Pr[yhat = 1], computable without a verified
    # negative, which is the point -- the unlabeled set is expected to hide positives.
    row["pu_f1"] = pu_f1(np.asarray(y_true, dtype=int), np.asarray(scores), threshold)
    row["proxy_auc"] = proxy_auc(np.asarray(y_true, dtype=int), np.asarray(scores))
    return row


def score_table(ensemble, repo, frame, name, threshold) -> tuple[dict, pd.DataFrame]:
    frame = frame.loc[~frame["ID"].astype(str).isin(cfgmod.EXCLUDE_IDS)].reset_index(drop=True)
    ids = frame["ID"].astype(str).tolist()
    X, missing = repo.matrix(ids)
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} IDs lack an embedding, e.g. {missing[:5]}")
    scores = ensemble.predict(X)
    out = frame[["ID", "Label"]].copy()
    out["score_mean"] = scores
    if "Taxon" in frame.columns:
        out["Taxon"] = frame["Taxon"].to_numpy()
    return evaluate(name, frame["Label"].to_numpy(dtype=int), scores, threshold), out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--embedding-dir", "-e", default=str(cfgmod.CACHE_ROOT))
    parser.add_argument("--test-variant", default="surfmatched",
                        choices=sorted(cfgmod.TEST_VARIANTS))
    parser.add_argument("--vaxign-variant", default="original",
                        choices=sorted(cfgmod.VAXIGN_VARIANTS),
                        help="'original' is the full 246-protein benchmark of Table 4")
    parser.add_argument("--output", "-o", default=None, help="write the table as CSV")
    parser.add_argument("--scores-dir", default=None,
                        help="also write the per-protein scores for each set here")
    args = parser.parse_args(argv)

    ensemble = AnnoVaxEnsemble.load(args.model)
    threshold = ensemble.threshold
    print(f"[model] {len(ensemble)} bags, arm={ensemble.arm}, threshold={threshold:.4f}\n")

    cache_dir = Path(args.embedding_dir)
    caches = lambda corpus: [cache_dir / f"{corpus}__{enc}.npz" for enc in ensemble.encoders]  # noqa: E731
    plgdl, vaxign = EmbeddingRepo(*caches("plgdl")), EmbeddingRepo(*caches("vaxign"))

    rows, per_protein = [], {}
    test = pd.read_csv(cfgmod.TEST_VARIANTS[args.test_variant])
    row, scores = score_table(ensemble, plgdl, test, "PLGDL test", threshold)
    rows.append(row)
    per_protein["plgdl_test"] = scores

    # Table 3: one ALL-trained model, read out on each taxon of the same test split.
    if "Taxon" in scores.columns:
        for taxon, group in scores.groupby("Taxon"):
            if group["Label"].nunique() == 2:
                rows.append(evaluate(f"  PLGDL test / {taxon}", group["Label"].to_numpy(),
                                     group["score_mean"].to_numpy(), threshold))

    vax = pd.read_csv(cfgmod.VAXIGN_VARIANTS[args.vaxign_variant])
    row, scores = score_table(ensemble, vaxign, vax, f"Vaxign ({args.vaxign_variant})", threshold)
    rows.append(row)
    per_protein["vaxign"] = scores

    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table[REPORT_COLUMNS].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\n[done] {args.output}")
    if args.scores_dir:
        target = Path(args.scores_dir)
        target.mkdir(parents=True, exist_ok=True)
        for name, frame in per_protein.items():
            frame.sort_values("score_mean", ascending=False).to_csv(
                target / f"scores_{name}.csv", index=False)
        print(f"[done] per-protein scores -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
