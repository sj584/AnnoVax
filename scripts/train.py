#!/usr/bin/env python
"""Train AnnoVax: PU bagging + GPU XGBoost, evaluated on the PLGDL test split and Vaxign.

With --save-models the K boosters and their manifest are written out, which is what
scripts/predict.py loads to score new proteins.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annovax import ensemble, evaluate, model, report               # noqa: E402
from annovax.bagging import run_bagging                              # noqa: E402
from annovax.config import (ARMS, PRIOR_GRID, TEST_VARIANTS,   # noqa: E402
                           TRAIN_VARIANTS, VAXIGN_VARIANTS, RunConfig)               # noqa: E402
from annovax.data import file_digests, load_all                      # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="esmc_biomistral", choices=sorted(ARMS),
                        help="embedding arm; the ablation trains all three")
    parser.add_argument("--train-variant", default="surfmatched",
                        choices=sorted(TRAIN_VARIANTS))
    parser.add_argument("--test-variant", default="surfmatched",
                        choices=sorted(TEST_VARIANTS))
    parser.add_argument("--vaxign-variant", default="surfmatched",
                        choices=sorted(VAXIGN_VARIANTS))
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="negatives per bag = ratio * positives per bag")
    parser.add_argument("--p-ratio", type=float, default=1.0,
                        help="fraction of P drawn into each bag; below 1.0 gives the "
                             "positives out-of-bag scores and unsaturates the training pool")
    parser.add_argument("--gpus", default="2", help="comma-separated physical GPU ids")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--priors", default=",".join(str(p) for p in PRIOR_GRID))
    parser.add_argument("--headline-prior", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--save-models", default=None, metavar="DIR",
                        help="write every bag's booster plus a manifest here, so the "
                             "ensemble can be reloaded by scripts/predict.py")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = RunConfig(
        tag=args.tag, arm=args.arm, train_variant=args.train_variant,
        test_variant=args.test_variant, seed=args.seed, k=args.k, ratio=args.ratio,
        p_ratio=args.p_ratio,
        gpus=tuple(int(g) for g in args.gpus.split(",") if g != ""),
        priors=tuple(float(p) for p in args.priors.split(",")),
        headline_prior=args.headline_prior, num_boost_round=args.rounds,
        bootstrap_resamples=args.bootstrap,
    )
    out = cfg.out_root / cfg.run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run] {cfg.run_id} -> {out}", flush=True)

    data = load_all(cfg.arm, cfg.train_variant, cfg.test_variant, args.vaxign_variant)
    train = data.train
    is_positive = train.y == 1
    X_p, X_u = train.X[is_positive], train.X[~is_positive]
    ids_p = [i for i, flag in zip(train.ids, is_positive) if flag]
    ids_u = [i for i, flag in zip(train.ids, is_positive) if not flag]
    print(f"[data] arm={cfg.arm} P={len(X_p)} U={len(X_u)} test={len(data.test.X)} "
          f"vaxign={len(data.vaxign.X)} dim={train.X.shape[1]}", flush=True)

    model_dir = Path(args.save_models) if args.save_models else None
    bags = run_bagging(X_p, X_u, data.test.X, data.vaxign.X, cfg, out / "scratch",
                       save_dir=None if model_dir is None else model_dir / "bags")
    print(f"[bag] {cfg.k} bags in {bags.seconds / 60:.1f} min "
          f"({bags.seconds / cfg.k:.2f} s/bag)", flush=True)

    oob_mean, oob_std = bags.oob_stats()
    oob_p_mean, _ = bags.oob_stats_p()
    inbag_p = bags.scores_p.mean(axis=0)
    # With p_ratio == 1.0 the positives are in every bag and have no out-of-bag score, so
    # their in-bag mean has to stand in and the training pool is saturated. Below 1.0 the
    # real OOB score exists and is used instead.
    pool_p = oob_p_mean if np.isfinite(oob_p_mean).any() else inbag_p
    p_scores_are_inbag = not np.isfinite(oob_p_mean).any()
    print(f"[pool] positive scores are {'in-bag (saturated)' if p_scores_are_inbag else 'out-of-bag'}",
          flush=True)
    # The training pool as the model sees it: observed positives scored in-bag (they are in
    # every bag by construction), unlabeled scored out-of-bag.
    never_oob = ~np.isfinite(oob_mean)
    if never_oob.any():
        print(f"[oob] {int(never_oob.sum())} unlabeled proteins were never out-of-bag and "
              f"are excluded from the training-pool PU metrics (expected only at small K; "
              f"at K=1000 the probability is ~0.327**1000)", flush=True)
    usable_p = np.isfinite(pool_p)
    pool_scores = np.concatenate([pool_p[usable_p], oob_mean[~never_oob]])
    pool_observed = np.concatenate([np.ones(int(usable_p.sum()), int),
                                    np.zeros(int((~never_oob).sum()), int)])

    scores_test = bags.scores_test.mean(axis=0)
    scores_vaxign = bags.scores_vaxign.mean(axis=0)

    rows = (evaluate.evaluate_labelled("test", data.test.y, scores_test, pool_scores, cfg.priors)
            + evaluate.evaluate_labelled("vaxign", data.vaxign.y, scores_vaxign, pool_scores,
                                         cfg.priors)
            + evaluate.evaluate_train_pool(pool_observed, pool_scores, cfg.priors,
                                           p_scores_are_inbag=p_scores_are_inbag))
    long = pd.DataFrame(rows)
    long.to_csv(out / "metrics_long.csv", index=False)
    long[long.threshold_rule.isin(["prior_transfer", "prior_insample"])].to_csv(
        out / "prior_sweep.csv", index=False)

    headline_cut = evaluate.thresholds_for(scores_test, pool_scores, data.test.y,
                                           cfg.headline_prior)["prior_transfer"]
    print(f"[thr] prior_transfer @ pi={cfg.headline_prior} -> {headline_cut:.4f}", flush=True)

    if model_dir is not None:
        # Written only now: the threshold is derived from the training-pool scores, which
        # do not exist until every bag has been trained and scored.
        model.write_manifest(
            model_dir, arm=cfg.arm, threshold=headline_cut,
            headline_prior=cfg.headline_prior, n_bags=cfg.k,
            train_rows=len(train.X), train_positives=int(is_positive.sum()),
            input_digests=file_digests(cfg.arm, cfg.train_variant, cfg.test_variant),
            extra={"run_id": cfg.run_id, "train_variant": cfg.train_variant,
                   "seed": cfg.seed, "ratio": cfg.ratio, "p_ratio": cfg.p_ratio})
        print(f"[model] {cfg.k} boosters + manifest -> {model_dir}", flush=True)

    ens = {}
    for name, bag_scores, y_true in (("test", bags.scores_test, data.test.y),
                                     ("vaxign", bags.scores_vaxign, data.vaxign.y)):
        per_bag = ensemble.per_bag_metrics(bag_scores, y_true, headline_cut)
        pd.DataFrame(per_bag).to_csv(out / f"per_bag_metrics_{name}.csv", index=False)
        pd.DataFrame(report.bag_variability(per_bag)).to_csv(
            out / f"bag_variability_{name}.csv", index=False)
        pd.DataFrame(ensemble.convergence(bag_scores, y_true, headline_cut,
                                          cfg.convergence_points)).to_csv(
            out / f"convergence_{name}.csv", index=False)
        pd.DataFrame(report.bootstrap_metric_ci(y_true, bag_scores.mean(axis=0), headline_cut,
                                                cfg.bootstrap_resamples, seed=cfg.seed)).to_csv(
            out / f"bootstrap_ci_{name}.csv", index=False)
        ens[name] = {**ensemble.spread_summary(bag_scores, y_true),
                     **ensemble.rank_stability(bag_scores, seed=cfg.seed),
                     **ensemble.bagging_gain(bag_scores, y_true)}

    for name, split, bag_scores in (("test", data.test, bags.scores_test),
                                    ("vaxign", data.vaxign, bags.scores_vaxign)):
        frame = pd.DataFrame({"ID": split.ids, "Label": split.y,
                              **ensemble.per_example_stats(bag_scores)})
        frame["pred_prior_transfer"] = (frame.score_mean >= headline_cut).astype(int)
        frame.sort_values("score_mean", ascending=False).to_csv(
            out / f"scores_{name}.csv", index=False)

    pd.DataFrame({"ID": ids_u, "oob_mean": oob_mean, "oob_std": oob_std,
                  "oob_count": bags.oob_count}).sort_values(
        "oob_mean", ascending=False).to_csv(out / "scores_train_oob.csv", index=False)
    # Both columns: with p_ratio < 1 the OOB mean is what the pool actually used, and
    # saving only the in-bag mean made the positives' real scores unrecoverable.
    oob_p_std = bags.oob_stats_p()[1]
    pd.DataFrame({"ID": ids_p, "inbag_mean": inbag_p,
                  "inbag_std": bags.scores_p.std(axis=0),
                  "oob_mean": oob_p_mean, "oob_std": oob_p_std,
                  "oob_count": bags.oob_mask_p.sum(axis=0)}).to_csv(
        out / "scores_train_positive.csv", index=False)

    np.savez_compressed(out / "bag_scores.npz",
                        test=bags.scores_test.astype(np.float16),
                        vaxign=bags.scores_vaxign.astype(np.float16),
                        u=bags.scores_u.astype(np.float16),
                        p=bags.scores_p.astype(np.float16),
                        # Without this the positives' OOB pool cannot be rebuilt from the
                        # artifacts, which is the whole point of retaining the matrices.
                        oob_mask_p=bags.oob_mask_p,
                        oob_mask=bags.oob_mask)

    report.write_json(out / "metrics.json", {
        "run_id": cfg.run_id, "headline_prior": cfg.headline_prior,
        "headline_threshold_prior_transfer": headline_cut,
        "seconds": bags.seconds, "seconds_per_bag": bags.seconds / cfg.k,
        "oob": {"count_min": int(bags.oob_count.min()),
                "count_mean": float(bags.oob_count.mean()),
                "count_sd": float(bags.oob_count.std())},
        "ensemble": ens, "vaxign_overlap": data.overlap,
        "headline_rows": long[(long.class_prior == cfg.headline_prior)
                              & (long.threshold_rule == "prior_transfer")].to_dict("records"),
    })
    report.write_json(out / "config.json", cfg.serializable())
    report.write_json(out / "env.json", report.environment_record(cfg, file_digests(cfg.arm, cfg.train_variant, cfg.test_variant)))
    for leftover in (out / "scratch").glob("*.npz"):
        leftover.unlink()
    (out / "scratch").rmdir()

    head = long[(long.dataset.isin(["test", "vaxign"]))
                & (long.class_prior == cfg.headline_prior)
                & (long.threshold_rule == "prior_transfer")]
    print("\n=== headline (prior_transfer @ pi=%.2f) ===" % cfg.headline_prior)
    print(head[["dataset", "roc_auc", "pr_auc", "balanced_accuracy", "mcc", "f1",
                "precision", "sensitivity"]].to_string(index=False))
    print(f"\n[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
