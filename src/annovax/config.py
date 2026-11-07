"""Run configuration. Every default is materialized into config.json at run time."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

#: Mean-pooled embedding caches. Not shipped in the repository -- build them with
#: scripts/extract_embedding.py, or fetch the published ones with
#: scripts/download_weights.sh. $ANNOVAX_CACHE overrides the location.
CACHE_ROOT = Path(os.environ.get("ANNOVAX_CACHE", PROJECT_ROOT / "embeddings"))

TRAIN_CSV = DATA_ROOT / "PLGDL_surface_matched" / "train_All_surfmatched.csv"
TEST_CSV = DATA_ROOT / "PLGDL_surface_matched" / "test_All_surfmatched.csv"
VAXIGN_CSV = DATA_ROOT / "Vaxign_benchmark" / "Vaxign_benchmark_surfmatched.csv"

#: "surfmatched" is the headline setting throughout the paper; "original" is PLGDL's
#: published split (5280 train / 1320 test) and the full 246-protein Vaxign benchmark.
TRAIN_VARIANTS = {"surfmatched": TRAIN_CSV,
                  "original": DATA_ROOT / "PLGDL" / "All" / "train_All.csv"}
TEST_VARIANTS = {"surfmatched": TEST_CSV,
                 "original": DATA_ROOT / "PLGDL" / "All" / "test_All.csv"}
VAXIGN_VARIANTS = {"surfmatched": VAXIGN_CSV,
                   "original": DATA_ROOT / "Vaxign_benchmark" / "Vaxign_benchmark.csv"}

#: Taxon-stratified PLGDL splits, for per-kingdom training or evaluation (paper Table 3).
TAXON_SPLITS = {
    "Bacteria": (DATA_ROOT / "PLGDL" / "Bacteria" / "train_Bacteria.csv",
                 DATA_ROOT / "PLGDL" / "Bacteria" / "test_Bacteria.csv"),
    "Eukaryota": (DATA_ROOT / "PLGDL" / "Eukaryota" / "train_Eukaryota.csv",
                  DATA_ROOT / "PLGDL" / "Eukaryota" / "test_Eukaryota.csv"),
    "Virus": (DATA_ROOT / "PLGDL" / "Virus" / "train_Viruses.csv",
              DATA_ROOT / "PLGDL" / "Virus" / "test_Viruses.csv"),
}

#: InterProScan annotation text, keyed by ID. The PLGDL split tables carry no `desc`
#: column; the text is joined in from here at embedding time.
ANNOTATION_CSV = {
    "plgdl": DATA_ROOT / "PLGDL" / "PLGDL_interproscan_no_dup_semantic.csv",
    "vaxign": DATA_ROOT / "Vaxign_benchmark" / "Vaxign_benchmark.csv",
}

#: Mean-pooled embedding caches, keyed by corpus then encoder. Each .npz holds exactly
#: `ids` (str) and `X` (float32, (n, D)).
CACHES = {
    corpus: {enc: CACHE_ROOT / f"{corpus}__{enc}.npz"
             for enc in ("esmc_6b", "biomistral_7b", "esm2_3b", "biobert")}
    for corpus in ("plgdl", "vaxign")
}

#: Mean-pooled width per encoder, asserted at load time so a swapped cache fails loudly.
#: ESM-2 is the 3B checkpoint -- ESM-2 ships 650M / 3B / 15B and has no 6B variant.
ENCODER_DIM = {"esmc_6b": 2560, "biomistral_7b": 4096, "esm2_3b": 2560, "biobert": 768}

#: The ablation arms (paper Table 2). Blocks concatenate in the order listed, fixed so a
#: feature index means the same thing across runs. `esmc_biomistral` is the released
#: model; esm2_3b and biobert are the weaker-encoder controls, one generation back from
#: ESM-C 6B and BioMistral-7B respectively.
ARMS = {
    "esmc_biomistral": ("esmc_6b", "biomistral_7b"),
    "esmc": ("esmc_6b",),
    "biomistral": ("biomistral_7b",),
    "esm2_3b": ("esm2_3b",),
    "biobert": ("biobert",),
}

#: P68462 has no ESM-C 6B embedding: that one remote extraction call errored out, so the
#: published plgdl__esmc_6b.npz has 6599 rows where every other cache has 6600. It is
#: dropped from *every* arm, including the BioMistral-only one that could keep it, so all
#: arms train on identical rows and a difference between them is the encoder rather than
#: one extra training protein. It occurs only in the training table. If you re-extract the
#: embeddings yourself and the call succeeds, clear this set.
EXCLUDE_IDS = frozenset({"P68462"})


def arm_caches(arm: str, corpus: str) -> tuple[Path, ...]:
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    return tuple(CACHES[corpus][key] for key in ARMS[arm])


def arm_dim(arm: str) -> int:
    return sum(ENCODER_DIM[key] for key in ARMS[arm])


#: ESM-C 6B (2560) ++ BioMistral-7B (4096), the released model.
FEATURE_DIM = arm_dim("esmc_biomistral")

#: Requested grid, plus the supplementary priors. Anything at or below the observed
#: positive rate (0.246 train / 0.280 test) makes the PU correction a no-op; those rows are
#: flagged `degenerate_prior` rather than silently reported.
PRIOR_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


@dataclass(frozen=True)
class RunConfig:
    tag: str = "main"
    #: Which embedding arm to train on; see ARMS. Part of run_id so the three arms of an
    #: ablation cannot overwrite one another.
    arm: str = "esmc_biomistral"
    #: Which training / test table (see TRAIN_VARIANTS, TEST_VARIANTS). Recorded in
    #: config.json; put it in `tag` too so run_ids of different cells cannot collide.
    train_variant: str = "surfmatched"
    test_variant: str = "surfmatched"
    seed: int = 0
    k: int = 1000
    #: bag size from U = ratio * (positives in the bag), drawn without replacement.
    ratio: float = 1.0
    #: Fraction of P drawn into each bag, without replacement. 1.0 is the Mordelet & Vert
    #: default and keeps every positive in every bag -- which leaves the positives with no
    #: out-of-bag score at all, saturating their in-bag scores and corrupting any threshold
    #: or PU metric read off the training pool. Below 1.0 the positives get OOB scores too.
    p_ratio: float = 1.0
    gpus: tuple[int, ...] = (2,)
    priors: tuple[float, ...] = PRIOR_GRID
    headline_prior: float = 0.25

    #: Frozen from scripts/tune_rounds.py: median best_iteration over 20 bags was 114,
    #: rounded up to the nearest 50. Not early-stopped per bag -- see README.
    num_boost_round: int = 150
    max_depth: int = 4
    eta: float = 0.1
    subsample: float = 0.8
    #: 0.3 rather than the usual 0.8: 6656 features against ~960 rows is severely p >> n,
    #: so column subsampling is the main variance-reduction lever.
    colsample_bytree: float = 0.3
    #: None leaves the loss unweighted. "auto" sets it to (negatives/positives) inside the
    #: bag, which rebalances the loss when ratio != 1. Tree splits shift; AUC (rank-based)
    #: and the quantile thresholds are largely insensitive to it.
    scale_pos_weight: float | str | None = None
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    #: 64 rather than 256: measured 1.3 s/bag against 3.4 s, and 960 rows cannot support
    #: 256 distinguishable split points anyway.
    max_bin: int = 64

    bootstrap_resamples: int = 2000
    convergence_points: tuple[int, ...] = (5, 10, 25, 50, 100, 250, 500, 750, 1000)
    out_root: Path = PROJECT_ROOT / "results"

    def xgb_params(self, seed: int) -> dict:
        return {
            "objective": "binary:logistic", "eval_metric": "logloss",
            # gpu_hist was removed in xgboost 3.x; device + hist is the replacement.
            "device": "cuda", "tree_method": "hist",
            "max_depth": self.max_depth, "eta": self.eta,
            "subsample": self.subsample, "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight, "reg_lambda": self.reg_lambda,
            "max_bin": self.max_bin, "seed": seed, "verbosity": 0,
        }

    #: Stamped once in __post_init__. A property recomputing datetime.now() on every
    #: access would let the output directory and the run_id inside metrics.json disagree.
    run_id: str = ""

    def __post_init__(self):
        if not self.run_id:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
            object.__setattr__(
                self, "run_id",
                f"{self.tag}_{self.arm}_K{self.k}_r{self.ratio:g}_s{self.seed}_{stamp}")

    def serializable(self) -> dict:
        values = asdict(self)
        values["out_root"] = str(values["out_root"])
        values["config_sha1"] = hashlib.sha1(
            json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return values
