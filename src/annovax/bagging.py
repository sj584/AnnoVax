"""PU bagging (Mordelet & Vert 2014) with GPU XGBoost.

xgboost is imported lazily inside the worker functions, never at module scope: a spawned
child must set CUDA_VISIBLE_DEVICES *before* the library initializes its CUDA context.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Worker-process globals, populated by _init_worker.
_G: dict = {}


def bag_rng(seed: int, k: int) -> np.random.Generator:
    """Bag k's generator.

    Seeded on the pair, not on seed + k: additive seeds give correlated streams, and the
    symptom (some unlabeled protein never landing out-of-bag) is easy to miss.
    """
    return np.random.default_rng([seed, k])


def _init_worker(gpu: int, arrays_path: str, threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import xgboost as xgb  # noqa: PLC0415 -- must follow the env var

    arrays = np.load(arrays_path)
    _G["xgb"] = xgb
    _G["X_p"] = arrays["X_p"]
    _G["X_u"] = arrays["X_u"]
    # One DMatrix per evaluation set, built once and reused by every bag. Rebuilding them
    # per bag would be ~1000 redundant host->device copies; cupy is not installed here, so
    # this is the copy that actually matters.
    _G["d_p"] = xgb.DMatrix(arrays["X_p"])
    _G["d_u"] = xgb.DMatrix(arrays["X_u"])
    _G["d_test"] = xgb.DMatrix(arrays["X_test"])
    _G["d_vaxign"] = xgb.DMatrix(arrays["X_vaxign"])
    _G["gpu"] = gpu


def _run_block(job: dict) -> dict:
    xgb = _G["xgb"]
    X_p, X_u = _G["X_p"], _G["X_u"]
    n_p, n_u = len(X_p), len(X_u)
    n_sub, n_psub = job["n_sub"], job["n_psub"]
    params_base, rounds, max_bin = job["params"], job["rounds"], job["max_bin"]
    bags = job["bags"]

    y_bag = np.concatenate([np.ones(n_psub, dtype=np.float32),
                            np.zeros(n_sub, dtype=np.float32)])
    out = {name: np.empty((len(bags), size), dtype=np.float32) for name, size in
           (("p", n_p), ("u", n_u), ("test", _G["d_test"].num_row()),
            ("vaxign", _G["d_vaxign"].num_row()))}
    oob = np.ones((len(bags), n_u), dtype=bool)
    oob_p = np.ones((len(bags), n_p), dtype=bool)
    best_iters = np.zeros(len(bags), dtype=np.int32)

    start = time.time()
    for position, k in enumerate(bags):
        rng = bag_rng(job["seed"], k)
        # P is drawn first so that a run with p_ratio=1.0 consumes the generator in the
        # same order as before only when it actually subsamples; the U draw follows.
        psub = (np.arange(n_p) if n_psub == n_p
                else rng.choice(n_p, size=n_psub, replace=False))
        oob_p[position, psub] = False
        sub = rng.choice(n_u, size=n_sub, replace=False)
        oob[position, sub] = False

        X_bag = np.ascontiguousarray(np.vstack([X_p[psub], X_u[sub]]))
        # QuantileDMatrix is the matched input for tree_method="hist": it sketches straight
        # into the bin-compressed form instead of materializing a full CSR first.
        dtrain = xgb.QuantileDMatrix(X_bag, label=y_bag, max_bin=max_bin)
        params = dict(params_base, seed=int(rng.integers(0, 2**31 - 1)))
        if job["scale_pos_weight"] is not None:
            params["scale_pos_weight"] = (n_sub / n_psub
                                          if job["scale_pos_weight"] == "auto"
                                          else float(job["scale_pos_weight"]))
        booster = xgb.train(params, dtrain, num_boost_round=rounds)
        best_iters[position] = rounds

        # Persisting the booster is what makes inference on new proteins possible at all:
        # without it the ensemble exists only for the lifetime of this process, and
        # scoring a new FASTA would mean retraining all K bags. UBJSON rather than JSON
        # for size, and rather than pickle so the artifact survives an xgboost upgrade.
        if job["save_dir"] is not None:
            booster.save_model(os.path.join(job["save_dir"], f"bag_{k:05d}.ubj"))

        for name, handle in (("p", _G["d_p"]), ("u", _G["d_u"]),
                             ("test", _G["d_test"]), ("vaxign", _G["d_vaxign"])):
            out[name][position] = booster.predict(handle)

        if job["verbose"] and (position + 1) % 50 == 0:
            rate = (time.time() - start) / (position + 1)
            print(f"  [gpu {_G['gpu']}] {position + 1}/{len(bags)} bags, "
                  f"{rate:.2f} s/bag", flush=True)
    return {"bags": np.asarray(bags), "oob": oob, "oob_p": oob_p,
            "best_iters": best_iters, **out}


@dataclass(frozen=True)
class BagResult:
    """Per-bag score matrices, K x n. Retained in full: collapsing to running sums would
    forfeit the convergence curve, the rank-stability figures and the per-bag spread."""
    bags: np.ndarray
    scores_p: np.ndarray
    scores_u: np.ndarray
    scores_test: np.ndarray
    scores_vaxign: np.ndarray
    oob_mask: np.ndarray
    oob_mask_p: np.ndarray
    seconds: float

    @property
    def oob_count(self) -> np.ndarray:
        return self.oob_mask.sum(axis=0)

    def oob_stats_p(self) -> tuple[np.ndarray, np.ndarray]:
        """OOB mean/std for the observed positives. All-NaN when p_ratio == 1.0, which is
        exactly the blind spot that subsampling P exists to remove."""
        masked = np.where(self.oob_mask_p, self.scores_p, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(masked, axis=0), np.nanstd(masked, axis=0)

    def oob_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """OOB mean and the *actual* across-bag std.

        The reference implementation stored a binomial placeholder p(1-p) here; retaining
        the full matrix means the real spread is available for nothing.
        """
        masked = np.where(self.oob_mask, self.scores_u, np.nan)
        with warnings.catch_warnings():
            # An all-NaN column means that protein was never out-of-bag. At K=1000 the
            # probability is 0.327**1000, so it only shows up in smoke runs -- but it must
            # not crash, and a silent zero would be worse than a NaN the caller can see.
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(masked, axis=0), np.nanstd(masked, axis=0)


def run_bagging(X_p, X_u, X_test, X_vaxign, cfg, scratch_dir: Path,
                verbose: bool = True, save_dir: Path | None = None) -> BagResult:
    """Train cfg.k bags and score P / U / test / Vaxign with each.

    With `save_dir` set, every bag's booster is written to `save_dir/bag_XXXXX.ubj` as it
    is trained; `model.AnnoVaxEnsemble.load` reads that directory back.
    """
    n_psub = max(1, int(round(getattr(cfg, "p_ratio", 1.0) * len(X_p))))
    n_sub = max(1, int(round(cfg.ratio * n_psub)))
    if n_sub > len(X_u):
        raise ValueError(f"bag size {n_sub} exceeds the unlabeled pool {len(X_u)}")

    scratch_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = scratch_dir / "arrays.npz"
    np.savez(arrays_path, X_p=X_p, X_u=X_u, X_test=X_test, X_vaxign=X_vaxign)

    gpus = list(cfg.gpus)
    # Contiguous blocks, not round-robin: each bag's RNG depends on its global index alone,
    # so the split across workers cannot change any result.
    blocks = [list(range(start, min(start + -(-cfg.k // len(gpus)), cfg.k)))
              for start in range(0, cfg.k, -(-cfg.k // len(gpus)))]
    blocks = [b for b in blocks if b]
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    jobs = [{"bags": block, "seed": cfg.seed, "n_sub": n_sub, "n_psub": n_psub,
             "scale_pos_weight": getattr(cfg, "scale_pos_weight", None),
             "params": cfg.xgb_params(cfg.seed), "rounds": cfg.num_boost_round,
             "max_bin": cfg.max_bin, "verbose": verbose,
             "save_dir": None if save_dir is None else str(save_dir)}
            for block in blocks]

    started = time.time()
    if len(jobs) == 1:
        _init_worker(gpus[0], str(arrays_path), 8)
        results = [_run_block(jobs[0])]
    else:
        # spawn, never fork: a forked child inherits the parent's CUDA context and would
        # either crash or quietly train on the wrong device.
        context = mp.get_context("spawn")
        with context.Pool(len(jobs), initializer=_init_worker,
                          initargs=(gpus[0], str(arrays_path), 8)) as pool:
            results = pool.map(_init_and_run, [(gpu, str(arrays_path), job)
                                               for gpu, job in zip(gpus, jobs)])
    elapsed = time.time() - started

    order = np.argsort(np.concatenate([r["bags"] for r in results]))
    stack = lambda key: np.concatenate([r[key] for r in results])[order]  # noqa: E731
    return BagResult(
        bags=np.concatenate([r["bags"] for r in results])[order],
        scores_p=stack("p"), scores_u=stack("u"),
        scores_test=stack("test"), scores_vaxign=stack("vaxign"),
        oob_mask=stack("oob"), oob_mask_p=stack("oob_p"), seconds=elapsed,
    )


def _init_and_run(payload):
    gpu, arrays_path, job = payload
    if _G.get("gpu") != gpu:
        _init_worker(gpu, arrays_path, 8)
    return _run_block(job)
