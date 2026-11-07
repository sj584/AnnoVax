"""Tables + embeddings, aligned by ID.

Only ID / Sequence / Label may be relied on: the Vaxign table carries desc / desc_nodup
and has no Taxon column.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfgmod


@dataclass(frozen=True)
class Split:
    name: str
    X: np.ndarray          # (n, 6656) float32, C-contiguous
    y: np.ndarray          # (n,) int  -- the table's Label column
    ids: list[str]
    frame: pd.DataFrame    # row-aligned with X; keeps Sequence and annotation columns

    @property
    def n_positive(self) -> int:
        return int((self.y == 1).sum())


class EmbeddingRepo:
    """Concatenates several mean-pooled .npz caches, looked up by protein ID.

    Each cache stores parallel `ids` and `X`. Row order in the npz is never assumed:
    alignment goes through a dict and is imposed by the order of the IDs requested.
    """

    def __init__(self, *npz_paths: Path):
        self.paths = [Path(p) for p in npz_paths]
        self.blocks = []
        for path in self.paths:
            loaded = np.load(path, allow_pickle=True)
            ids = loaded["ids"].astype(str)
            self.blocks.append(({pid: row for row, pid in enumerate(ids)}, loaded["X"]))

    @property
    def dim(self) -> int:
        return sum(block.shape[1] for _, block in self.blocks)

    def matrix(self, ids: list[str]) -> tuple[np.ndarray, list[str]]:
        """Returns (X, missing_ids). X covers only the IDs present in every block."""
        missing = [pid for pid in ids
                   if any(pid not in index for index, _ in self.blocks)]
        dropped = set(missing)
        kept = [pid for pid in ids if pid not in dropped]
        parts = [block[[index[pid] for pid in kept]] for index, block in self.blocks]
        X = np.ascontiguousarray(np.hstack(parts), dtype=np.float32)
        return X, missing


def _read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")
    return frame


def _build(name: str, frame: pd.DataFrame, repo: EmbeddingRepo, arm: str) -> Split:
    X, missing = repo.matrix(frame["ID"].tolist())
    if missing:
        # Dropping is the only safe response: a zero row sits at the feature-space origin
        # and XGBoost routes it into whichever leaf the majority class occupies, which
        # manufactures a confident but meaningless score. Known-missing IDs are removed
        # up front in load_all via cfgmod.EXCLUDE_IDS, so anything reaching here is new
        # and has to stop the run rather than silently shrink the split.
        raise RuntimeError(f"{name}: {len(missing)} IDs lack an embedding: {missing[:5]}")
    expected = cfgmod.arm_dim(arm)
    if X.shape[1] != expected:
        raise ValueError(f"{name}/{arm}: expected {expected} features, got {X.shape[1]}")
    return Split(name=name, X=X, y=frame["Label"].to_numpy(dtype=int),
                 ids=frame["ID"].tolist(), frame=frame.reset_index(drop=True))


@dataclass(frozen=True)
class Datasets:
    train: Split
    test: Split
    vaxign: Split
    overlap: dict


def _overlap_report(train: pd.DataFrame, test: pd.DataFrame,
                    vaxign: pd.DataFrame) -> dict:
    """Vaxign shares material with PLGDL. Recorded, not acted on."""
    def counts(a: pd.DataFrame, b: pd.DataFrame) -> dict:
        return {"by_id": len(set(a["ID"]) & set(b["ID"])),
                "by_sequence": len(set(a["Sequence"]) & set(b["Sequence"]))}
    return {
        "train_vs_test": counts(train, test),
        "train_vs_vaxign": counts(train, vaxign),
        "test_vs_vaxign": counts(test, vaxign),
        "note": ("Vaxign is evaluated in full, nothing excluded. These counts are recorded "
                 "so the 19 sequence-identical, all-positive proteins shared with PLGDL "
                 "(~11% of Vaxign) are on the record when its numbers are quoted."),
    }


def _drop_excluded(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Remove cfgmod.EXCLUDE_IDS, reporting what was actually dropped.

    Applied to every arm identically, so the three ablation arms run on the same rows.
    """
    hit = frame["ID"].isin(cfgmod.EXCLUDE_IDS)
    if hit.any():
        print(f"[data] {name}: excluding {sorted(frame.loc[hit, 'ID'])} "
              f"(cfgmod.EXCLUDE_IDS, applied to all arms)", flush=True)
    return frame.loc[~hit].reset_index(drop=True)


def load_all(arm: str = "esmc_biomistral", train_variant: str = "surfmatched",
             test_variant: str = "surfmatched",
             vaxign_variant: str = "surfmatched") -> Datasets:
    train_frame = _drop_excluded("train", _read_table(cfgmod.TRAIN_VARIANTS[train_variant]))
    test_frame = _drop_excluded("test", _read_table(cfgmod.TEST_VARIANTS[test_variant]))
    vaxign_frame = _drop_excluded("vaxign", _read_table(cfgmod.VAXIGN_VARIANTS[vaxign_variant]))

    plgdl = EmbeddingRepo(*cfgmod.arm_caches(arm, "plgdl"))
    vaxign_repo = EmbeddingRepo(*cfgmod.arm_caches(arm, "vaxign"))

    train = _build("train", train_frame, plgdl, arm)
    test = _build("test", test_frame, plgdl, arm)
    vaxign = _build("vaxign", vaxign_frame, vaxign_repo, arm)

    overlap = _overlap_report(train_frame, test_frame, vaxign_frame)
    if overlap["train_vs_test"]["by_sequence"] or overlap["train_vs_test"]["by_id"]:
        raise RuntimeError(f"train and test overlap: {overlap['train_vs_test']}")
    return Datasets(train=train, test=test, vaxign=vaxign, overlap=overlap)


def file_digests(arm: str = "esmc_biomistral", train_variant: str = "surfmatched",
                 test_variant: str = "surfmatched") -> dict[str, str]:
    """sha256 of every input file this arm actually read, for the provenance record."""
    paths = [cfgmod.TRAIN_VARIANTS[train_variant], cfgmod.TEST_VARIANTS[test_variant],
             cfgmod.VAXIGN_CSV,
             *cfgmod.arm_caches(arm, "plgdl"), *cfgmod.arm_caches(arm, "vaxign")]
    digests = {}
    for path in paths:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digests[str(path)] = digest.hexdigest()
    return digests
