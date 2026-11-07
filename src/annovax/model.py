"""The released artifact: K saved boosters plus everything needed to score a new protein.

`run_bagging` trains the ensemble and throws the boosters away unless it is given a
`save_dir`. This module is the other half: it writes the manifest that says what those
boosters expect as input, and reads the pair back as something with a `.predict`.

The manifest matters more than it looks. A bag file on its own does not record which
encoders produced its 6656 columns, in what order, or where the decision threshold came
from -- and a silently reordered feature block produces scores that look plausible and
mean nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as cfgmod

MANIFEST_NAME = "manifest.json"
BAGS_DIRNAME = "bags"
#: Bumped only when the on-disk layout changes in a way an older reader would misread.
FORMAT_VERSION = 1


def _bag_paths(bags_dir: Path) -> list[Path]:
    paths = sorted(bags_dir.glob("bag_*.ubj"))
    if not paths:
        raise FileNotFoundError(f"no bag_*.ubj under {bags_dir}")
    return paths


def write_manifest(model_dir: Path, *, arm: str, threshold: float,
                   headline_prior: float, n_bags: int,
                   train_rows: int, train_positives: int,
                   input_digests: dict[str, str] | None = None,
                   extra: dict | None = None) -> dict:
    """Record what the saved boosters expect, so a wrong input fails instead of scoring."""
    import xgboost as xgb

    blocks = cfgmod.ARMS[arm]
    manifest = {
        "format_version": FORMAT_VERSION,
        "model": "AnnoVax",
        "arm": arm,
        # Order is load-bearing: data.EmbeddingRepo hstacks in exactly this sequence.
        "feature_blocks": [{"encoder": enc, "dim": cfgmod.ENCODER_DIM[enc]}
                           for enc in blocks],
        "feature_dim": cfgmod.arm_dim(arm),
        "n_bags": int(n_bags),
        "headline_prior": float(headline_prior),
        # The prior_transfer cut: fixed on the training-pool score distribution and
        # carried over unchanged. It reads no labels, which is why it can ship.
        "threshold": float(threshold),
        "threshold_rule": "prior_transfer",
        "train_rows": int(train_rows),
        "train_positives": int(train_positives),
        "xgboost_version": xgb.__version__,
        "numpy_version": np.__version__,
        "input_sha256": input_digests or {},
    }
    if extra:
        manifest.update(extra)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass
class AnnoVaxEnsemble:
    """K boosters scored by their mean probability, plus the shipped threshold."""

    boosters: list
    manifest: dict

    @property
    def arm(self) -> str:
        return self.manifest["arm"]

    @property
    def threshold(self) -> float:
        return float(self.manifest["threshold"])

    @property
    def feature_dim(self) -> int:
        return int(self.manifest["feature_dim"])

    @property
    def encoders(self) -> tuple[str, ...]:
        return tuple(block["encoder"] for block in self.manifest["feature_blocks"])

    def __len__(self) -> int:
        return len(self.boosters)

    @classmethod
    def load(cls, model_dir, *, limit: int | None = None) -> "AnnoVaxEnsemble":
        import xgboost as xgb

        model_dir = Path(model_dir)
        manifest_path = model_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} is missing -- a bags/ directory alone does not say which "
                "encoders or threshold it was built for. Re-train with --save-models, or "
                "download the published model with scripts/download_weights.sh.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format_version", 1) > FORMAT_VERSION:
            raise ValueError(
                f"{manifest_path} is format v{manifest['format_version']}, but this "
                f"annovax reads up to v{FORMAT_VERSION}. Upgrade the package.")

        paths = _bag_paths(model_dir / BAGS_DIRNAME)
        if limit is not None:
            paths = paths[:limit]
        boosters = []
        for path in paths:
            booster = xgb.Booster()
            booster.load_model(str(path))
            boosters.append(booster)

        declared = manifest.get("n_bags")
        if limit is None and declared is not None and declared != len(boosters):
            raise RuntimeError(
                f"{model_dir}: manifest declares {declared} bags but {len(boosters)} "
                "files are present -- the download or the training run was truncated.")
        return cls(boosters=boosters, manifest=manifest)

    def _dmatrix(self, X: np.ndarray):
        import xgboost as xgb

        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
        if X.shape[1] != self.feature_dim:
            blocks = ", ".join(f"{b['encoder']}({b['dim']})"
                               for b in self.manifest["feature_blocks"])
            raise ValueError(
                f"expected {self.feature_dim} features ({blocks}), got {X.shape[1]}. "
                "The embedding blocks must be concatenated in the manifest's order.")
        return xgb.DMatrix(X)

    def predict(self, X: np.ndarray, *, return_bags: bool = False):
        """Mean probability over the bags; `return_bags` also gives the K x n matrix.

        The DMatrix is built once and reused across all K boosters -- rebuilding it per
        bag turns a single conversion into K of them, which dominates the runtime.
        """
        handle = self._dmatrix(X)
        bags = np.empty((len(self.boosters), handle.num_row()), dtype=np.float32)
        for row, booster in enumerate(self.boosters):
            bags[row] = booster.predict(handle)
        mean = bags.mean(axis=0)
        return (mean, bags) if return_bags else mean

    def predict_frame(self, ids, X: np.ndarray):
        """ID / score_mean / score_std / pred, in the input's own order."""
        import pandas as pd

        mean, bags = self.predict(X, return_bags=True)
        return pd.DataFrame({
            "ID": list(ids),
            "score_mean": mean,
            "score_std": bags.std(axis=0),
            "pred": (mean >= self.threshold).astype(int),
        })
