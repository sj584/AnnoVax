#!/usr/bin/env python
"""Score proteins with a trained AnnoVax ensemble.

    python scripts/predict.py -i example/example_preprocessed.csv \
        -e embeddings -m models/annovax_K1000 -o example/result.csv

The embedding directory must hold one cache per encoder the model was trained on, named
<corpus>__<encoder>.npz (see scripts/extract_embedding.py). Every input ID has to appear
in every one of them: a protein with no embedding is an error, never a zero row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annovax.data import EmbeddingRepo           # noqa: E402
from annovax.model import AnnoVaxEnsemble        # noqa: E402
from annovax.preprocess import read_table        # noqa: E402


def resolve_caches(embedding_dir: Path, encoders, corpus: str | None) -> list[Path]:
    """Find one .npz per encoder, requiring a --corpus only when the choice is ambiguous."""
    paths = []
    for encoder in encoders:
        if corpus:
            candidate = embedding_dir / f"{corpus}__{encoder}.npz"
            if not candidate.exists():
                raise FileNotFoundError(f"{candidate} not found")
            paths.append(candidate)
            continue
        matches = sorted(embedding_dir.glob(f"*__{encoder}.npz"))
        if not matches:
            raise FileNotFoundError(
                f"no *__{encoder}.npz in {embedding_dir}. Build it with "
                "scripts/extract_embedding.py, or fetch the published caches with "
                "scripts/download_weights.sh.")
        if len(matches) > 1:
            names = ", ".join(p.name for p in matches)
            raise ValueError(f"several {encoder} caches in {embedding_dir} ({names}); "
                             "pick one with --corpus")
        paths.append(matches[0])
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True,
                        help="FASTA, or a CSV with an ID column")
    parser.add_argument("--embedding-dir", "-e", required=True)
    parser.add_argument("--model", "-m", required=True,
                        help="model directory holding manifest.json and bags/")
    parser.add_argument("--output", "-o", default=None,
                        help="default: <input stem>_annovax.csv beside the input")
    parser.add_argument("--corpus", "-c", default=None,
                        help="cache prefix, when the embedding directory holds several")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the model's shipped decision threshold")
    parser.add_argument("--bags", type=int, default=None,
                        help="score with only the first N bags (a faster, noisier run)")
    args = parser.parse_args(argv)

    ensemble = AnnoVaxEnsemble.load(args.model, limit=args.bags)
    threshold = args.threshold if args.threshold is not None else ensemble.threshold
    print(f"[model] {len(ensemble)} bags, arm={ensemble.arm}, "
          f"dim={ensemble.feature_dim}, threshold={threshold:.4f}")

    frame = read_table(args.input)
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].duplicated().any():
        raise ValueError(f"duplicate IDs in {args.input}")

    caches = resolve_caches(Path(args.embedding_dir), ensemble.encoders, args.corpus)
    print("[features] " + " ++ ".join(p.name for p in caches))
    repo = EmbeddingRepo(*caches)
    X, missing = repo.matrix(frame["ID"].tolist())
    if missing:
        raise RuntimeError(
            f"{len(missing)} protein(s) have no embedding in every cache, e.g. "
            f"{missing[:5]}. Extract them first -- scoring them would mean inventing "
            "features, and the score would look confident and mean nothing.")

    scores = ensemble.predict_frame(frame["ID"].tolist(), X)
    if args.threshold is not None:
        scores["pred"] = (scores["score_mean"] >= threshold).astype(int)
    if "Label" in frame.columns:
        scores["Label"] = frame["Label"].to_numpy()

    source = Path(args.input)
    out = Path(args.output) if args.output else source.with_name(f"{source.stem}_annovax.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)
    print(f"[done] {len(scores)} proteins, {int(scores.pred.sum())} predicted protective "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
