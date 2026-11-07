#!/usr/bin/env python
"""Preprocessed CSV -> mean-pooled embedding caches (one .npz per encoder).

    python scripts/extract_embedding.py -i example/example_preprocessed.csv \
        -o embeddings --corpus example

writes embeddings/example__esmc_6b.npz and embeddings/example__biomistral_7b.npz, the
pair the released model needs. ESM-C 6B is served remotely and needs $ESM_API_KEY; the
published caches for PLGDL and Vaxign can be fetched with scripts/download_weights.sh
instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annovax import config as cfgmod                       # noqa: E402
from annovax.embed import ENCODERS, TEXT_COLUMN, save_cache  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True,
                        help="preprocessed CSV with ID, Sequence and desc_nodup")
    parser.add_argument("--output-dir", "-o", default=str(cfgmod.CACHE_ROOT),
                        help=f"default: {cfgmod.CACHE_ROOT}")
    parser.add_argument("--corpus", "-c", default=None,
                        help="cache name prefix, as in <corpus>__<encoder>.npz "
                             "(default: the input file's stem)")
    parser.add_argument("--encoders", "-e", default="esmc_6b,biomistral_7b",
                        help="comma-separated; choose from " + ", ".join(ENCODERS))
    parser.add_argument("--device", default="cuda", help="for the local encoders")
    parser.add_argument("--token", default=None,
                        help="ESM API token; $ESM_API_KEY is used when omitted")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild caches that already exist")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input)
    frame["ID"] = frame["ID"].astype(str)
    corpus = args.corpus or Path(args.input).stem
    out_dir = Path(args.output_dir)

    requested = [name.strip() for name in args.encoders.split(",") if name.strip()]
    unknown = [name for name in requested if name not in ENCODERS]
    if unknown:
        parser.error(f"unknown encoder(s) {unknown}; choose from {sorted(ENCODERS)}")

    for name in requested:
        encode, dim, column = ENCODERS[name]
        path = out_dir / f"{corpus}__{name}.npz"
        if path.exists() and not args.overwrite:
            print(f"[skip] {path} exists (--overwrite to rebuild)")
            continue
        if column not in frame.columns:
            parser.error(f"{name} reads column {column!r}, absent from {args.input}. "
                         f"Run scripts/preprocess.py first to produce {TEXT_COLUMN}.")

        print(f"[run] {name} over {len(frame)} proteins -> {path}")
        kwargs = {"token": args.token} if name == "esmc_6b" else {"device": args.device}
        ids, X, failed = encode(frame, **kwargs)
        if failed:
            # Not fatal here, but loading any split containing these IDs will raise.
            print(f"[warn] {len(failed)} protein(s) produced no {name} embedding: "
                  f"{failed[:5]}{' ...' if len(failed) > 5 else ''}")
        save_cache(path, ids, X, expected_dim=dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
