#!/usr/bin/env python
"""FASTA + InterProScan TSV -> *_preprocessed.csv (ID, Sequence, desc, desc_nodup).

Produce the TSV first, e.g. with InterProScan 6 through Nextflow:

    nextflow run ebi-pf-team/interproscan6 -profile docker \
        --input proteins.fasta --formats tsv --outdir interproscan_out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from annovax.preprocess import DEFAULT_SIMILARITY_THRESHOLD, preprocess  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True,
                        help="protein FASTA, or a CSV with ID and Sequence columns")
    parser.add_argument("--interproscan", "-ip", required=True,
                        help="InterProScan TSV for those proteins")
    parser.add_argument("--output", "-o", default=None,
                        help="default: <input stem>_preprocessed.csv beside the input")
    parser.add_argument("--similarity-threshold", type=float,
                        default=DEFAULT_SIMILARITY_THRESHOLD,
                        help=f"cosine cut for semantic dedup (default {DEFAULT_SIMILARITY_THRESHOLD})")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args(argv)

    source = Path(args.input)
    out = Path(args.output) if args.output else source.with_name(f"{source.stem}_preprocessed.csv")
    frame = preprocess(source, args.interproscan, args.similarity_threshold, args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[done] {len(frame)} proteins -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
