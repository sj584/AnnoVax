"""InterProScan TSV -> the annotation text AnnoVax embeds.

Two steps, matching Methods 2.4. First every signature matched on a protein is pooled
into one pipe-joined string, taking the *Signature description* (column 6) and
*InterPro description* (column 13) fields. Those overlap heavily -- the same domain is
described once per member database -- so the second step drops any description whose
cosine similarity to an already-kept one is at or above the threshold, under
all-MiniLM-L6-v2.

Ported from the author's VFTextSeq preprocessing, minus the MMseqs2/GTDB taxonomy half:
AnnoVax takes no taxonomy feature.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

#: The 15 columns InterProScan writes in TSV mode, in order. Descriptions are optional
#: and come through as "-" when the member database supplies none.
INTERPROSCAN_COLUMNS = [
    "ID", "Sequence MD5 Digest", "Sequence Length", "Analysis",
    "Signature Accession", "Signature Description", "Start", "Stop", "Score",
    "Status", "Date", "InterPro Accession", "InterPro Description",
    "GO Annotations", "Pathway Annotations",
]

#: The two description fields pooled into `desc` (columns 6 and 13, 1-based).
DESCRIPTION_COLUMNS = ("Signature Description", "InterPro Description")

#: Cosine similarity at or above which a description is treated as a duplicate.
#: See README: the manuscript reports 0.80.
DEFAULT_SIMILARITY_THRESHOLD = 0.80

#: What BioMistral is fed when a protein has no InterProScan hit at all. The published
#: embedding cache was built with this literal, so it is kept rather than an empty string.
EMPTY_ANNOTATION = "NaN"


def read_fasta(path) -> pd.DataFrame:
    from Bio import SeqIO

    rows = [(record.id, str(record.seq)) for record in SeqIO.parse(str(path), "fasta")]
    if not rows:
        raise ValueError(f"no sequences parsed from {path}")
    return pd.DataFrame(rows, columns=["ID", "Sequence"])


def read_table(path) -> pd.DataFrame:
    """A FASTA or a CSV carrying at least ID and Sequence."""
    path = Path(path)
    if path.suffix.lower() in {".fasta", ".fa", ".faa", ".fna"}:
        return read_fasta(path)
    frame = pd.read_csv(path)
    missing = {"ID", "Sequence"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks column(s) {sorted(missing)}")
    return frame


def parse_interproscan(tsv_path) -> pd.DataFrame:
    """Read an InterProScan TSV. Rows vary in width when trailing fields are absent."""
    frame = pd.read_csv(tsv_path, sep="\t", header=None, comment="#",
                        names=INTERPROSCAN_COLUMNS, usecols=range(13),
                        dtype=str, engine="python")
    frame["ID"] = frame["ID"].astype(str)
    return frame


def build_descriptions(interpro: pd.DataFrame) -> dict[str, str]:
    """ID -> pipe-joined set of descriptions, sorted for reproducibility."""
    descriptions = {}
    for protein_id, group in interpro.groupby("ID"):
        terms = set()
        for column in DESCRIPTION_COLUMNS:
            if column not in group:
                continue
            for value in group[column].dropna():
                value = str(value).strip()
                if value and value != "-":
                    terms.add(value)
        descriptions[str(protein_id)] = "|".join(sorted(terms))
    return descriptions


def remove_semantic_duplicates(desc, model,
                               similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
    """Greedily drop near-duplicate descriptions from one pipe-joined string.

    Longest first, so the survivor of a duplicate pair is the more specific wording.
    """
    from sentence_transformers import util

    if not desc or pd.isna(desc):
        return desc
    terms = sorted(str(desc).split("|"), key=len, reverse=True)
    if len(terms) < 2:
        return "|".join(terms)

    embeddings = model.encode(terms, convert_to_tensor=True)
    keep = [True] * len(terms)
    for i in range(len(terms)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(terms)):
            if keep[j] and util.pytorch_cos_sim(embeddings[i], embeddings[j]).item() >= similarity_threshold:
                keep[j] = False
    return "|".join(term for term, kept in zip(terms, keep) if kept)


def preprocess(sequences, interproscan_tsv,
               similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
               model_name: str = "all-MiniLM-L6-v2",
               progress: bool = True) -> pd.DataFrame:
    """ID / Sequence / desc / desc_nodup, ready for scripts/extract_embedding.py."""
    from sentence_transformers import SentenceTransformer

    frame = read_table(sequences).copy()
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].duplicated().any():
        duplicated = frame.loc[frame["ID"].duplicated(), "ID"].unique()[:5]
        raise ValueError(f"duplicate IDs in {sequences}: {list(duplicated)}")

    descriptions = build_descriptions(parse_interproscan(interproscan_tsv))
    frame["desc"] = frame["ID"].map(descriptions).fillna("")

    annotated = int((frame["desc"] != "").sum())
    print(f"[preprocess] {annotated}/{len(frame)} proteins carry an InterProScan hit")

    model = SentenceTransformer(model_name)
    iterator = frame["desc"]
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(frame), desc="semantic dedup")
        except ImportError:
            pass
    frame["desc_nodup"] = [remove_semantic_duplicates(d, model, similarity_threshold)
                           for d in iterator]
    # A protein with no hits still needs a text to embed; see EMPTY_ANNOTATION.
    frame.loc[frame["desc_nodup"] == "", "desc_nodup"] = EMPTY_ANNOTATION
    return frame
