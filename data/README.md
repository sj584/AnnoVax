# Datasets

Every table here carries sequences and labels only. The embeddings the model actually
consumes are not in this repository — build them with `scripts/extract_embedding.py` or
fetch the published caches with `scripts/download_weights.sh`.

## Columns

| Column | Meaning |
|---|---|
| `ID` | UniProt / RefSeq / GenBank accession. The join key for every table and cache. |
| `Sequence` | Amino-acid sequence. |
| `Label` | 1 = experimentally supported protective antigen, 0 = unlabeled. |
| `Taxon` | `Bacteria`, `Eukaryota` or `Viruses` (PLGDL tables only). |
| `desc` | InterProScan descriptions for the protein, pipe-joined. |
| `desc_nodup` | `desc` after semantic deduplication — **this is what BioMistral-7B embeds**. |

`Label = 0` means *unlabeled*, not *verified non-protective*. These proteins were never
tested for protection; treating them as hard negatives is exactly the bias the PU setup
exists to avoid. The one exception is the Vaxign benchmark, whose negatives were reviewed
against the literature.

The PLGDL split tables carry no `desc` column. Annotation text for all 6,600 PLGDL
proteins lives in `PLGDL/PLGDL_interproscan_no_dup_semantic.csv` and is joined by `ID`.

## PLGDL

Positives are protective antigens from the Protegen database plus a PubMed literature
search, each retained only on in vivo protection evidence or an antigen-specific immune
response tied to protection, then clustered with CD-HIT v4.8.1 at 50% identity. Negatives
are UniProt proteins reduced at the same 50% cutoff, with anything sharing >20% similarity
to a positive removed by BLASTP (v2.14.0, E < 1e-5). The 1:10 positive-to-negative ratio
is deliberate: protective antigens are scarce in a real proteome.

| File | n | positive | unlabeled |
|---|---|---|---|
| `PLGDL/All/train_All.csv` | 5,280 | 480 | 4,800 |
| `PLGDL/All/test_All.csv` | 1,320 | 120 | 1,200 |
| `PLGDL/Bacteria/train_Bacteria.csv` | 3,922 | 311 | 3,611 |
| `PLGDL/Bacteria/test_Bacteria.csv` | 957 | 75 | 882 |
| `PLGDL/Eukaryota/train_Eukaryota.csv` | 888 | 77 | 811 |
| `PLGDL/Eukaryota/test_Eukaryota.csv` | 233 | 18 | 215 |
| `PLGDL/Virus/train_Viruses.csv` | 470 | 92 | 378 |
| `PLGDL/Virus/test_Viruses.csv` | 130 | 27 | 103 |

The taxon files partition the `All` split; they are the per-kingdom readout behind Table 3.
The Eukaryota test set holds 18 positives, which is too few for a stable threshold-
dependent metric — read those rows with that in mind.

## PLGDL, surface-matched — *the model is trained on this*

Roughly 3 positives in 4 are surface-exposed against 1 in 4 unlabeled proteins, which a
classifier can exploit as a shortcut: call anything surface-exposed an antigen and score
well without learning anything else. Non-surface unlabeled proteins were therefore dropped
at random until the surface ratio matched between classes.

| File | n | positive | unlabeled |
|---|---|---|---|
| `PLGDL_surface_matched/train_All_surfmatched.csv` | 1,948 | 480 | 1,468 |
| `PLGDL_surface_matched/test_All_surfmatched.csv` | 429 | 120 | 309 |

`surface_flags.csv` carries the per-protein surface call for all 6,600 proteins — `surface`,
the matched `surface_categories` and `surface_terms`, and `kept`, which records whether the
protein survived matching. Categories were assigned by keyword match against the
InterProScan text.

## Vaxign independent benchmark

Merged from two earlier collections of bacterial protective antigens, 100BPA (Dalsass et
al.) and 200BPA (originally VaxiJen, extended by Bowman et al. and Heinson et al.), with
Protegen duplicates removed. Its negatives were checked against the literature and kept
only where no protection was reported, so they approximate true non-protective proteins.
Curated independently of Protegen and used at no stage of training or model selection.

| File | n | positive | negative |
|---|---|---|---|
| `Vaxign_benchmark/Vaxign_benchmark.csv` | 246 | 131 | 115 |
| `Vaxign_benchmark/Vaxign_benchmark_surfmatched.csv` | 173 | 131 | 42 |

The full 246-protein set is the Table 4 benchmark. Nineteen of its proteins are sequence-
identical to PLGDL entries and all are positives (~11% of the set); `annovax.data` reports
this overlap on every load rather than silently dropping them.

## Two things to know before reproducing

**`P68462` has no ESM-C 6B embedding.** That one remote extraction call failed, so the
published `plgdl__esmc_6b.npz` holds 6,599 rows where every other cache holds 6,600. It is
excluded from *every* arm (`annovax.config.EXCLUDE_IDS`) so the ablation arms train on
identical rows. It appears only in the training table, which therefore contributes 1,947
usable rows rather than 1,948. Clear `EXCLUDE_IDS` if you re-extract and the call succeeds.

**The shipped annotation text is from the InterProScan run used for the paper.** Re-running
InterProScan today gives slightly different wording — member databases revise descriptions
between releases. The text here is what the published embeddings were built from, so use it
as-is to reproduce the paper's numbers, and re-derive it only when annotating new proteins.
