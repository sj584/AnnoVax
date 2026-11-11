# AnnoVax

**Protective antigen prediction with interpretable InterProScan annotation and sparse
autoencoders under positive-unlabeled learning.**

AnnoVax ranks pathogen proteins by how likely they are to be protective antigens — the
bottleneck step in subunit vaccine design, where a proteome offers thousands of candidates
and animal challenge experiments can test only a handful.

Two things make it work. Each protein is represented by **two modalities**: its sequence,
encoded with ESM-C 6B, and its InterProScan annotation text, encoded with BioMistral-7B.
And because proteins that have never been tested for protection are *unlabeled* rather
than non-protective, the model is trained and evaluated as a **positive-unlabeled** problem
— 1,000 bootstrap-resampled XGBoost bags whose predictions are averaged.

## Install

```bash
git clone https://github.com/sj584/AnnoVax.git
cd AnnoVax

conda env create -f environment.yml
conda activate annovax
```

Scoring proteins from published embeddings needs only numpy/pandas/xgboost. Torch,
transformers and the ESM SDK are needed only if you build embeddings yourself.

## Quick start — score your own proteins

Model weights and embedding caches: Zenodo [10.5281/zenodo.24095960](https://doi.org/10.5281/zenodo.24095960).

```bash
# 0. model weights and the published embedding caches
bash scripts/download_weights.sh

# 1. annotate (InterProScan 6; several hours for a large FASTA)
#    --datadir is required: it holds the member-database files (Pfam, CDD, PANTHER, ...),
#    about 34 GB, which InterProScan downloads into that directory on the first run.
nextflow run ebi-pf-team/interproscan6 -profile docker \
    --input my_proteins.fasta --datadir interproscan_data \
    --formats tsv --outdir interproscan_out

# 2. pool the annotations into one text per protein and drop near-duplicates
python scripts/preprocess.py -i my_proteins.fasta \
    -ip interproscan_out/my_proteins.fasta.tsv

# 3. embed (ESM-C 6B is served remotely and needs a token)
export ESM_API_KEY=...          # or pass --token; see "Notes on the encoders"
python scripts/extract_embedding.py -i my_proteins_preprocessed.csv \
    -o embeddings --corpus mine

# 4. score
python scripts/predict.py -i my_proteins_preprocessed.csv \
    -e embeddings -c mine -m models/annovax_K1000 -o predictions.csv
```

Steps 1–3 are the expensive part. Step 4 takes seconds.

### Output

| ID | score_mean | score_std | pred |
|---|---|---|---|
| YP_016503.2 | 0.8412 | 0.0671 | 1 |
| P29280 | 0.1104 | 0.0388 | 0 |

`score_mean` is the mean over the 1,000 bags and is what you rank by. `score_std` is their
disagreement — a high score with a wide spread rests on fewer bags than a tight one.
`pred` applies the model's shipped threshold, chosen on the *training* score distribution
so that the predicted-positive rate matches the assumed class prior (π = 0.25). It reads no
labels. For screening a proteome, rank by `score_mean` and take as many candidates as you
can test, rather than trusting the cut.

A worked example with all intermediate files is in [`example/`](example/).

## Repository layout

```
data/                 sequences, labels and annotation text — see data/README.md
src/annovax/
  config.py           paths, encoder arms, XGBoost hyperparameters
  preprocess.py       InterProScan TSV -> desc -> semantic dedup
  embed.py            encoders -> mean-pooled .npz caches
  data.py             tables + embeddings, joined by ID
  bagging.py          PU bagging; saves one booster per bag
  model.py            AnnoVaxEnsemble: save / load / predict
  thresholds.py       the label-free threshold rules
  evaluate.py         metric assembly
  ensemble.py         convergence, rank stability, bagging gain
  metrics/            standard and PU metrics
scripts/              preprocess / extract_embedding / train / predict / evaluate_benchmark
results/predictions/  AnnoVax scores on the PLGDL test and Vaxign sets — see its README
tests/                pytest suite (no GPU required)
```

## Training

```bash
python scripts/train.py --arm esmc_biomistral --k 1000 --ratio 2.0 --p-ratio 0.8 \
    --seed 0 --gpus 0,2 --save-models models/annovax_K1000
```

~26 minutes on two RTX A6000s. Each of the 1,000 bags draws 80% of the positives and twice
that many unlabeled proteins, without replacement, and trains an XGBoost classifier
(depth 4, η 0.1, 150 rounds, no early stopping). `--save-models` writes every booster plus
a `manifest.json` recording the encoders, their concatenation order and the threshold —
without it the ensemble exists only for the lifetime of the process.

Drawing 80% rather than all of the positives is deliberate: it leaves each positive
out-of-bag in some bags, which is the only way to get an honest score for the training
positives and hence a threshold that is not read off saturated predictions.

### Reproducing the paper's tables

```bash
python scripts/evaluate_benchmark.py -m models/annovax_K1000 -e embeddings
```

prints the PLGDL test split (Table 1), its taxon strata (Table 3) and the Vaxign
independent benchmark (Table 4). The ablation of Table 2 is a training flag:

```bash
for arm in esmc_biomistral esmc biomistral esm2_3b biobert; do
    python scripts/train.py --arm "$arm" --k 1000 --ratio 2.0 --p-ratio 0.8 --tag ablation
done
```

The published caches cover ESM-C 6B and BioMistral-7B, so the first three arms run as they
stand. The `esm2_3b` and `biobert` controls need their own caches, which are not published:
build them with `scripts/extract_embedding.py -e esm2_3b,biobert` for both the `plgdl` and
`vaxign` corpora first. Both encoders run locally and need no API token.

Both PLGDL splits ship in two variants: `--train-variant original` is PLGDL's published
5,280/1,320 split, and `surfmatched` (the default, and what the released model uses) is the
surface-ratio-matched subset. The reason is in [`data/README.md`](data/README.md).

## Evaluation metrics

Standard metrics (ROC-AUC, PR-AUC, balanced accuracy, F1, MCC) are reported alongside PU
metrics, because precision measured against unlabeled proteins is systematically
*understated* — some of those proteins are protective antigens nobody has tested yet, so
a correct prediction on one is scored as a false positive.

So AnnoVax also reports **Recall@K%** (K = 5, 10, 20), the share of known positives inside
the top K% of the ranking, which is what proteome screening actually cares about; and
**PU-F1** (Lee & Liu), `recall² / Pr[ŷ = 1]`, whose two quantities both come from labelled
positives and the predictions alone. PU-F1 increases monotonically with the true F1 under
the SCAR assumption, so it ranks models correctly even though its absolute value is not a
standard F1 and should not be read as one.

## Notes on the encoders

| Encoder | Input | Output | Where it runs |
|---|---|---|---|
| ESM-C 6B `esmc-6b-2024-12` | sequence | layer 80, mean-pooled, 2560-d | remote (`$ESM_API_KEY`) |
| BioMistral-7B | `desc_nodup`, 512 tokens | last layer, mean-pooled, 4096-d | local GPU |
| ESM-2 3B *(ablation)* | sequence | mean-pooled, 2560-d | local GPU |
| BioBERT *(ablation)* | `desc_nodup`, 512 tokens | CLS token, 768-d | local GPU |

No local weights are published for ESM-C 6B, so it is served through the ESM SDK and needs
an API token:

```bash
export ESM_API_KEY=your_token_here      # or $ESM_FORGE_TOKEN, or --token on the command line
```

Without it, step 3 cannot embed new sequences. The published caches need no token and
cover every protein in `data/`, which is all you need to reproduce the paper.

Semantic deduplication uses `all-MiniLM-L6-v2` at cosine ≥ 0.80. The same domain is often
described once per member database, and leaving those near-duplicates in would let a
frequently-annotated domain dominate the text embedding by repetition alone.


## References

1. ESM-C — protein language model · [EvolutionaryScale](https://github.com/evolutionaryscale/esm)
2. BioMistral-7B — biomedical language model · [HuggingFace](https://huggingface.co/BioMistral/BioMistral-7B)
3. InterProScan — protein signature annotation · [docs](https://interproscan-docs.readthedocs.io/)
4. Protegen — protective antigen database · [VIOLIN](https://violinet.org/protegen/)
5. Mordelet F, Vert JP. *A bagging SVM to learn from positive and unlabeled examples.* Pattern Recognit Lett. 2014.
6. Lee WS, Liu B. *Learning with positive and unlabeled examples using weighted logistic regression.* ICML 2003.

## License

MIT — see [LICENSE](LICENSE).
