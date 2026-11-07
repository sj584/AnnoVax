# Worked example

Six proteins from the PLGDL test split — two bacterial, two viral, two eukaryotic, one of
each class — with every intermediate file the pipeline produces.

| File | Produced by | What it is |
|---|---|---|
| `example.fasta` | — | the six sequences |
| `example_interproscan.tsv` | InterProScan 6 | 78 signature matches across the six |
| `example_preprocessed.csv` | `scripts/preprocess.py` | `ID,Sequence,desc,desc_nodup` |

Reproduce the preprocessing step:

```bash
python scripts/preprocess.py -i example/example.fasta \
    -ip example/example_interproscan.tsv -o example/example_preprocessed.csv
```

Deduplication is doing real work here — the six proteins carry 62 descriptions between
them and keep 44, because the same domain is described once per member database.

Then embed and score (ESM-C 6B needs `$ESM_API_KEY`; see the main README):

```bash
python scripts/extract_embedding.py -i example/example_preprocessed.csv \
    -o example/embeddings --corpus example
python scripts/predict.py -i example/example_preprocessed.csv \
    -e example/embeddings -c example -m models/annovax_K1000
```

These six proteins are in the PLGDL test split, so the published
`embeddings/plgdl__*.npz` caches already cover them — `-e embeddings -c plgdl` works
without an API token and gives the same scores as the paper reports for them.
