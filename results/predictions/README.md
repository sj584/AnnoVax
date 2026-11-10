# AnnoVax predictions on the two benchmark sets

Per-protein output of the released AnnoVax model on the PLGDL test set and the Vaxign
independent set. These are the scores behind the numbers reported in the paper.

| File | Evaluation set | Rows (pos/neg) | ROC-AUC | MCC |
|---|---|---|---|---|
| `PLGDL_test_predictions.csv` | PLGDL test, original (`data/PLGDL/All/test_All.csv`) | 1320 (120/1200) | 0.918 | 0.676 |
| `Vaxign_independent_predictions.csv` | Vaxign independent (`data/Vaxign_benchmark/Vaxign_benchmark.csv`) | 246 (131/115) | 0.889 | 0.681 |

## Columns

- `ID`, `Label` — carried over from the input CSV
- `score_mean` — the model's score: the mean over all 1000 bagged boosters
- `score_std`, `score_q05`, `score_q50`, `score_q95` — spread of that score across the bags
- `pred` — binary call at the MCC-optimal threshold used in the paper:
  **0.6440** for the PLGDL test set, **0.5690** for Vaxign. The two sets have very
  different positive rates (9% vs 53%), so each has its own threshold; re-thresholding
  `score_mean` yourself is fine for any other operating point.

## How these were produced

A single run of the training command in the top-level [README](../../README.md) —

```bash
python scripts/train.py --arm esmc_biomistral --k 1000 --ratio 2.0 --p-ratio 0.8 \
    --seed 0 --test-variant original --vaxign-variant original
```

— i.e. ESM C + BioMistral embeddings, PU bagging over 1000 XGBoost boosters, trained on
the **surface-matched** PLGDL training set and scored on the **original** (not
surface-matched) test and Vaxign sets, which is how both are reported in the paper.
Running `scripts/predict.py` with the published weights reproduces `score_mean` here.
