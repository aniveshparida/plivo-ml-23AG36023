# Starter kit

- `baseline.py` — the silence-only baseline; also shows the exact predict.py
  interface you must ship.
- `features.py` — audio loading, framing, energy, autocorrelation pitch
  tracker. Utilities only; the features are your job.
- `train.py` — trains the final RandomForest on both languages at once and
  saves `rf_model.pkl`. Also runs a grouped (by `turn_id`) 5-fold
  cross-validation before that final fit and prints an honest, held-out
  estimate of score.py's own metric per language, so you're not just
  reading an in-sample number. Actual interface (differs from the original
  skeleton's `--data_dir`/`--out`, since this version needs both language
  folders and always writes to `rf_model.pkl`):
  `python train.py --data_root ../eot_data [--n_splits 5]`
- `score.py` — the official scorer. Dev loop actually used for this submission:

```
python baseline.py --data_dir ../eot_data/english --out base.csv
python score.py    --data_dir ../eot_data/english --pred base.csv

python train.py     --data_root ../eot_data
python predict.py   --data_dir ../eot_data/english --out english_preds.csv
python predict.py   --data_dir ../eot_data/hindi   --out hindi_preds.csv
python score.py     --data_dir ../eot_data/english --pred english_preds.csv
python score.py     --data_dir ../eot_data/hindi   --pred hindi_preds.csv
```

Log every score in RUNLOG.md. Listen to your errors — that is where the
points are.

See `requirements.txt` for the exact pinned dependency versions this was
trained and tested against.
