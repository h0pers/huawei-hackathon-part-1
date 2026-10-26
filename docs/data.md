# Data

Column meanings and rules are in `references/brief.md`. This file holds what we measured and how
to get the data.

## Getting it
- Kaggle dataset `dmytrohryshchenko/huawei-ireland-tech-arena-2026` (private, ask for access).
- Downloading the whole dataset returns 404. Download each file on its own:
  `train.csv`, `evaluation.csv`, `example_submission.csv`.
- Put them in `data/raw/`. `data/` is gitignored.

## Files
| File | Rows | Events | Size |
|---|---|---|---|
| `train.csv` | 6,868,240 | 686,824 | 7.26 GB |
| `evaluation.csv` | 529,390 | 52,939 (one per user) | |
| `example_submission.csv` | 529,390 | | constant 0.0629 |

## Measured facts
- Row CTR 6.29%, stable from day to day. 62.9% of train events have a click.
- Train: 53,003 users, 50,000 items, events 2025-01-20 00:00 to 2025-01-26 23:59 UTC.
- Eval: events 2025-01-27 to 2025-02-02 UTC (each user's first event after the cutoff).
- 1,453 eval users (2.7%) never appear in train. Every eval item appears in train.
- Empty history: 0.33% of train rows, 1.26% of eval rows.
- Histories hold clicks only, max 20, oldest first, and can include warm-up clicks from before
  2025-01-20. A user can have history without any train rows.

## Gotchas
- History fields are JSON arrays inside quoted CSV fields. Never split lines on commas.
- `train.csv` doesn't fit comfortably in pandas. Stream it, or use Parquet (below).
- `timestamp` is Unix seconds UTC in whole minutes. `time_bucket` and `day_type` come from it.

## Parquet pipeline (in progress)
Lives in `git stash@{0}` and isn't merged yet. When merged, `techarena/dataset.py` will have
`download`, `to_parquet` and `sample` commands writing to `data/interim/`:
- `train.parquet` and `eval.parquet`: typed, histories as lists, plus `event_time` and
  `history_len`. IDs stay strings. Integrity checks against the brief run on write.
- `train_sample.parquet`: 50,000 random whole events, for fast prototyping.

It adds `polars`, `pyarrow` and `kagglehub`. These files may already exist locally, but the
code that builds them is not on the branch yet.