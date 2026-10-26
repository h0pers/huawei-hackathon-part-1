# AGENTS.md

Huawei TechArena 2026, Phase 1, team project. Predict the click probability for every row of
`evaluation.csv`. Scored by global ROC AUC over all rows; only ranking matters, not calibration.
Final ranking uses the private leaderboard.

## Task in one paragraph
An event is one user shown 10 candidate items; each row is one candidate. In train, `clicked` is
1 for at most one row per event (37% of events have no click). Features are user, item, time and
the user's last ≤20 clicks before the event (`history_item_ids`, `history_timestamps`). Train is
2025-01-20 to 2025-01-26; evaluation is each user's first event from 2025-01-27 on.

## Current state
- Repo is the Cookiecutter Data Science template; `techarena/` modules are still template stubs.
- Raw CSVs are in `data/raw/` (gitignored, `train.csv` is 7.3 GB).
- A Parquet pipeline (Kaggle download, CSV → `data/interim/*.parquet`, 50k-event sample) is in
  progress in `git stash@{0}`, not merged. Don't assume its code or deps exist on the branch.

## Hard rules
- `rowid`, `event_id`, `user_id`, `item_id` are opaque strings. Never cast them to numbers.
- Never load all of `train.csv` with pandas; stream it, or use Parquet or the sample.
- Parse history columns as JSON after reading the CSV with a CSV parser.
- Split validation by time and keep all 10 rows of an event on one side.
- Compute target statistics (user/item CTR etc.) only from the fit window, never from the rows
  being scored.
- Predictions must cover eval users unseen in train (2.7%) and empty histories.
- Submission: header `rowid,predicted_probability`, one row per eval `rowid`, values in [0, 1].
- The team has only 10 uploads per day. Don't suggest uploading to "try something".

## Commands
- `uv sync` installs deps. Ask before adding a dependency.
- `make lint` / `make format` run ruff (line length 99).

## Read when relevant
- `references/brief.md`: the organiser's brief, verbatim. Check exact rules here.
- `docs/data.md`: data access, sizes, measured stats, gotchas.
- `docs/validation.md`: local validation and the pre-upload checklist.
- `docs/approach.md`: baseline and current modelling direction.
- `docs/experiments.md`: what we already tried and what it scored.

## Working with the team
- Commit messages: short and plain. Commit only when asked.
- After a scored experiment, add a row to `docs/experiments.md`.