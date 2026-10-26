# Validation

We have no eval labels and only 10 uploads a day, so local validation decides what gets submitted.

## Default split
- Fit on 2025-01-20 to 2025-01-25. Validate on 2025-01-26 (the last train day).
- Split by event time. All 10 rows of an event stay on one side.
- Any label-based statistic (user CTR, item CTR, co-clicks) uses fit-window labels only.
  Recompute it on the full train set for the final submission.
- History columns are safe to use on both sides: they only hold clicks before the event.

## How this differs from the official eval
- Eval is one event per user (their first after the cutoff). Validation has many events per
  user. To mimic eval, also score only each user's first event in the validation day.
- Eval spans 7 days after the cutoff, so some users are seen days after their last train event.
- 2.7% of eval users have no train rows. Check the score on validation users absent from the fit window.

## Metric
- ROC AUC over all rows pooled, not averaged per event or per user (`sklearn.metrics.roc_auc_score`).
- Public and private leaderboards are random row splits of eval. The final ranking uses the
  private one, so don't overfit public LB noise.

## Before every upload
- Header is exactly `rowid,predicted_probability`.
- 529,390 rows, every eval `rowid` exactly once, no extras, IDs unchanged.
- All values finite and in [0, 1]. No NaN.
- Check the submission history before re-uploading after an interrupted upload.
- Log the result in `docs/experiments.md`.