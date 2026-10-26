from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import polars as pl
from sklearn.metrics import roc_auc_score

from techarena.config import EXPERIMENTS_LOG, TimeSplit

SUBMISSION_COLUMNS = ["rowid", "predicted_probability"]


DAY = timedelta(days=1)


@dataclass(frozen=True)
class Window:
    stats: pl.LazyFrame  # labelled events before the window, for click-rate stats and ALS
    rows: pl.LazyFrame  # rows to build features for


@dataclass(frozen=True)
class Windows:
    ranker: list[Window]
    target: Window


def first_event_per_user(rows: pl.LazyFrame) -> pl.LazyFrame:
    """Keep each user's earliest event, like evaluation.csv does."""
    first_events = (
        rows.select("user_id", "event_id", "timestamp")
        .unique()
        .sort("timestamp", "event_id")
        .group_by("user_id")
        .agg(pl.col("event_id").first())
    )
    return rows.join(first_events, on=["user_id", "event_id"], how="semi")


def make_windows(
    train: pl.LazyFrame, split: TimeSplit, evaluation: pl.LazyFrame | None = None
) -> Windows:
    """Cut train by day; the target is evaluation if given, else the first event per user."""

    def seconds(day: datetime) -> int:
        return int(day.timestamp())

    def before(day: datetime) -> pl.LazyFrame:
        return train.filter(pl.col("timestamp") < seconds(day))

    ranker = [
        Window(
            stats=before(day),
            rows=train.filter(
                pl.col("timestamp").is_between(seconds(day), seconds(day + DAY), "left")
            ),
        )
        for day in split.ranker_days
    ]
    target_rows = (
        evaluation
        if evaluation is not None
        else first_event_per_user(train.filter(pl.col("timestamp") >= seconds(split.target_day)))
    )
    return Windows(ranker=ranker, target=Window(stats=before(split.target_day), rows=target_rows))


def global_auc(clicked: pl.Series, predicted: pl.Series) -> float:
    return float(roc_auc_score(clicked.to_numpy(), predicted.to_numpy()))


def check_submission(path: Path, evaluation_rowids: pl.Series) -> None:
    """Raise if the file breaks any rule from the competition instructions."""
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if header != SUBMISSION_COLUMNS:
        raise ValueError(f"{path.name}: wrong header {header}")

    sub = pl.read_csv(path, schema={"rowid": pl.String, "predicted_probability": pl.Float64})
    expected = evaluation_rowids.to_frame("rowid")
    pred = pl.col("predicted_probability")
    problems = {
        "wrong row count": sub.height != expected.height,
        "duplicate rowids": sub["rowid"].is_duplicated().any(),
        "missing rowids": expected.join(sub, on="rowid", how="anti").height > 0,
        "unknown rowids": sub.join(expected, on="rowid", how="anti").height > 0,
        "invalid values": sub.select(
            (pred.is_null() | ~pred.is_finite() | (pred < 0) | (pred > 1)).any()
        ).item(),
    }
    failed = [name for name, bad in problems.items() if bad]
    if failed:
        raise ValueError(f"{path.name}: {', '.join(failed)}")


def log_experiment(model: str, split: str, auc: float, **params) -> None:
    """Append one run to reports/experiments.jsonl."""
    record = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model,
        "split": split,
        "auc": round(auc, 6),
        **params,
    }
    with EXPERIMENTS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
