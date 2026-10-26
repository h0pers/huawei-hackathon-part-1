from datetime import UTC, datetime

import polars as pl
import pytest

from techarena.config import VALID_SPLIT
from techarena.validation import check_submission, make_windows


def at(day: int, hour: int = 0) -> int:
    return int(datetime(2025, 1, day, hour, tzinfo=UTC).timestamp())


@pytest.fixture
def train() -> pl.LazyFrame:
    events = [
        ("e1", "u1", at(24)),
        ("e2", "u1", at(25)),
        ("e3", "u1", at(26, 5)),
        ("e4", "u1", at(26, 1)),
        ("e5", "u2", at(26, 9)),
    ]
    rows = [
        {"rowid": f"{e}_{i}", "event_id": e, "user_id": u, "item_id": f"i{i}", "timestamp": ts}
        for e, u, ts in events
        for i in range(2)
    ]
    return pl.LazyFrame(rows)


def event_ids(frame: pl.LazyFrame) -> set[str]:
    return set(frame.select("event_id").collect()["event_id"])


def test_windows_split_by_time(train):
    windows = make_windows(train, VALID_SPLIT)
    assert event_ids(windows.stats) == {"e1"}
    assert event_ids(windows.ranker) == {"e2"}
    # Only each user's first event on Jan 26, with all of its candidate rows.
    assert event_ids(windows.target) == {"e4", "e5"}
    assert windows.target.collect().height == 4


def test_windows_use_evaluation_as_target(train):
    evaluation = pl.LazyFrame({"event_id": ["x"], "rowid": ["r"]})
    assert event_ids(make_windows(train, VALID_SPLIT, evaluation).target) == {"x"}


@pytest.fixture
def rowids() -> pl.Series:
    return pl.Series(["row_a", "row_b"])


def write(tmp_path, text: str):
    path = tmp_path / "submission.csv"
    path.write_text(text)
    return path


def test_check_submission_accepts_valid_file(tmp_path, rowids):
    check_submission(write(tmp_path, "rowid,predicted_probability\nrow_b,1\nrow_a,2e-3\n"), rowids)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("id,predicted_probability\nrow_a,0.1\nrow_b,0.2\n", "wrong header"),
        ("rowid,predicted_probability\nrow_a,0.1\n", "missing rowids"),
        ("rowid,predicted_probability\nrow_a,0.1\nrow_a,0.2\n", "duplicate rowids"),
        ("rowid,predicted_probability\nrow_a,0.1\nrow_c,0.2\n", "unknown rowids"),
        ("rowid,predicted_probability\nrow_a,1.5\nrow_b,0.2\n", "invalid values"),
        ("rowid,predicted_probability\nrow_a,\nrow_b,0.2\n", "invalid values"),
    ],
)
def test_check_submission_rejects_bad_file(tmp_path, rowids, text, problem):
    with pytest.raises(ValueError, match=problem):
        check_submission(write(tmp_path, text), rowids)
