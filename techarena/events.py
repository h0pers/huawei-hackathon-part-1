"""Compact integer-array cache of train and evaluation events, for fast experiments.

Identifiers stay strings in the parquet files; the arrays only hold positions in sorted lookup
tables built here, and evaluation rowids are saved as strings next to them.
"""

from dataclasses import dataclass

from loguru import logger
import numpy as np
import polars as pl

from techarena.config import EVAL_PARQUET, PROCESSED_DATA_DIR, TRAIN_PARQUET

CACHE = PROCESSED_DATA_DIR / "events.npz"
HISTORY = 20
CANDIDATES = 10
NO_CLICK = CANDIDATES


@dataclass(frozen=True)
class Events:
    """One record per event. Item and user ids are 1-based; 0 means padding."""

    user: np.ndarray  # (n,)
    ts: np.ndarray  # (n,) Unix seconds
    time_bucket: np.ndarray  # (n,)
    day_type: np.ndarray  # (n,)
    hist: np.ndarray  # (n, 20) right-aligned, so the last column is the newest click
    hist_ts: np.ndarray  # (n, 20) aligned with hist, 0 where padded
    cand: np.ndarray  # (n, 10)
    label: np.ndarray  # (n,) clicked position, or NO_CLICK; -1 on evaluation events

    def take(self, mask: np.ndarray) -> "Events":
        return Events(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    def __len__(self) -> int:
        return len(self.user)


@dataclass(frozen=True)
class Cache:
    train: Events
    evaluation: Events
    n_users: int  # ids run 1..n_users
    n_items: int
    eval_rowids: np.ndarray  # (n_eval * 10,) strings, in the order of evaluation.cand.ravel()


def _reshape(path, users: pl.DataFrame, items: pl.DataFrame) -> tuple[Events, np.ndarray]:
    rows = pl.read_parquet(path)
    labelled = "clicked" in rows.columns
    # Row order within an event carries nothing; rowid is random, so sorting by it is neutral.
    rows = rows.sort("event_id", "rowid")
    rows = rows.join(items, on="item_id", how="left", maintain_order="left")
    rows = rows.join(users, on="user_id", how="left", maintain_order="left")
    n = rows.height // CANDIDATES
    if rows.height % CANDIDATES:
        raise ValueError(f"{path}: rows are not a multiple of {CANDIDATES}")
    firsts = rows.gather_every(CANDIDATES)

    hist_rows = (
        firsts.select(
            pl.int_range(pl.len()).alias("e"),
            pl.col("history_item_ids").alias("item_id"),
            pl.col("history_timestamps").alias("t"),
        )
        .explode("item_id", "t")
        .drop_nulls("item_id")
        .with_columns(pos=pl.int_range(pl.len()).over("e"), length=pl.len().over("e"))
        .join(items, on="item_id", how="left", maintain_order="left")
    )
    hist = np.zeros((n, HISTORY), dtype=np.int32)
    hist_ts = np.zeros((n, HISTORY), dtype=np.int64)
    column = (HISTORY - hist_rows["length"] + hist_rows["pos"]).to_numpy()
    event = hist_rows["e"].to_numpy()
    hist[event, column] = hist_rows["item"].to_numpy()
    hist_ts[event, column] = hist_rows["t"].to_numpy()

    label = np.full(n, -1, dtype=np.int64)
    if labelled:
        clicked = rows["clicked"].to_numpy().reshape(n, CANDIDATES)
        if clicked.sum(1).max() > 1:
            raise ValueError(f"{path}: an event has more than one click")
        label = np.where(clicked.max(1) == 1, clicked.argmax(1), NO_CLICK)

    events = Events(
        user=firsts["user"].to_numpy().astype(np.int32),
        ts=firsts["timestamp"].to_numpy().astype(np.int64),
        time_bucket=firsts["time_bucket"].to_numpy().astype(np.int8),
        day_type=firsts["day_type"].to_numpy().astype(np.int8),
        hist=hist,
        hist_ts=hist_ts,
        cand=rows["item"].to_numpy().astype(np.int32).reshape(n, CANDIDATES),
        label=label,
    )
    return events, rows["rowid"].to_numpy().astype(str)


def build_cache() -> None:
    """Index every user and item that appears anywhere, including only inside histories."""
    both = pl.concat(
        [
            pl.scan_parquet(TRAIN_PARQUET).select("user_id", "item_id", "history_item_ids"),
            pl.scan_parquet(EVAL_PARQUET).select("user_id", "item_id", "history_item_ids"),
        ]
    )
    user_ids = both.select("user_id").unique().sort("user_id").collect()
    history_items = both.select(pl.col("history_item_ids").alias("item_id")).explode("item_id")
    item_ids = (
        pl.concat([both.select("item_id"), history_items])
        .drop_nulls()
        .unique()
        .sort("item_id")
        .collect()
    )
    users = user_ids.with_row_index("user", offset=1).cast({"user": pl.Int32})
    items = item_ids.with_row_index("item", offset=1).cast({"item": pl.Int32})

    train, _ = _reshape(TRAIN_PARQUET, users, items)
    evaluation, eval_rowids = _reshape(EVAL_PARQUET, users, items)
    arrays = {f"train_{f}": getattr(train, f) for f in Events.__dataclass_fields__}
    arrays |= {f"eval_{f}": getattr(evaluation, f) for f in Events.__dataclass_fields__}
    np.savez(CACHE, **arrays, n_users=users.height, n_items=items.height, eval_rowids=eval_rowids)
    logger.success(f"Saved {CACHE}: {len(train):,} train, {len(evaluation):,} evaluation events")


def load_cache() -> Cache:
    if not CACHE.exists():
        build_cache()
    with np.load(CACHE) as z:

        def events(prefix: str) -> Events:
            return Events(*(z[f"{prefix}_{f}"] for f in Events.__dataclass_fields__))

        return Cache(
            train=events("train"),
            evaluation=events("eval"),
            n_users=int(z["n_users"]),
            n_items=int(z["n_items"]),
            eval_rowids=z["eval_rowids"],
        )


if __name__ == "__main__":
    build_cache()
