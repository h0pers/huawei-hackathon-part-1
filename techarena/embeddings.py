from dataclasses import dataclass

from implicit.als import AlternatingLeastSquares
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

# Untuned starting guesses; tune against validation AUC.
ALS_PARAMS = {
    "factors": 64,
    "regularization": 0.05,
    "alpha": 10.0,
    "iterations": 15,
    "random_state": 42,
}


@dataclass(frozen=True)
class Als:
    users: pl.DataFrame  # user_id -> user_idx
    items: pl.DataFrame  # item_id -> item_idx
    user_factors: np.ndarray
    item_factors: np.ndarray
    item_unit: np.ndarray  # item_factors scaled to length 1, for cosine similarity


def fit_als(clicks: pl.LazyFrame) -> Als:
    """Learn user and item taste vectors from who clicked what."""
    counts = clicks.group_by("user_id", "item_id").agg(pl.len().alias("n")).collect()
    users = counts.select("user_id").unique().with_row_index("user_idx")
    items = counts.select("item_id").unique().with_row_index("item_idx")
    counts = counts.join(users, on="user_id").join(items, on="item_id")
    matrix = csr_matrix(
        (
            counts["n"].to_numpy().astype(np.float32),
            (counts["user_idx"].to_numpy(), counts["item_idx"].to_numpy()),
        ),
        shape=(users.height, items.height),
    )

    model = AlternatingLeastSquares(**ALS_PARAMS)
    # implicit parallelises itself; extra BLAS threads slow it down (per implicit's docs).
    with threadpool_limits(1, "blas"):
        model.fit(matrix, show_progress=False)

    item_factors = np.asarray(model.item_factors)
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    return Als(
        users=users,
        items=items,
        user_factors=np.asarray(model.user_factors),
        item_factors=item_factors,
        item_unit=item_factors / np.maximum(norms, 1e-9),
    )


def row_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def als_features(als: Als, target: pl.DataFrame) -> pl.DataFrame:
    """Taste-match scores per row: user vector, mean history vector and last clicked item."""
    events = (
        target.select("event_id", "history_item_ids")
        .unique("event_id")
        .with_row_index("event_idx")
    )
    rows = (
        target.select("rowid", "event_id", "user_id", "item_id")
        .join(events.select("event_id", "event_idx"), on="event_id")
        .join(als.users, on="user_id", how="left")
        .join(als.items, on="item_id", how="left")
    )
    history = (
        events.select("event_idx", pl.col("history_item_ids").alias("item_id"))
        .explode("item_id")
        .with_columns(pos=pl.int_range(pl.len()).over("event_idx"))
        .join(als.items, on="item_id")
    )

    # Mean unit vector of each event's history items (NaN when none are known).
    weights = csr_matrix(
        (
            np.ones(history.height, dtype=np.float32),
            (history["event_idx"].to_numpy(), history["item_idx"].to_numpy()),
        ),
        shape=(events.height, als.items.height),
    )
    with np.errstate(invalid="ignore"):
        hist_vec = (weights @ als.item_unit) / np.asarray(weights.sum(axis=1))
    last_item = np.full(events.height, -1)
    last = history.sort("pos").group_by("event_idx").agg(pl.col("item_idx").last())
    last_item[last["event_idx"].to_numpy()] = last["item_idx"].to_numpy()

    item = rows["item_idx"].fill_null(-1).to_numpy()
    user = rows["user_idx"].fill_null(-1).to_numpy()
    event = rows["event_idx"].to_numpy()
    known_item = item >= 0
    user_score = np.full(rows.height, np.nan)
    hist_sim = np.full(rows.height, np.nan)
    last_sim = np.full(rows.height, np.nan)

    both = known_item & (user >= 0)
    user_score[both] = row_dot(als.user_factors[user[both]], als.item_factors[item[both]])
    hist_sim[known_item] = row_dot(hist_vec[event[known_item]], als.item_unit[item[known_item]])
    has_last = known_item & (last_item[event] >= 0)
    last_sim[has_last] = row_dot(
        als.item_unit[last_item[event[has_last]]], als.item_unit[item[has_last]]
    )

    return pl.DataFrame(
        {
            "rowid": rows["rowid"],
            "als_user_score": user_score,
            "als_hist_sim": hist_sim,
            "als_last_sim": last_sim,
        }
    ).fill_nan(None)
