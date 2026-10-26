from dataclasses import dataclass

import polars as pl

from techarena.choice import Choice, choice_features, fit_choice
from techarena.embeddings import Als, als_features, fit_als
from techarena.validation import Window

SMOOTHING_K = 50
DAY_SECONDS = 86_400

FEATURES = [
    # Repeat clicks: has the user clicked this candidate before?
    "cand_hist_count",
    "cand_since_last_click",
    "cand_last_pos",
    # User activity, available even for users unseen in the stats window.
    "hist_len",
    "hist_unique_ratio",
    "since_last_click",
    "clicks_per_day",
    # Click-rate stats from the stats window.
    "item_impressions",
    "item_ctr",
    "item_clicks",
    "item_clicks_last_day",
    "user_impressions",
    "user_ctr",
    "user_event_click_rate",
    # Similarity between the candidate and the user's history.
    "covis_sum",
    "covis_max",
    "covis_recent",
    # ALS taste vectors: does the candidate match the user's taste?
    "als_user_score",
    "als_hist_sim",
    "als_last_sim",
    # Softmax choice model trained on every impression: candidates compete with each other and
    # with clicking nothing.
    "choice_prob",
    "choice_score",
    "choice_no_click",
    # Candidate compared with the other candidates of the same event.
    "event_cands_in_hist",
    "item_ctr_rank",
    "item_ctr_vs_event",
    "covis_rank",
    "als_hist_rank",
    "choice_rank",
    "time_bucket",
    "day_type",
]


@dataclass(frozen=True)
class StatsTables:
    items: pl.LazyFrame
    users: pl.LazyFrame
    covis: pl.LazyFrame
    als: Als
    choice: Choice


def click_log(stats: pl.LazyFrame) -> pl.LazyFrame:
    """Every known click (user, item, time) before the end of the stats window."""
    history = stats.select(
        "user_id",
        pl.col("history_item_ids").alias("item_id"),
        pl.col("history_timestamps").alias("timestamp"),
    ).explode("item_id", "timestamp")
    labelled = stats.filter(pl.col("clicked") == 1).select("user_id", "item_id", "timestamp")
    return pl.concat([history, labelled]).drop_nulls().unique()


def build_stats_tables(stats: pl.LazyFrame) -> StatsTables:
    """Aggregate the stats window once; reused for every target frame."""
    global_ctr = stats.select(pl.col("clicked").mean()).collect().item()
    smoothed_ctr = (pl.col("clicked").sum() + SMOOTHING_K * global_ctr) / (pl.len() + SMOOTHING_K)
    clicks = click_log(stats)
    last_day = stats.select(pl.col("timestamp").max()).collect().item() - DAY_SECONDS

    item_clicks = clicks.group_by("item_id").agg(
        pl.len().alias("item_clicks"),
        (pl.col("timestamp") >= last_day).sum().alias("item_clicks_last_day"),
    )
    items = (
        stats.group_by("item_id")
        .agg(pl.len().alias("item_impressions"), smoothed_ctr.alias("item_ctr"))
        .join(item_clicks, on="item_id", how="full", coalesce=True)
    )
    user_events = (
        stats.group_by("user_id", "event_id")
        .agg(pl.col("clicked").max())
        .group_by("user_id")
        .agg(pl.col("clicked").mean().alias("user_event_click_rate"))
    )
    users = (
        stats.group_by("user_id")
        .agg(pl.len().alias("user_impressions"), smoothed_ctr.alias("user_ctr"))
        .join(user_events, on="user_id")
    )

    # Cosine co-click similarity: users who clicked A also clicked B.
    interacted = clicks.select("user_id", "item_id").unique()
    item_users = interacted.group_by("item_id").agg(pl.len().sqrt().alias("norm"))
    covis = (
        interacted.join(interacted, on="user_id", suffix="_hist")
        .filter(pl.col("item_id") != pl.col("item_id_hist"))
        .group_by("item_id", "item_id_hist")
        .agg(pl.len().alias("co"))
        .join(item_users, on="item_id")
        .join(
            item_users.rename({"item_id": "item_id_hist", "norm": "norm_hist"}), on="item_id_hist"
        )
        .select(
            "item_id",
            "item_id_hist",
            (pl.col("co") / (pl.col("norm") * pl.col("norm_hist"))).alias("covis"),
        )
    )
    als = fit_als(clicks)
    return StatsTables(
        items=items,
        users=users,
        covis=covis.collect().lazy(),
        als=als,
        choice=fit_choice(stats, als),
    )


def history_features(target: pl.LazyFrame, covis: pl.LazyFrame) -> pl.LazyFrame:
    """Compare each candidate with every item in its row's click history."""
    match = pl.col("item_id_hist") == pl.col("item_id")
    return (
        target.select(
            "rowid",
            "item_id",
            "timestamp",
            pl.col("history_item_ids").alias("item_id_hist"),
            pl.col("history_timestamps").alias("timestamp_hist"),
        )
        .explode("item_id_hist", "timestamp_hist")
        .with_columns(pos_from_end=pl.int_range(pl.len()).reverse().over("rowid"))
        .join(covis, on=["item_id", "item_id_hist"], how="left")
        .group_by("rowid")
        .agg(
            match.sum().alias("cand_hist_count"),
            (pl.col("timestamp").first() - pl.col("timestamp_hist").filter(match).max()).alias(
                "cand_since_last_click"
            ),
            pl.col("pos_from_end").filter(match).min().alias("cand_last_pos"),
            pl.col("covis").sum().alias("covis_sum"),
            pl.col("covis").max().alias("covis_max"),
            (pl.col("covis") / (1 + pl.col("pos_from_end"))).sum().alias("covis_recent"),
        )
    )


def build_features(tables: StatsTables, target: pl.LazyFrame) -> pl.LazyFrame:
    """One feature row per target row; keeps rowid, event_id and clicked when present."""
    history = pl.col("history_item_ids")
    first_ts = pl.col("history_timestamps").list.first()
    last_ts = pl.col("history_timestamps").list.last()
    hist_len = history.list.len()
    span_days = (last_ts - first_ts) / DAY_SECONDS

    keep = [c for c in ("rowid", "event_id", "clicked") if c in target.collect_schema()]
    target_rows = target.select(
        "rowid", "event_id", "user_id", "item_id", "history_item_ids"
    ).collect()
    als = als_features(tables.als, target_rows)
    choice = choice_features(tables.choice, target_rows)
    return (
        target.with_columns(
            hist_len=hist_len,
            hist_unique_ratio=pl.when(hist_len > 0).then(history.list.n_unique() / hist_len),
            since_last_click=pl.col("timestamp") - last_ts,
            clicks_per_day=hist_len / (span_days + 1),
        )
        .join(history_features(target, tables.covis), on="rowid", how="left")
        .join(tables.items, on="item_id", how="left")
        .join(tables.users, on="user_id", how="left")
        .join(als.lazy(), on="rowid", how="left")
        .join(choice.lazy(), on="rowid", how="left")
        .with_columns(
            pl.col("cand_hist_count", "covis_sum", "covis_recent", "item_clicks").fill_null(0)
        )
        .with_columns(
            event_cands_in_hist=(pl.col("cand_hist_count") > 0).sum().over("event_id"),
            item_ctr_rank=pl.col("item_ctr").rank(descending=True).over("event_id"),
            item_ctr_vs_event=pl.col("item_ctr") - pl.col("item_ctr").mean().over("event_id"),
            covis_rank=pl.col("covis_sum").rank(descending=True).over("event_id"),
            als_hist_rank=pl.col("als_hist_sim").rank(descending=True).over("event_id"),
            choice_rank=pl.col("choice_prob").rank(descending=True).over("event_id"),
        )
        .select(*keep, *FEATURES)
    )


def window_features(window: Window) -> pl.DataFrame:
    """Features for a window's rows, using stats from before the window."""
    return build_features(build_stats_tables(window.stats), window.rows).collect()
