from pathlib import Path

from loguru import logger
import polars as pl
import typer

from techarena.config import (
    EVAL_PARQUET,
    SUBMISSION_PATH,
    SUBMIT_SPLIT,
    TRAIN_PARQUET,
    VALID_SPLIT,
)
from techarena.validation import check_submission, global_auc, log_experiment, make_windows

app = typer.Typer()

# Same constants as starter_solution.py from the brief.
SMOOTHING_K = 50
CF_WEIGHT = 25.0


def logit(p: pl.Expr) -> pl.Expr:
    p = p.clip(1e-6, 1 - 1e-6)
    return (p / (1 - p)).log()


def predict(labelled: pl.LazyFrame, target: pl.LazyFrame) -> pl.LazyFrame:
    """Port of starter_solution.py: item CTR + user CTR in log-odds, plus co-click bonus."""
    global_ctr = labelled.select(pl.col("clicked").mean()).collect().item()

    def smoothed_ctr(key: str) -> pl.LazyFrame:
        ctr = (pl.col("clicked").sum() + SMOOTHING_K * global_ctr) / (pl.len() + SMOOTHING_K)
        return labelled.group_by(key).agg(ctr.alias(f"{key}_ctr"))

    # Items each user clicked, from histories and labels.
    interacted = (
        pl.concat(
            [
                labelled.select("user_id", pl.col("history_item_ids").alias("item_id")).explode(
                    "item_id"
                ),
                labelled.filter(pl.col("clicked") == 1).select("user_id", "item_id"),
            ]
        )
        .drop_nulls()
        .unique()
    )
    popularity = interacted.group_by("item_id").agg(pl.len().sqrt().alias("pop"))
    co_clicks = (
        interacted.join(interacted, on="user_id", suffix="_hist")
        .filter(pl.col("item_id") != pl.col("item_id_hist"))
        .group_by("item_id", "item_id_hist")
        .agg(pl.len().alias("co"))
    )
    cf = (
        target.select("rowid", "item_id", pl.col("history_item_ids").alias("item_id_hist"))
        .explode("item_id_hist")
        .join(co_clicks, on=["item_id", "item_id_hist"])
        .join(popularity, on="item_id")
        .join(popularity.rename({"item_id": "item_id_hist", "pop": "pop_hist"}), on="item_id_hist")
        .group_by("rowid")
        .agg((pl.col("co") / (pl.col("pop") * pl.col("pop_hist"))).sum().alias("cf"))
    )

    score = (
        logit(pl.col("item_id_ctr").fill_null(global_ctr))
        + logit(pl.col("user_id_ctr").fill_null(global_ctr))
        - logit(pl.lit(global_ctr))
        + CF_WEIGHT * pl.col("cf").fill_null(0.0)
    )
    return (
        target.join(smoothed_ctr("item_id"), on="item_id", how="left")
        .join(smoothed_ctr("user_id"), on="user_id", how="left")
        .join(cf, on="rowid", how="left")
        .select("rowid", (1 / (1 + (-score).exp())).alias("predicted_probability"))
    )


@app.command()
def main(submit: bool = False, output_path: Path = SUBMISSION_PATH):
    """Score the validation day, or with --submit write a submission for evaluation.csv."""
    train = pl.scan_parquet(TRAIN_PARQUET)
    if not submit:
        target = make_windows(train, VALID_SPLIT).target
        scored = (
            predict(target.stats, target.rows)
            .join(target.rows.select("rowid", "clicked"), on="rowid")
            .collect()
        )
        auc = global_auc(scored["clicked"], scored["predicted_probability"])
        log_experiment("baseline", "valid", auc)
        logger.success(f"Validation AUC: {auc:.5f} on {scored.height:,} rows")
        return

    evaluation = pl.scan_parquet(EVAL_PARQUET)
    target = make_windows(train, SUBMIT_SPLIT, evaluation).target
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predict(target.stats, target.rows).collect().write_csv(output_path)
    check_submission(output_path, evaluation.select("rowid").collect()["rowid"])
    logger.success(f"Wrote submission to {output_path}")


if __name__ == "__main__":
    app()
