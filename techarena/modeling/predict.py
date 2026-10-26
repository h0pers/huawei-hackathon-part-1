from pathlib import Path

import lightgbm as lgb
from loguru import logger
import polars as pl
import typer

from techarena.config import EVAL_PARQUET, SUBMISSION_PATH, SUBMIT_SPLIT, TRAIN_PARQUET
from techarena.features import FEATURES, build_features, build_stats_tables
from techarena.modeling.train import SUBMIT_MODEL
from techarena.validation import check_submission, make_windows

app = typer.Typer()


@app.command()
def main(model_path: Path = SUBMIT_MODEL, output_path: Path = SUBMISSION_PATH):
    """Score evaluation.csv with the submit model and write a checked submission."""
    evaluation = pl.scan_parquet(EVAL_PARQUET)
    windows = make_windows(pl.scan_parquet(TRAIN_PARQUET), SUBMIT_SPLIT, evaluation)
    rows = build_features(build_stats_tables(windows.stats), windows.target).collect()

    model = lgb.Booster(model_file=model_path)
    predicted = model.predict(rows.select(FEATURES).to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.select("rowid").with_columns(predicted_probability=predicted).write_csv(output_path)

    check_submission(output_path, evaluation.select("rowid").collect()["rowid"])
    logger.success(f"Wrote submission to {output_path}")


if __name__ == "__main__":
    app()
