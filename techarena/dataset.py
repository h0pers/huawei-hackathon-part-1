from pathlib import Path

import kagglehub
from loguru import logger
import polars as pl
import typer

from techarena.config import INTERIM_DATA_DIR, KAGGLE_DATASET, RAW_DATA_DIR

app = typer.Typer()

RAW_FILES = ("train.csv", "evaluation.csv", "example_submission.csv")

# Identifiers are opaque text; histories are JSON arrays stored as quoted CSV fields.
CSV_SCHEMA = {
    "rowid": pl.String,
    "event_id": pl.String,
    "user_id": pl.String,
    "item_id": pl.String,
    "timestamp": pl.Int64,
    "time_bucket": pl.Int8,
    "day_type": pl.Int8,
    "history_item_ids": pl.String,
    "history_timestamps": pl.String,
    "clicked": pl.Int8,
}


@app.command()
def download(
    dataset: str = KAGGLE_DATASET,
    output_dir: Path = RAW_DATA_DIR,
    force: bool = False,
):
    """Download the competition CSVs from Kaggle into data/raw."""
    for name in RAW_FILES:
        if not force and (output_dir / name).exists():
            logger.info(f"{name} already present, skipping (use --force).")
            continue
        logger.info(f"Downloading {name} from {dataset}...")
        kagglehub.dataset_download(
            dataset, path=name, force_download=force, output_dir=str(output_dir)
        )
    logger.success(f"Raw files ready in {output_dir}")


def csv_to_parquet(input_path: Path, output_path: Path) -> None:
    """Stream a raw CSV into a typed Parquet file with histories decoded into lists."""
    with input_path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    schema = {col: CSV_SCHEMA[col] for col in header}

    logger.info(f"Converting {input_path.name} -> {output_path.name}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_csv(input_path, schema=schema)
        .with_columns(
            pl.col("history_item_ids").str.json_decode(pl.List(pl.String)),
            pl.col("history_timestamps").str.json_decode(pl.List(pl.Int64)),
        )
        .sink_parquet(output_path, compression="zstd")
    )

    stats = (
        pl.scan_parquet(output_path)
        .select(
            pl.len().alias("rows"),
            (pl.col("history_item_ids").list.len() != pl.col("history_timestamps").list.len())
            .sum()
            .alias("misaligned_histories"),
        )
        .collect()
        .row(0, named=True)
    )
    if stats["misaligned_histories"]:
        raise ValueError(
            f"{output_path.name}: {stats['misaligned_histories']} misaligned histories"
        )
    logger.success(f"Wrote {stats['rows']:,} rows to {output_path}")


@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR,
    output_dir: Path = INTERIM_DATA_DIR,
):
    """Convert the raw train/evaluation CSVs into Parquet in data/interim."""
    for name in ("train", "evaluation"):
        csv_to_parquet(input_dir / f"{name}.csv", output_dir / f"{name}.parquet")


if __name__ == "__main__":
    app()
