import lightgbm as lgb
from loguru import logger
import polars as pl
import typer

from techarena.config import MODELS_DIR, SUBMIT_SPLIT, TRAIN_PARQUET, VALID_SPLIT
from techarena.features import FEATURES, build_features, build_stats_tables
from techarena.validation import global_auc, log_experiment, make_windows

app = typer.Typer()

VALID_MODEL = MODELS_DIR / "lgbm_valid.txt"
SUBMIT_MODEL = MODELS_DIR / "lgbm_submit.txt"

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


def to_dataset(frame: pl.DataFrame) -> lgb.Dataset:
    return lgb.Dataset(
        frame.select(FEATURES).to_numpy(), label=frame["clicked"].to_numpy(), feature_name=FEATURES
    )


@app.command()
def main(submit: bool = False):
    """Train on the validation split and report AUC, or with --submit retrain one day later."""
    train = pl.scan_parquet(TRAIN_PARQUET)
    windows = make_windows(train, SUBMIT_SPLIT if submit else VALID_SPLIT)
    tables = build_stats_tables(windows.stats)
    ranker_rows = build_features(tables, windows.ranker).collect()
    logger.info(f"Training on {ranker_rows.height:,} rows")

    if submit:
        # Reuse the number of rounds that early stopping picked on the validation split.
        rounds = lgb.Booster(model_file=VALID_MODEL).current_iteration()
        model = lgb.train(PARAMS, to_dataset(ranker_rows), num_boost_round=rounds)
        model.save_model(SUBMIT_MODEL)
        logger.success(f"Saved {SUBMIT_MODEL} ({rounds} rounds)")
        return

    target_rows = build_features(tables, windows.target).collect()
    model = lgb.train(
        PARAMS,
        to_dataset(ranker_rows),
        num_boost_round=3000,
        valid_sets=[to_dataset(target_rows)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    model.save_model(VALID_MODEL, num_iteration=model.best_iteration)

    predicted = pl.Series(model.predict(target_rows.select(FEATURES).to_numpy()))
    auc = global_auc(target_rows["clicked"], predicted)
    log_experiment("lgbm", "valid", auc, rounds=model.best_iteration, features=len(FEATURES))

    importance = sorted(
        zip(FEATURES, model.feature_importance("gain")), key=lambda x: x[1], reverse=True
    )
    for name, gain in importance:
        logger.info(f"{name:24s} {gain:,.0f}")
    logger.success(f"Validation AUC: {auc:.5f} ({model.best_iteration} rounds)")


if __name__ == "__main__":
    app()
