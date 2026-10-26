import lightgbm as lgb
from loguru import logger
import polars as pl
import typer

from techarena.config import MODELS_DIR, SUBMIT_SPLIT, TRAIN_PARQUET, VALID_SPLIT
from techarena.features import FEATURES, window_features
from techarena.validation import global_auc, log_experiment, make_windows

app = typer.Typer()

VALID_MODEL = MODELS_DIR / "lgbm_valid.txt"
SUBMIT_MODEL = MODELS_DIR / "lgbm_submit.txt"

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    # Tuned together with ALS_PARAMS by modeling.tune.
    "learning_rate": 0.014,
    "num_leaves": 86,
    "min_data_in_leaf": 130,
    "feature_fraction": 0.91,
    "bagging_fraction": 0.74,
    "bagging_freq": 1,
    "lambda_l2": 0.0077,
    "verbose": -1,
}


def to_dataset(frame: pl.DataFrame) -> lgb.Dataset:
    return lgb.Dataset(
        frame.select(FEATURES).to_numpy(), label=frame["clicked"].to_numpy(), feature_name=FEATURES
    )


def train_and_score(
    ranker_rows: pl.DataFrame, target_rows: pl.DataFrame, params: dict = PARAMS
) -> tuple[lgb.Booster, float]:
    """Train with early stopping on the target rows and return the model and its AUC."""
    model = lgb.train(
        params,
        to_dataset(ranker_rows),
        num_boost_round=3000,
        valid_sets=[to_dataset(target_rows)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    predicted = pl.Series(model.predict(target_rows.select(FEATURES).to_numpy()))
    return model, global_auc(target_rows["clicked"], predicted)


@app.command()
def main(submit: bool = False, ranker_days: int = 3):
    """Train on the validation split and report AUC, or with --submit retrain one day later."""
    train = pl.scan_parquet(TRAIN_PARQUET)
    windows = make_windows(train, SUBMIT_SPLIT if submit else VALID_SPLIT)
    ranker_rows = pl.concat(
        [window_features(w) for w in windows.ranker[-ranker_days:]], how="vertical_relaxed"
    )
    logger.info(f"Training on {ranker_rows.height:,} rows from {ranker_days} days")

    if submit:
        # Reuse the number of rounds that early stopping picked on the validation split.
        rounds = lgb.Booster(model_file=VALID_MODEL).current_iteration()
        model = lgb.train(PARAMS, to_dataset(ranker_rows), num_boost_round=rounds)
        model.save_model(SUBMIT_MODEL)
        logger.success(f"Saved {SUBMIT_MODEL} ({rounds} rounds)")
        return

    target_rows = window_features(windows.target)
    model, auc = train_and_score(ranker_rows, target_rows)
    model.save_model(VALID_MODEL, num_iteration=model.best_iteration)
    log_experiment(
        "lgbm",
        "valid",
        auc,
        rounds=model.best_iteration,
        features=len(FEATURES),
        ranker_days=ranker_days,
    )

    importance = sorted(
        zip(FEATURES, model.feature_importance("gain")), key=lambda x: x[1], reverse=True
    )
    for name, gain in importance:
        logger.info(f"{name:24s} {gain:,.0f}")
    logger.success(f"Validation AUC: {auc:.5f} ({model.best_iteration} rounds)")


if __name__ == "__main__":
    app()
