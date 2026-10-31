from dataclasses import replace

from loguru import logger
import optuna
import polars as pl
import typer

from techarena.choice import fit_choice
from techarena.config import MODELS_DIR, OPTUNA_DB, TRAIN_PARQUET, VALID_SPLIT
from techarena.embeddings import ALS_PARAMS, fit_als
from techarena.features import StatsTables, build_features, build_stats_tables, click_log
from techarena.modeling.train import PARAMS, train_and_score
from techarena.validation import Window, log_experiment, make_windows

app = typer.Typer()

# Reruns add trials to this study; rename it when features change so old trials don't mix in.
STUDY_NAME = "als-choice-lgbm"


def suggest_params(trial: optuna.Trial) -> tuple[dict, dict]:
    als = {
        **ALS_PARAMS,
        # The choice model prefers few factors with a large alpha.
        "factors": trial.suggest_int("factors", 8, 32, log=True),
        "regularization": trial.suggest_float("regularization", 0.05, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 20.0, 2000.0, log=True),
        "iterations": trial.suggest_int("iterations", 10, 40),
    }
    lgbm = {
        **PARAMS,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1000, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
    }
    return als, lgbm


def current_params() -> dict:
    """Today's settings in the search space's names, to run as the first trial."""
    als = ("factors", "regularization", "alpha", "iterations")
    lgbm = (
        "learning_rate",
        "num_leaves",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "lambda_l2",
    )
    return {k: ALS_PARAMS[k] for k in als} | {k: PARAMS[k] for k in lgbm}


@app.command()
def main(n_trials: int = 40):
    """Search ALS and LightGBM settings together, maximising validation AUC."""
    windows = make_windows(pl.scan_parquet(TRAIN_PARQUET), VALID_SPLIT)
    # Stats tables per window are fixed; only ALS, and the choice model built on it, are refitted
    # in each trial.
    prepared = [
        (window, build_stats_tables(window.stats), click_log(window.stats))
        for window in [*windows.ranker, windows.target]
    ]

    def refit(tables: StatsTables, window: Window, clicks: pl.LazyFrame, params: dict):
        als = fit_als(clicks, params)
        return replace(tables, als=als, choice=fit_choice(window.stats, als))

    def objective(trial: optuna.Trial) -> float:
        als_params, lgbm_params = suggest_params(trial)
        frames = [
            build_features(refit(tables, window, clicks, als_params), window.rows).collect()
            for window, tables, clicks in prepared
        ]
        ranker_rows = pl.concat(frames[:-1], how="vertical_relaxed")
        target_rows = frames[-1]
        model, auc = train_and_score(ranker_rows, target_rows, lgbm_params)
        log_experiment(
            "lgbm",
            "valid",
            auc,
            trial=trial.number,
            rounds=model.best_iteration,
            als=als_params,
            lgbm=lgbm_params,
        )
        return auc

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{OPTUNA_DB}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(current_params())
    study.optimize(objective, n_trials=n_trials)

    logger.success(f"Best AUC {study.best_value:.5f} (trial {study.best_trial.number})")
    for name, value in study.best_params.items():
        logger.info(f"{name:18s} {value}")
    logger.info("Parameter importance:")
    for name, importance in optuna.importance.get_param_importances(study).items():
        logger.info(f"{name:18s} {importance:.3f}")


if __name__ == "__main__":
    app()
