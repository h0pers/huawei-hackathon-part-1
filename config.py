from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

KAGGLE_DATASET = "dmytrohryshchenko/huawei-ireland-tech-arena-2026"

TRAIN_PARQUET = INTERIM_DATA_DIR / "train.parquet"
EVAL_PARQUET = INTERIM_DATA_DIR / "evaluation.parquet"


@dataclass(frozen=True)
class TimeSplit:
    ranker_days: tuple[datetime, ...]  # ranker learns on each day, with stats from earlier days
    target_day: datetime  # predict each user's first event from this day on, stats from before it


# Evaluation is each user's first event over the week after the cutoff, so a one-day target
# over-weights active users; first events over Jan 24-26 match evaluation's history lengths.
VALID_SPLIT = TimeSplit(
    ranker_days=tuple(datetime(2025, 1, day, tzinfo=UTC) for day in range(21, 24)),
    target_day=datetime(2025, 1, 24, tzinfo=UTC),
)
SUBMIT_SPLIT = TimeSplit(
    ranker_days=tuple(datetime(2025, 1, day, tzinfo=UTC) for day in range(24, 27)),
    target_day=datetime(2025, 1, 27, tzinfo=UTC),
)

MODELS_DIR = PROJ_ROOT / "models"
SUBMISSION_PATH = MODELS_DIR / "submission.csv"
OPTUNA_DB = MODELS_DIR / "optuna.db"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_LOG = REPORTS_DIR / "experiments.jsonl"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
