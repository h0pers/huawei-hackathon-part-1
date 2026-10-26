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
    stats_until: datetime  # stats: events before this
    ranker_until: datetime  # ranker: events from stats_until to this; target: after


# Practice exam on Jan 26; the real one is the same split shifted a day.
VALID_SPLIT = TimeSplit(datetime(2025, 1, 25, tzinfo=UTC), datetime(2025, 1, 26, tzinfo=UTC))
SUBMIT_SPLIT = TimeSplit(datetime(2025, 1, 26, tzinfo=UTC), datetime(2025, 1, 27, tzinfo=UTC))

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
