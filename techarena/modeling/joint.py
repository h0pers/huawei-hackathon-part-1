"""Best known model: LightGBM over softmax MF features and the older ALS and choice features.

Every feature for a row comes from a model fitted strictly before that row's window. Rows are
each user's first event in the window, like evaluation.csv.
"""

from pathlib import Path

import lightgbm as lgb
from loguru import logger
import numpy as np
import polars as pl
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
import typer

from techarena import softmf as sm
from techarena.config import EVAL_PARQUET, MODELS_DIR, TRAIN_PARQUET
from techarena.events import CANDIDATES, Cache, Events, load_cache
from techarena.features import FEATURES as OLD_FEATURES
from techarena.features import build_features, build_stats_tables
from techarena.validation import check_submission, first_event_per_user, log_experiment

app = typer.Typer()

DAY = 86_400
CUTOFF = 1_737_936_000  # 2025-01-27 00:00 UTC
OUTPUT = MODELS_DIR / "submission_joint.csv"

SOFTMAX = sm.Config(
    epochs=30,
    batch=4096,
    lr=0.003,
    l2_user=0.3,
    l2_item=0.3,
    l2_bias=10,
    ball_user=2,
    ball_item=2.7,
)
LGBM = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5,
    "verbose": -1,
    "num_threads": 8,
    "seed": 1,
}
# Early stopping on the Jan 24-26 validation picked 481-631 rounds across runs.
ROUNDS = 560


def per_event(values: np.ndarray) -> np.ndarray:
    return np.repeat(values[:, None], CANDIDATES, 1)


def softmax_features(
    net: sm.SoftMF, events: Events, logits: np.ndarray, fit: Events, cache: Cache
) -> dict[str, np.ndarray]:
    """Softmax outputs, fitted vectors and click counts for each row, as (events, 10) arrays."""
    v = net.v.detach().cpu().numpy()
    u = net.u.detach().cpu().numpy()
    bias = net.b.detach().cpu().numpy()
    propensity = net.a_user.detach().cpu().numpy()
    scores, outside = logits[:, :CANDIDATES], logits[:, CANDIDATES]
    probs = sm.probabilities(logits)

    item_rows = np.bincount(fit.cand.ravel(), minlength=cache.n_items + 1).astype(float)
    item_clicks = np.bincount(
        fit.cand.ravel(), weights=sm.row_labels(fit).ravel(), minlength=cache.n_items + 1
    )
    user_events = np.bincount(fit.user, minlength=cache.n_users + 1).astype(float)
    user_clicks = np.bincount(
        fit.user, weights=(fit.label < CANDIDATES) * 1.0, minlength=cache.n_users + 1
    )

    slot, count = sm.repeat_features(events)
    length = (events.hist > 0).sum(1)
    last = events.hist_ts[:, -1]
    first = np.where(events.hist > 0, events.hist_ts, np.iinfo(np.int64).max).min(1)
    unique = np.array([len(set(row[row > 0])) for row in events.hist]) / np.maximum(length, 1)

    unit = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    hist_vectors, cand_vectors = unit[events.hist], unit[events.cand]
    pooled = hist_vectors.sum(1) / np.maximum(length, 1)[:, None]
    sims = np.einsum("nci,nhi->nch", cand_vectors, hist_vectors)
    item_ctr = (item_clicks[events.cand] + 0.063 * 50) / (item_rows[events.cand] + 50)

    return {
        "sm_prob": probs,
        "sm_score": scores,
        "sm_out": per_event(outside),
        "sm_lse": per_event(logsumexp(scores, 1)),
        "sm_noclick": per_event(1 - probs.sum(1)),
        "sm_rank": (-scores).argsort(1).argsort(1),
        "sm_vs_mean": scores - scores.mean(1, keepdims=True),
        "uv": (u[events.user][:, None, :] * v[events.cand]).sum(-1),
        "b": bias[events.cand],
        "a_user": per_event(propensity[events.user]),
        "u_norm": per_event(np.linalg.norm(u[events.user], axis=1)),
        "v_norm": np.linalg.norm(v[events.cand], axis=2),
        "slot": slot,
        "count": count,
        "hist_len": per_event(length),
        "since_last": per_event(np.where(length > 0, events.ts - last, np.nan)),
        "click_rate": per_event(length / ((last - first) / DAY + 1)),
        "uniq": per_event(unique),
        "hist_sim": (pooled[:, None, :] * cand_vectors).sum(-1),
        "hist_max": sims.max(-1),
        "last_sim": sims[:, :, -1],
        "u_imps": per_event(user_events[events.user]),
        "u_rate": per_event(
            (user_clicks[events.user] + 0.63 * 5) / (user_events[events.user] + 5)
        ),
        "i_imps": item_rows[events.cand],
        "i_clicks": item_clicks[events.cand],
        "i_ctr": item_ctr,
        "i_ctr_rank": (-item_ctr).argsort(1).argsort(1),
        "i_ctr_vs": item_ctr - item_ctr.mean(1, keepdims=True),
        "in_hist_n": per_event((slot > 0).sum(1)),
        "tb": per_event(events.time_bucket),
        "dt": per_event(events.day_type),
    }


def old_features(cut: int, end: int | None) -> tuple[np.ndarray, np.ndarray]:
    """The older feature set, sorted like the event cache, and its rowids."""
    train = pl.scan_parquet(TRAIN_PARQUET)
    if end is None:
        rows = pl.scan_parquet(EVAL_PARQUET)
    else:
        rows = first_event_per_user(train.filter(pl.col("timestamp").is_between(cut, end, "left")))
    tables = build_stats_tables(train.filter(pl.col("timestamp") < cut))
    frame = build_features(tables, rows).collect().sort("event_id", "rowid")
    return frame.select(OLD_FEATURES).to_numpy().astype(np.float32), frame["rowid"].to_numpy()


def window(cache: Cache, cut: int, end: int | None) -> tuple[np.ndarray, np.ndarray | None]:
    """All features for one window, and labels unless it is the evaluation window."""
    train = cache.train
    if end is None:
        target, fit = cache.evaluation, train
    else:
        target = sm.first_per_user(train.take((train.ts >= cut) & (train.ts < end)))
        fit = train.take(train.ts < cut)
    split = sm.Split(fit=fit, target=target, name=str(cut), cut=cut)
    log = sm.click_log(fit, target)
    init = sm.als_item_vectors(log, cache.n_users, cache.n_items, f"joint_{cut}_{end}")
    net, logits, _ = sm.fit(split, cache.n_users, cache.n_items, SOFTMAX, init, log, verbose=False)
    new = softmax_features(net, target, logits, fit, cache)
    columns = np.stack([f.ravel().astype(np.float32) for f in new.values()], 1)
    old, rowids = old_features(cut, end)
    # Both sets count each candidate's impressions before the window, so this checks every row.
    old_imps = np.nan_to_num(old[:, OLD_FEATURES.index("item_impressions")])
    if not np.array_equal(old_imps, new["i_imps"].ravel().astype(np.float32)):
        raise ValueError(f"window {cut}: old and new features are not aligned")
    if end is None and not np.array_equal(rowids.astype(str), cache.eval_rowids.astype(str)):
        raise ValueError("evaluation rowids are not aligned")
    labels = None if end is None else sm.row_labels(target).ravel()
    return np.hstack([columns, old]), labels


@app.command()
def validate(target_day: int = 1_737_676_800, target_end: int = CUTOFF):
    """Report AUC on each user's first event from target_day on (default Jan 24-26)."""
    cache = load_cache()
    parts = [window(cache, target_day - k * DAY, target_day) for k in (3, 2, 1)]
    features, labels = window(cache, target_day, target_end)
    data = lgb.Dataset(np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    model = lgb.train(
        LGBM,
        data,
        4000,
        valid_sets=[lgb.Dataset(features, labels, reference=data)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    auc = roc_auc_score(labels, model.predict(features))
    log_experiment("joint", "valid", auc, rounds=model.best_iteration)
    logger.success(f"Validation AUC {auc:.5f} ({model.best_iteration} rounds)")


@app.command()
def submit(output: Path = OUTPUT, rounds: int = ROUNDS):
    """Train on the last three days and write a checked submission for evaluation.csv."""
    cache = load_cache()
    parts = [window(cache, CUTOFF - k * DAY, CUTOFF) for k in (3, 2, 1)]
    features, _ = window(cache, CUTOFF, None)
    data = lgb.Dataset(np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    predicted = lgb.train({**LGBM, "metric": "None"}, data, rounds).predict(features)
    rowids = cache.eval_rowids.astype(str).tolist()
    pl.DataFrame({"rowid": rowids, "predicted_probability": np.clip(predicted, 0, 1)}).write_csv(
        output
    )
    check_submission(output, pl.Series(rowids))
    logger.success(f"Wrote and checked {output}")


@app.command()
def check(path: Path):
    """Check any submission file against every rule in the competition instructions."""
    rowids = pl.read_parquet(EVAL_PARQUET, columns=["rowid"])["rowid"]
    check_submission(path, rowids)
    logger.success(f"{path} is valid")


if __name__ == "__main__":
    app()
