"""Softmax matrix factorization with an outside option, fitted by MAP on every impression.

An event's chosen outcome follows a softmax over its 10 candidates and "no click":

    candidate i: b_i + <u, v_i> + repeat(slot, count)     no click: a + a_user + g(history length)

Warm-up clicks have no logged impressions, so each contributes a partial likelihood: the clicked
item against negatives drawn from the impression distribution, with no outside option because
we know a click happened.
"""

from dataclasses import dataclass

from implicit.als import AlternatingLeastSquares
from loguru import logger
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from techarena.config import PROCESSED_DATA_DIR
from techarena.embeddings import ALS_PARAMS, ALS_SEEDS, consensus
from techarena.events import CANDIDATES, HISTORY, NO_CLICK, Events

torch.set_num_threads(1)  # see techarena/choice.py: OpenMP clash with lightgbm on macOS
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
WARMUP_END = 1737331200  # 2025-01-20 00:00 UTC


@dataclass(frozen=True)
class Split:
    fit: Events  # labelled events before the cut
    target: Events  # events to predict
    name: str
    cut: int


def first_per_user(events: Events) -> Events:
    order = np.lexsort((np.arange(len(events)), events.ts))
    _, first = np.unique(events.user[order], return_index=True)
    return events.take(np.sort(order[first]))


def click_log(fit: Events, target: Events) -> np.ndarray:
    """Distinct (user, item, ts) clicks knowable before the target: histories plus fit labels."""
    logs = []
    for events in (fit, target):
        m = events.hist > 0
        users = np.repeat(events.user, HISTORY).reshape(-1, HISTORY)
        logs.append(np.stack([users[m], events.hist[m], events.hist_ts[m]], 1))
    clicked = fit.label < NO_CLICK
    items = np.take_along_axis(fit.cand[clicked], fit.label[clicked][:, None], 1).ravel()
    logs.append(np.stack([fit.user[clicked], items, fit.ts[clicked]], 1))
    return np.unique(np.vstack(logs), axis=0)


def als_item_vectors(
    log: np.ndarray,
    n_users: int,
    n_items: int,
    key: str,
    seeds: int = ALS_SEEDS,
    seed_base: int = 42,
) -> np.ndarray:
    """Consensus ALS item directions, (n_items + 1, 8) with row 0 for padding. Cached by key."""
    path = PROCESSED_DATA_DIR / f"als_{key}.npy"
    if path.exists():
        return np.load(path)
    matrix = csr_matrix(
        (np.ones(len(log), dtype=np.float32), (log[:, 0], log[:, 1])),
        shape=(n_users + 1, n_items + 1),
    )
    matrix.sum_duplicates()
    blocks = []
    for seed in range(seeds):
        model = AlternatingLeastSquares(**{**ALS_PARAMS, "random_state": seed_base + seed})
        with threadpool_limits(1, "blas"):
            model.fit(matrix, show_progress=False)
        vectors = np.asarray(model.item_factors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        blocks.append(vectors / np.maximum(norms, 1e-9) / np.sqrt(seeds))
    unit = consensus(np.hstack(blocks))
    unit[0] = 0
    np.save(path, unit)
    return unit


def repeat_features(events: Events) -> tuple[np.ndarray, np.ndarray]:
    """For each candidate: 1 + slot of its latest history occurrence (0 = never), and count."""
    match = (events.cand[:, :, None] == events.hist[:, None, :]) & (events.hist[:, None, :] > 0)
    slot = np.where(match.any(-1), HISTORY - match[:, :, ::-1].argmax(-1), 0)
    return slot.astype(np.int64), match.sum(-1).astype(np.float32)


@dataclass
class Config:
    epochs: int = 12
    batch: int = 8192
    lr: float = 0.01
    l2_user: float = 1.0  # prior precision, in units of one event's log-likelihood
    l2_item: float = 1.0
    l2_bias: float = 0.1
    l2_outside: float = 0.3
    init_scale: float = 2.0
    warm_negatives: int = 9
    seed: int = 0
    ball_user: float = 0.0  # if set, cap user vector norms here after every step
    ball_item: float = 0.0  # same for item vectors


class SoftMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, cfg: Config, init: np.ndarray):
        super().__init__()
        self.v = nn.Parameter(torch.from_numpy(init.astype(np.float32)) * cfg.init_scale)
        self.u = nn.Parameter(torch.zeros(n_users + 1, init.shape[1]))
        self.b = nn.Parameter(torch.zeros(n_items + 1))
        self.a_user = nn.Parameter(torch.zeros(n_users + 1))
        self.a = nn.Parameter(torch.zeros(1))
        self.g = nn.Parameter(torch.zeros(HISTORY + 1))
        self.repeat_slot = nn.Parameter(torch.zeros(HISTORY + 1))
        self.repeat_count = nn.Parameter(torch.zeros(1))

    def logits(self, user, cand, slot, count, length) -> torch.Tensor:
        scores = self.b[cand] + (self.u[user].unsqueeze(1) * self.v[cand]).sum(-1)
        scores = scores + self.repeat_slot[slot] + self.repeat_count * torch.log1p(count)
        outside = self.a + self.a_user[user] + self.g[length]
        return torch.cat([scores, outside.unsqueeze(1)], 1)

    def prior(self, cfg: Config) -> torch.Tensor:
        return (
            cfg.l2_user * self.u.square().sum()
            + cfg.l2_item * self.v.square().sum()
            + cfg.l2_bias * self.b.square().sum()
            + cfg.l2_outside * self.a_user.square().sum()
        )


def to_device(array: np.ndarray, kind=torch.long) -> torch.Tensor:
    return torch.as_tensor(array).to(DEVICE, dtype=kind)


def event_tensors(events: Events) -> tuple[torch.Tensor, ...]:
    slot, count = repeat_features(events)
    return (
        to_device(events.user),
        to_device(events.cand),
        to_device(slot),
        to_device(count, torch.float32),
        to_device((events.hist > 0).sum(1)),
    )


def predict_logits(
    net: SoftMF, tensors: tuple[torch.Tensor, ...], size: int = 16384
) -> np.ndarray:
    with torch.no_grad():
        parts = [net.logits(*chunk) for chunk in zip(*(t.split(size) for t in tensors))]
    return torch.cat(parts).cpu().numpy().astype(np.float64)


def probabilities(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(1, keepdims=True))
    return (exp / exp.sum(1, keepdims=True))[:, :CANDIDATES]


def row_labels(events: Events) -> np.ndarray:
    out = np.zeros((len(events), CANDIDATES), dtype=np.int8)
    clicked = events.label < NO_CLICK
    out[np.flatnonzero(clicked), events.label[clicked]] = 1
    return out


def fit(
    split: Split,
    n_users: int,
    n_items: int,
    cfg: Config,
    init: np.ndarray,
    log: np.ndarray,
    verbose: bool = True,
) -> tuple[SoftMF, np.ndarray, list[float]]:
    """Fit by Adam on the MAP objective; returns the net, target logits and per-epoch AUC."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = SoftMF(n_users, n_items, cfg, init).to(DEVICE)
    train = event_tensors(split.fit)
    labels = to_device(split.fit.label)
    target = event_tensors(split.target)
    truth = row_labels(split.target).ravel() if split.target.label[0] >= 0 else None
    n = len(split.fit)

    warm = log[log[:, 2] < WARMUP_END]
    warm_user, warm_item = to_device(warm[:, 0]), to_device(warm[:, 1])
    counts = np.bincount(split.fit.cand.ravel(), minlength=n_items + 1).astype(np.float64)
    cdf = np.cumsum(counts / counts.sum())

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    aucs: list[float] = []
    total = n + len(warm)
    for epoch in range(cfg.epochs):
        net.train()
        order = torch.randperm(n, device=DEVICE)
        steps = max(1, n // cfg.batch)
        warm_order = torch.randperm(len(warm), device=DEVICE)
        warm_size = int(len(warm) / steps) + 1
        negatives = to_device(
            np.searchsorted(cdf, rng.random((len(warm), cfg.warm_negatives))).clip(1, n_items)
        )
        epoch_loss = 0.0
        for step in range(steps):
            index = order[step * cfg.batch : (step + 1) * cfg.batch]
            batch = tuple(t[index] for t in train)
            loss = nn.functional.cross_entropy(net.logits(*batch), labels[index], reduction="sum")
            w = warm_order[step * warm_size : (step + 1) * warm_size]
            if len(w):
                # The clicked item sits in column 0 of each choice set.
                choices = torch.cat([warm_item[w].unsqueeze(1), negatives[w]], 1)
                scores = net.b[choices] + (net.u[warm_user[w]].unsqueeze(1) * net.v[choices]).sum(
                    -1
                )
                loss = loss + nn.functional.cross_entropy(
                    scores,
                    torch.zeros(len(w), dtype=torch.long, device=DEVICE),
                    reduction="sum",
                )
            # Scale the prior to the share of the data this step sees, so one pass adds up to
            # the full MAP objective.
            size = cfg.batch + warm_size
            objective = (loss + net.prior(cfg) * size / total) / size
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                for param, radius in ((net.u, cfg.ball_user), (net.v, cfg.ball_item)):
                    if radius > 0:
                        norm = param.norm(dim=1, keepdim=True).clamp_min(1e-9)
                        param.mul_((radius / norm).clamp_max(1.0))
            epoch_loss += float(loss.detach())
        net.eval()
        if truth is not None:
            auc = roc_auc_score(truth, probabilities(predict_logits(net, target)).ravel())
            aucs.append(auc)
            if verbose:
                logger.info(
                    f"{split.name} epoch {epoch + 1}: loss {epoch_loss / total:.4f} auc {auc:.5f}"
                )
    return net, predict_logits(net, target), aucs
