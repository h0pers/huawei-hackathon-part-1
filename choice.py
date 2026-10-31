from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import nn

from techarena.embeddings import Als

HISTORY = 20
CANDIDATES = 10
NO_CLICK = CANDIDATES  # label of events where nothing was clicked

# Picked on the Jan 24-26 validation. The item vectors stay fixed at ALS's: training them, or a
# 50-dim taste vector per user, memorises the training events and AUC falls with every epoch
# after the first (0.794 -> 0.737 alone). With only scalars per item and per user left to
# learn, AUC alone is 0.795 and stable over 4-8 epochs.
CHOICE_PARAMS = {
    "epochs": 8,
    "lr": 1e-3,
    "batch_size": 1024,
    "seed": 42,
}

# torch and the lightgbm wheel load separate OpenMP runtimes on macOS, which segfault when both
# start thread pools (pytorch#191933, LightGBM#6595). With lightgbm imported first (see
# techarena/__init__.py) and torch kept to one CPU thread, only lightgbm's pool ever starts;
# the GPU, when present, keeps training fast.
torch.set_num_threads(1)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ChoiceNet(nn.Module):
    """Scores each candidate and a no-click option; softmax over the 11 gives the choice."""

    def __init__(self, item_vectors: torch.Tensor, n_users: int):
        super().__init__()
        n_items = item_vectors.shape[0]
        self.user_no_click = nn.Embedding(n_users, 1)  # each user's propensity to click nothing
        nn.init.zeros_(self.user_no_click.weight)
        self.item = nn.Embedding.from_pretrained(item_vectors.clone(), freeze=True)
        self.scale = nn.Parameter(torch.tensor(5.0))  # sharpness of the taste match
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.item_bias.weight)
        self.recency = nn.Parameter(torch.zeros(HISTORY))  # weight of each history slot
        self.repeat_slot = nn.Embedding(HISTORY + 1, 1)  # bonus by slot of the latest repeat
        nn.init.zeros_(self.repeat_slot.weight)
        self.repeat_count = nn.Parameter(torch.zeros(1))
        self.no_click_len = nn.Embedding(HISTORY + 1, 1)
        nn.init.zeros_(self.no_click_len.weight)
        self.no_click = nn.Parameter(torch.zeros(1))

    def forward(
        self, user_idx: torch.Tensor, hist: torch.Tensor, cand: torch.Tensor
    ) -> torch.Tensor:
        """Logits (batch, 11) with no-click last."""
        known = hist > 0
        w = torch.where(known, self.recency.exp(), torch.zeros_like(self.recency))
        w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
        # Cosine between the user's mean history direction and the candidate.
        user = nn.functional.normalize((w.unsqueeze(-1) * self.item(hist)).sum(1), dim=-1)
        item = self.item(cand)

        match = (cand.unsqueeze(-1) == hist.unsqueeze(1)) & known.unsqueeze(1)
        slot = torch.arange(HISTORY, device=hist.device)
        latest = torch.where(match, slot, -1).max(-1).values + 1  # 0 = never clicked
        scores = (
            self.scale * (user.unsqueeze(1) * item).sum(-1)
            + self.item_bias(cand).squeeze(-1)
            + self.repeat_slot(latest).squeeze(-1)
            + self.repeat_count * torch.log1p(match.sum(-1).float())
        )
        no_click = (
            self.no_click
            + self.no_click_len(known.sum(1)).squeeze(-1)
            + self.user_no_click(user_idx).squeeze(-1)
        )
        return torch.cat([scores, no_click.unsqueeze(1)], 1)


@dataclass(frozen=True)
class Choice:
    users: pl.DataFrame  # user_id -> idx, 0 is reserved for users unseen in the stats window
    items: pl.DataFrame  # item_id -> idx, 0 is reserved for unknown items and padding
    net: ChoiceNet


@dataclass(frozen=True)
class Events:
    rowid: pl.Series  # candidate rows in the same order as cand.ravel()
    user: np.ndarray  # (events,) user idx
    hist: np.ndarray  # (events, 20) item idx, right-aligned so the last column is newest
    cand: np.ndarray  # (events, 10) item idx
    label: np.ndarray  # (events,) clicked candidate position, or NO_CLICK

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.from_numpy(a).to(DEVICE) for a in (self.user, self.hist, self.cand, self.label)
        )


def to_events(rows: pl.DataFrame, users: pl.DataFrame, items: pl.DataFrame) -> Events:
    """Reshape candidate rows into one record per event."""
    has_label = "clicked" in rows.columns
    events = (
        rows.sort("event_id", "rowid")
        .group_by("event_id", maintain_order=True)
        .agg(
            pl.col("user_id").first(),
            pl.col("history_item_ids").first(),
            pl.col("rowid"),
            pl.col("item_id"),
            *([pl.col("clicked")] if has_label else []),
        )
        .with_row_index("e")
    )

    def idx(col: pl.Expr, table: pl.DataFrame = items) -> pl.Expr:
        # replace_strict keeps row order, so candidates stay aligned with their labels.
        return col.replace_strict(table[table.columns[0]], table["idx"], default=0)

    history = (
        events.select("e", "history_item_ids")
        .explode("history_item_ids")
        .drop_nulls()
        .select(
            "e",
            idx(pl.col("history_item_ids")).alias("idx"),
            pos=pl.int_range(pl.len()).over("e"),
            n=pl.len().over("e"),
        )
    )
    hist = np.zeros((events.height, HISTORY), dtype=np.int64)
    column = (HISTORY - history["n"] + history["pos"]).to_numpy()
    hist[history["e"].to_numpy(), column] = history["idx"].to_numpy()
    cand = events.select(idx(pl.col("item_id").explode()))["item_id"].to_numpy()

    label = np.full(events.height, NO_CLICK)
    if has_label:
        clicked = np.stack(events["clicked"].to_numpy())
        label = np.where(clicked.max(1) == 1, clicked.argmax(1), NO_CLICK)
    user = events.select(idx(pl.col("user_id"), users))["user_id"].to_numpy()
    return Events(
        rowid=events["rowid"].explode(),
        user=user.astype(np.int64),
        hist=hist,
        cand=cand.astype(np.int64).reshape(-1, CANDIDATES),
        label=label.astype(np.int64),
    )


def initial_vectors(item_unit: np.ndarray) -> torch.Tensor:
    """Unit-length ALS item vectors behind a zero row for idx 0."""
    unit = torch.from_numpy(np.asarray(item_unit, dtype=np.float32))
    return torch.cat([torch.zeros(1, unit.shape[1]), unit])


def fit_choice(stats: pl.LazyFrame, als: Als, params: dict = CHOICE_PARAMS) -> Choice:
    """Train on every impression in the stats window, on top of the ALS item vectors."""
    items = als.items.select("item_id", (pl.col("item_idx") + 1).alias("idx"))
    rows = stats.select(
        "event_id", "rowid", "user_id", "item_id", "history_item_ids", "clicked"
    ).collect()
    users = (
        rows.select("user_id").unique().sort("user_id").with_row_index("idx", offset=1)
    ).select("user_id", "idx")

    torch.manual_seed(params["seed"])
    net = ChoiceNet(initial_vectors(als.item_unit), users.height + 1).to(DEVICE)
    user, hist, cand, label = to_events(rows, users, items).tensors()
    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=params["lr"])
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(params["epochs"]):
        for batch in torch.randperm(len(label), device=DEVICE).split(params["batch_size"]):
            loss = loss_fn(net(user[batch], hist[batch], cand[batch]), label[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    net.eval()
    return Choice(users=users, items=items, net=net)


def choice_features(choice: Choice, target: pl.DataFrame) -> pl.DataFrame:
    """Per row: click probability, candidate score and the event's no-click score."""
    events = to_events(target, choice.users, choice.items)
    user, hist, cand, _ = events.tensors()
    batches = zip(user.split(8192), hist.split(8192), cand.split(8192))
    with torch.no_grad():
        logits = torch.cat([choice.net(u, h, c) for u, h, c in batches]).cpu()
    probs = torch.softmax(logits, 1)
    return pl.DataFrame(
        {
            "rowid": events.rowid,
            "choice_prob": probs[:, :CANDIDATES].numpy().ravel(),
            "choice_score": logits[:, :CANDIDATES].numpy().ravel(),
            "choice_no_click": np.repeat(logits[:, NO_CLICK].numpy(), CANDIDATES),
        }
    )
