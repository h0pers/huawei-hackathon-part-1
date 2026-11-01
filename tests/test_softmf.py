import numpy as np

from techarena.events import CANDIDATES, HISTORY, NO_CLICK, Events
from techarena.softmf import (
    click_log,
    first_per_user,
    probabilities,
    repeat_features,
    row_labels,
)


def events(user, ts, hist=None, hist_ts=None, cand=None, label=None) -> Events:
    n = len(user)
    return Events(
        user=np.array(user),
        ts=np.array(ts),
        time_bucket=np.zeros(n, dtype=np.int8),
        day_type=np.zeros(n, dtype=np.int8),
        hist=np.zeros((n, HISTORY), dtype=np.int32) if hist is None else np.array(hist),
        hist_ts=np.zeros((n, HISTORY), dtype=np.int64) if hist_ts is None else np.array(hist_ts),
        cand=np.arange(1, n * CANDIDATES + 1).reshape(n, CANDIDATES) if cand is None else cand,
        label=np.full(n, NO_CLICK) if label is None else np.array(label),
    )


def test_first_per_user_keeps_each_users_earliest_event():
    kept = first_per_user(events(user=[1, 2, 1, 2], ts=[30, 20, 10, 40]))
    assert sorted(zip(kept.user.tolist(), kept.ts.tolist())) == [(1, 10), (2, 20)]


def test_repeat_features_find_latest_slot_and_count():
    hist = np.zeros((1, HISTORY), dtype=np.int32)
    hist[0, -3:] = [7, 9, 7]  # item 7 clicked twice, most recently in the last slot
    cand = np.array([[7, 9, 3, 4, 5, 6, 8, 10, 11, 12]])
    slot, count = repeat_features(events(user=[1], ts=[1], hist=hist, cand=cand))
    assert slot[0, 0] == HISTORY and count[0, 0] == 2
    assert slot[0, 1] == HISTORY - 1 and count[0, 1] == 1
    assert slot[0, 2] == 0 and count[0, 2] == 0


def test_row_labels_leave_no_click_events_empty():
    labels = row_labels(events(user=[1, 2], ts=[1, 2], label=[3, NO_CLICK]))
    assert labels[0].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert labels[1].sum() == 0


def test_click_log_never_contains_target_labels():
    fit = events(user=[1], ts=[100], label=[0])
    target = events(user=[1], ts=[200], label=[5])
    log = click_log(fit, target)
    assert log.tolist() == [[1, int(fit.cand[0, 0]), 100]]


def test_probabilities_leave_room_for_no_click():
    logits = np.zeros((1, CANDIDATES + 1))
    probs = probabilities(logits)
    assert probs.shape == (1, CANDIDATES)
    assert np.isclose(probs.sum(), CANDIDATES / (CANDIDATES + 1))
