import polars as pl

from techarena.choice import CANDIDATES, HISTORY, NO_CLICK, to_events


def event(event_id: str, user: str, history: list[str], clicked_at: int | None) -> list[dict]:
    return [
        {
            "event_id": event_id,
            "rowid": f"{event_id}_{i}",
            "user_id": user,
            "item_id": f"i{i}",
            "history_item_ids": history,
            "clicked": int(i == clicked_at),
        }
        for i in range(CANDIDATES)
    ]


def test_to_events_aligns_labels_history_and_unknowns():
    rows = pl.DataFrame(event("b", "u1", ["i3", "i1"], clicked_at=7) + event("a", "u9", [], None))
    users = pl.DataFrame({"user_id": ["u1"], "idx": [1]})
    items = pl.DataFrame({"item_id": [f"i{i}" for i in range(9)], "idx": list(range(1, 10))})

    events = to_events(rows.sample(fraction=1, shuffle=True, seed=0), users, items)

    # Events sorted by id: "a" (no click, unknown user) then "b".
    assert events.label.tolist() == [NO_CLICK, 7]
    assert events.user.tolist() == [0, 1]
    assert events.rowid.to_list()[:2] == ["a_0", "a_1"]
    # History is right-aligned, newest last; i9 is not in the item table, so it maps to 0.
    assert events.hist[1, -2:].tolist() == [4, 2]
    assert events.hist[0].tolist() == [0] * HISTORY
    assert events.cand[1].tolist() == [*range(1, 10), 0]
