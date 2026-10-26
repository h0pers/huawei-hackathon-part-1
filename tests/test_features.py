import polars as pl

from techarena.features import history_features


def test_history_features_find_repeat_clicks():
    target = pl.LazyFrame(
        {
            "rowid": ["seen", "new", "empty"],
            "item_id": ["a", "c", "a"],
            "timestamp": [1000, 1000, 1000],
            "history_item_ids": [["a", "b", "a"], ["a", "b", "a"], []],
            "history_timestamps": [[100, 200, 300], [100, 200, 300], []],
        },
        schema_overrides={"history_timestamps": pl.List(pl.Int64)},
    )
    covis = pl.LazyFrame(
        {"item_id": ["c"], "item_id_hist": ["b"], "covis": [0.5]},
    )
    rows = {r["rowid"]: r for r in history_features(target, covis).collect().to_dicts()}

    assert rows["seen"]["cand_hist_count"] == 2
    assert rows["seen"]["cand_since_last_click"] == 700
    assert rows["seen"]["cand_last_pos"] == 0
    assert rows["new"]["cand_hist_count"] == 0
    assert rows["new"]["cand_since_last_click"] is None
    assert rows["new"]["covis_sum"] == 0.5
    assert rows["new"]["covis_recent"] == 0.5 / 2
    # Empty histories produce no row here; build_features fills them with 0.
    assert "empty" not in rows
