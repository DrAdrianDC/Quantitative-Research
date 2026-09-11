from __future__ import annotations

import pandas as pd

from src.events import EVENT_CATALOG, cluster_episodes, event_hit_mask, events_between, events_for_era


def test_events_between_does_not_score_future_catalog_on_gfc_block():
    block = events_between("2007-12-31", "2011-12-31")
    names = {e.name for e in block}
    assert "GFC crash week" in names
    assert "WTI negative settlement" not in names
    dates = {e.date for e in EVENT_CATALOG}
    assert "2020-04-20" in dates
    assert "2008-10-10" in dates
    assert events_for_era("oos")
    assert events_for_era("calibration")
    assert events_for_era("pre")
    for event in events_for_era("oos"):
        assert event.timestamp() > pd.Timestamp("2019-12-31")
    for event in events_for_era("pre"):
        assert event.timestamp() < pd.Timestamp("2010-01-01")
    for event in events_for_era("calibration"):
        assert pd.Timestamp("2010-01-01") <= event.timestamp() <= pd.Timestamp("2019-12-31")


def test_cluster_merges_single_gap_but_not_unrelated_months():
    dates = pd.date_range("2020-04-01", periods=20, freq="D")
    flags = [0] * 20
    flags[1] = flags[2] = flags[4] = 1  # gap of one quiet day → one episode
    flags[15] = 1
    episodes = cluster_episodes(dates, flags, max_gap_days=1)
    assert len(episodes) == 2
    assert int(episodes.iloc[0]["n_days"]) == 3
    assert int(episodes.iloc[1]["n_days"]) == 1


def test_event_hit_uses_trading_neighbourhood():
    dates = pd.bdate_range("2020-04-13", periods=10)
    flags = [0] * 10
    scores = pd.DataFrame({"Date": dates, "anomaly": flags})
    from src.events import MarketEvent

    event = MarketEvent("2020-04-20", "negative print", "oos")
    miss = event_hit_mask(scores, [event], flag_column="anomaly")
    assert int(miss.iloc[0]["hit"]) == 0
    # Flag the nearest session
    nearest = int((pd.to_datetime(scores["Date"]) - pd.Timestamp("2020-04-20")).abs().argmin())
    scores.loc[nearest, "anomaly"] = 1
    hit = event_hit_mask(scores, [event], flag_column="anomaly")
    assert int(hit.iloc[0]["hit"]) == 1
