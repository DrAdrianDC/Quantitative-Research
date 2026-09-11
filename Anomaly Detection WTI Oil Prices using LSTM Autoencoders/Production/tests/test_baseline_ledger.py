from __future__ import annotations

import numpy as np
import pandas as pd

from src.baseline import RobustReturnZScore, RollingVolBaseline
from src.ledger import append_new_rows, assert_event_integrity, LedgerError
import pytest


def test_zscore_survives_negative_print():
    rng = np.random.default_rng(0)
    quiet = 20.0 + np.cumsum(rng.normal(0, 0.3, size=80))
    close = np.concatenate([quiet, np.array([-37.63, 10.0])])
    det = RobustReturnZScore(percentile=90.0).fit(quiet)
    change, z, flags = det.transform(close)
    assert np.isfinite(z[-2])
    assert flags[-2] == 1
    assert change[-2] < -30


def test_rolling_vol_flags_loud_window_not_single_quiet_day():
    rng = np.random.default_rng(0)
    quiet_chg = rng.normal(0, 0.2, size=80)
    close_quiet = 50 + np.cumsum(quiet_chg)
    loud_chg = rng.normal(0.0, 3.0, size=15)
    close = 50 + np.cumsum(np.concatenate([quiet_chg, loud_chg]))
    det = RollingVolBaseline(lookback=10, percentile=90.0).fit(close_quiet)
    _vol, flags = det.transform(close)
    assert flags[-1] == 1
    assert flags[20] == 0


def test_append_does_not_rewrite_history():
    ledger = pd.DataFrame(
        {
            "Date": ["2020-04-20", "2020-04-21"],
            "reconstruction_mse": [0.2, 0.1],
            "anomaly": [1, 1],
        }
    )
    scored = pd.DataFrame(
        {
            "Date": ["2020-04-20", "2020-04-22"],
            "reconstruction_mse": [9.9, 0.01],
            "anomaly": [0, 0],
        }
    )
    out = append_new_rows(ledger, scored)
    assert list(out["Date"]) == ["2020-04-20", "2020-04-21", "2020-04-22"]
    assert float(out.loc[out["Date"] == "2020-04-20", "reconstruction_mse"].iloc[0]) == 0.2
    assert int(out.loc[out["Date"] == "2020-04-20", "anomaly"].iloc[0]) == 1


def test_event_integrity_rejects_dropped_flag():
    ledger = pd.DataFrame(
        {"Date": ["2020-04-20"], "reconstruction_mse": [0.2], "anomaly": [0]}
    )
    with pytest.raises(LedgerError):
        assert_event_integrity(ledger, ("2020-04-20",))
