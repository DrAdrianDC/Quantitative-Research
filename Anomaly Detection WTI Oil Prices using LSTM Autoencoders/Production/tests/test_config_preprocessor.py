from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.preprocessor import TimeSeriesPreprocessor


def test_load_config_monitor_defaults():
    cfg = load_config()
    assert cfg.data.ticker == "CL=F"
    assert cfg.data.train_start_date == "2010-01-01"
    assert cfg.data.train_end_date == "2019-12-31"
    assert cfg.training.epochs == 200
    assert cfg.model.threshold_percentile == 99.0
    assert cfg.training.full_history is False
    assert cfg.preprocessing.vol_lookback == 60
    assert cfg.output.state_path.name == "feature_state.json"
    assert cfg.walkforward.folds


def test_preprocessor_state_roundtrip_is_json(tmp_path):
    n = 300
    rng = np.random.default_rng(0)
    close = 60 + np.cumsum(rng.normal(0, 0.5, size=n))
    frame = pd.DataFrame(
        {"Date": pd.date_range("2010-01-01", periods=n, freq="B"), "Close": close}
    )
    prep = TimeSeriesPreprocessor(lookback=10, vol_lookback=20, min_scale_floor=0.05)
    x_train, x_val, meta = prep.prepare_calibration(
        frame, train_end_date="2010-12-31", validation_fraction=0.2, fit_state=True
    )
    assert x_train.ndim == 3
    assert x_train.shape[1:] == (10, 1)
    assert meta["n_train_windows"] == x_train.shape[0]
    path = tmp_path / "feature_state.json"
    prep.save_state(path)
    payload = json.loads(path.read_text())
    assert "scale_floor" in payload
    assert "py/object" not in path.read_text()  # not a pickle
    loaded = TimeSeriesPreprocessor.load_state(path)
    assert loaded.scale_floor == pytest.approx(prep.scale_floor)
    assert loaded.lookback == 10


def test_quiet_mask_drops_loud_windows():
    n = 250
    changes = np.ones(n) * 0.2
    changes[200:210] = 8.0
    close = 50 + np.cumsum(changes)
    frame = pd.DataFrame(
        {"Date": pd.date_range("2010-01-01", periods=n, freq="B"), "Close": close}
    )
    prep = TimeSeriesPreprocessor(lookback=10, vol_lookback=20)
    prep.fit(frame)
    features = prep.build_frame(frame)
    quiet = prep.quiet_mask(features)
    assert quiet.sum() < features.valid_mask.sum()
    assert quiet.sum() > 32


def test_chronological_split_excludes_pre_start_and_keeps_oos_tail():
    n = 800
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2008-01-01", periods=n),
            "Close": 50.0 + np.linspace(0, 10, n),
        }
    )
    prep = TimeSeriesPreprocessor(lookback=10, vol_lookback=20)
    cal, oos = prep.chronological_split(
        frame,
        train_start_date="2010-01-01",
        train_end_date="2010-06-30",
        validation_fraction=0.15,
    )
    assert cal["Date"].min() >= pd.Timestamp("2010-01-01")
    assert cal["Date"].max() <= pd.Timestamp("2010-06-30")
    assert oos["Date"].min() > pd.Timestamp("2010-06-30")
    assert (cal["Date"] < pd.Timestamp("2010-01-01")).sum() == 0
