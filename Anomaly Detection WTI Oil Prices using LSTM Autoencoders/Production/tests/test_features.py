"""Causal feature contract — the monitor's input, not the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureSpec,
    build_feature_frame,
    causal_rolling_mad,
    create_windows,
    freeze_scale_floor,
    normalize_changes,
    price_changes,
)


def _series_with_crash(n: int = 400, crash_at: int = 250) -> tuple[np.ndarray, pd.DatetimeIndex]:
    rng = np.random.default_rng(0)
    changes = rng.normal(0.0, 0.4, size=n)
    changes[crash_at] = -25.0
    close = 50.0 + np.cumsum(changes)
    dates = pd.date_range("2008-01-01", periods=n, freq="B")
    return close, dates


def test_price_changes_first_day_is_nan():
    close = np.array([10.0, 11.0, 9.0])
    delta = price_changes(close)
    assert np.isnan(delta[0])
    assert delta[1] == pytest.approx(1.0)
    assert delta[2] == pytest.approx(-2.0)


def test_negative_print_does_not_nan():
    close = np.array([20.0, 18.0, -37.63, 10.0])
    delta = price_changes(close)
    assert delta[2] == pytest.approx(-55.63)
    assert np.isfinite(delta[2:]).all()


def test_causal_scale_excludes_today():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=40)
    scale = causal_rolling_mad(values, window=10)
    values_shocked = values.copy()
    values_shocked[20] = 50.0
    shocked = causal_rolling_mad(values_shocked, window=10)
    # Today's spike is not in today's scale (no look-ahead).
    assert shocked[20] == pytest.approx(scale[20])
    # It is allowed to affect tomorrow's scale (the past of t+1).
    # MAD may still ignore a single point; the hard contract is the line above.


def test_no_lookahead_when_future_close_changes():
    close, dates = _series_with_crash()
    spec = FeatureSpec(lookback=10, vol_lookback=20, scale_floor=0.05)
    f1 = build_feature_frame(close, dates, spec)
    close2 = close.copy()
    close2[-1] = close2[-1] * 3.0
    f2 = build_feature_frame(close2, dates, spec)
    # Everything except the last day's delta / x must be identical.
    np.testing.assert_allclose(f1.local_scale[:-1], f2.local_scale[:-1], equal_nan=True)
    np.testing.assert_allclose(f1.normalized_return[:-1], f2.normalized_return[:-1], equal_nan=True)


def test_crash_is_large_in_normalized_units():
    close, dates = _series_with_crash()
    spec = FeatureSpec(lookback=10, vol_lookback=20, scale_floor=0.05)
    frame = build_feature_frame(close, dates, spec)
    crash_x = frame.normalized_return[250]
    typical = np.nanmedian(np.abs(frame.normalized_return))
    assert np.isfinite(crash_x)
    assert abs(crash_x) > 10 * typical


def test_window_alignment_last_row_is_last_close():
    x = np.arange(15, dtype=np.float32)
    windows = create_windows(x, lookback=10)
    assert windows.shape == (6, 10, 1)
    assert windows[-1, -1, 0] == 14
    assert windows[0, 0, 0] == 0


def test_scale_floor_is_positive_and_frozen_from_distribution():
    scale = np.concatenate([np.full(100, 1.0), np.full(100, 2.0)])
    floor = freeze_scale_floor(scale, quantile=0.05, min_floor=0.05)
    assert floor >= 0.05
    assert floor <= 1.0
