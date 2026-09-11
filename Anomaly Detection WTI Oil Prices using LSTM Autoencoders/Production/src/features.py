"""Causal locally-normalized returns for the WTI anomaly monitor.

Reconstructing RobustScaler Close *levels* makes reconstruction MSE track
10-day realized volatility (Spearman ~0.72 on this series). That flags
busy markets, not unusual 10-day shapes. Dividing each ΔClose by a causal
60-day MAD of *past* differences puts every day in local-scale units: a
$2 move when recent MAD is $1 is large; the same $2 when recent MAD is $4
is ordinary. The network never sees the WTI print in dollars.

1. First difference of Close in USD/bbl (percent/log returns explode when
   WTI prints negative, 20 April 2020).
2. Divide by a **causal** 60-day rolling MAD of past differences. Today's
   shock does not enter today's scale.
3. Freeze a scale floor at calibration so a dead tape cannot explode x.

Look-ahead contract (enforced by tests)
---------------------------------------
``normalized_return[t]`` and ``local_scale[t]`` are functions of
``Close[0], …, Close[t]`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MAD_TO_STD = 1.4826


class FeatureError(ValueError):
    """Raised when causal features cannot be built."""


@dataclass(frozen=True)
class FeatureSpec:
    lookback: int = 10
    vol_lookback: int = 60
    scale_floor: float = 0.10
    mad_to_std: float = MAD_TO_STD
    date_column: str = "Date"
    target_column: str = "Close"

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookback": int(self.lookback),
            "vol_lookback": int(self.vol_lookback),
            "scale_floor": float(self.scale_floor),
            "mad_to_std": float(self.mad_to_std),
            "date_column": self.date_column,
            "target_column": self.target_column,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSpec:
        return cls(
            lookback=int(payload.get("lookback", 10)),
            vol_lookback=int(payload.get("vol_lookback", 60)),
            scale_floor=float(payload.get("scale_floor", 0.10)),
            mad_to_std=float(payload.get("mad_to_std", MAD_TO_STD)),
            date_column=str(payload.get("date_column", "Date")),
            target_column=str(payload.get("target_column", "Close")),
        )


def price_changes(close: np.ndarray) -> np.ndarray:
    """ΔClose in USD/bbl. First observation is NaN (no previous close)."""
    arr = np.asarray(close, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise FeatureError("Close series is empty.")
    out = np.full(arr.size, np.nan, dtype=np.float64)
    if arr.size >= 2:
        out[1:] = np.diff(arr)
    return out


def causal_rolling_mad(
    values: np.ndarray,
    window: int,
    *,
    mad_to_std: float = MAD_TO_STD,
) -> np.ndarray:
    """Rolling MAD of ``values[t-window:t]`` (excludes ``values[t]``).

    ``scale[t]`` is therefore a function of the past only. A spike at t
    inflates scale from t+1 onward, not t itself — that is what makes
    today's move large in normalized units.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if window < 8:
        raise FeatureError("vol_lookback must be >= 8.")
    n = arr.size
    scale = np.full(n, np.nan, dtype=np.float64)
    for t in range(window, n):
        past = arr[t - window : t]
        if np.isnan(past).any():
            continue
        med = float(np.median(past))
        mad = float(np.median(np.abs(past - med)))
        scale[t] = mad_to_std * mad
    return scale


def normalize_changes(
    changes: np.ndarray,
    scale: np.ndarray,
    *,
    floor: float,
) -> np.ndarray:
    """x_t = ΔClose_t / max(scale_t, floor). NaN where scale is unavailable."""
    if floor <= 0:
        raise FeatureError("scale_floor must be > 0.")
    denom = np.maximum(np.asarray(scale, dtype=np.float64), float(floor))
    x = np.asarray(changes, dtype=np.float64) / denom
    x = np.where(np.isfinite(scale), x, np.nan)
    return x


def create_windows(values: np.ndarray, lookback: int) -> np.ndarray:
    """Inclusive sliding windows. Shape ``(n - lookback + 1, lookback, 1)``.

    Rows that contain NaN are *kept* as NaN so the caller can align to
    calendar dates and drop them explicitly.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise FeatureError(f"Expected 1D or 2D array, got shape {array.shape}.")
    if lookback < 2:
        raise FeatureError("lookback must be >= 2.")
    n_rows = array.shape[0]
    if n_rows < lookback:
        raise FeatureError(
            f"Need at least {lookback} rows to build windows; received {n_rows}."
        )
    n_windows = n_rows - lookback + 1
    windows = np.stack([array[i : i + lookback] for i in range(n_windows)], axis=0)
    return windows.astype(np.float32, copy=False)


def window_end_indices(n_rows: int, lookback: int) -> np.ndarray:
    """Row index (into the Close series) of the last observation of each window."""
    if n_rows < lookback:
        return np.array([], dtype=np.int64)
    return np.arange(lookback - 1, n_rows, dtype=np.int64)


@dataclass
class FeatureFrame:
    """Aligned per-day features plus the 3D tensor the LSTM consumes."""

    dates: pd.DatetimeIndex
    close: np.ndarray
    price_change: np.ndarray
    local_scale: np.ndarray
    normalized_return: np.ndarray
    windows: np.ndarray
    end_index: np.ndarray
    spec: FeatureSpec

    @property
    def valid_mask(self) -> np.ndarray:
        """Windows with a finite, fully-observed lookback of x_t."""
        return np.isfinite(self.windows).all(axis=(1, 2))

    def valid_windows(self) -> np.ndarray:
        return self.windows[self.valid_mask]

    def valid_end_index(self) -> np.ndarray:
        return self.end_index[self.valid_mask]


def build_feature_frame(close: np.ndarray, dates: pd.Series | np.ndarray, spec: FeatureSpec) -> FeatureFrame:
    """Build the monitor input from a Close series (oldest first)."""
    close_arr = np.asarray(close, dtype=np.float64).reshape(-1)
    if np.isnan(close_arr).any():
        raise FeatureError("Close contains nulls; clean the series in the data loader.")
    parsed = pd.to_datetime(dates)
    if len(parsed) != close_arr.size:
        raise FeatureError("dates and Close must have the same length.")

    changes = price_changes(close_arr)
    scale = causal_rolling_mad(changes, spec.vol_lookback, mad_to_std=spec.mad_to_std)
    x = normalize_changes(changes, scale, floor=spec.scale_floor)
    windows = create_windows(x, spec.lookback)
    end_index = window_end_indices(close_arr.size, spec.lookback)
    return FeatureFrame(
        dates=pd.DatetimeIndex(parsed),
        close=close_arr,
        price_change=changes,
        local_scale=scale,
        normalized_return=x,
        windows=windows,
        end_index=end_index,
        spec=spec,
    )


def freeze_scale_floor(scale: np.ndarray, *, quantile: float = 0.05, min_floor: float = 0.05) -> float:
    """Calibration-time floor: a low quantile of finite causal scales.

    Frozen after ``train``. Weekly jobs must not recompute it.
    """
    finite = np.asarray(scale, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size < 16:
        return float(min_floor)
    return float(max(min_floor, np.quantile(finite, quantile)))
