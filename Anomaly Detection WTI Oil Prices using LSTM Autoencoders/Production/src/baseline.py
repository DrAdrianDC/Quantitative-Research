"""Classical baselines at the same rarity budget as the LSTM flag.

The LSTM is only interesting if it catches events a rolling-vol rule
misses, or misses events the rule catches for a reason we can name.

Two baselines share the LSTM's percentile language (default P99 of the
*calibration* sample, then frozen):

* ``RobustReturnZScore`` — was *today's* ΔClose large vs the historical
  MAD of ΔClose? One-day jump detector.
* ``RollingVolBaseline`` — was the last 10 days of ΔClose loud vs the
  calibration distribution of 10-day realized vol? Honest competitor
  for a windowed autoencoder.

Percent/log returns are not used. WTI settled at −$37.63 on 20 April 2020.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.features import MAD_TO_STD, price_changes

logger = logging.getLogger(__name__)


class BaselineError(ValueError):
    """Raised when a baseline cannot be fitted or applied."""


class RobustReturnZScore:
    """Frozen robust z-score of first differences of Close (USD/bbl)."""

    def __init__(self, percentile: float = 99.0) -> None:
        if not 0.0 < percentile < 100.0:
            raise BaselineError("percentile must be in (0, 100).")
        self.percentile = float(percentile)
        self.median_: float | None = None
        self.mad_: float | None = None
        self.scale_: float | None = None
        self.threshold_: float | None = None

    @property
    def fitted(self) -> bool:
        return self.threshold_ is not None

    def fit(self, close: np.ndarray) -> RobustReturnZScore:
        deltas = price_changes(close)
        finite = deltas[np.isfinite(deltas)]
        if finite.size < 8:
            raise BaselineError(
                f"Need at least 8 price changes to fit the z-score; got {finite.size}."
            )
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        scale = MAD_TO_STD * mad if mad > 0.0 else 1.0
        z = (finite - median) / scale
        self.median_ = median
        self.mad_ = mad
        self.scale_ = scale
        self.threshold_ = float(np.percentile(np.abs(z), self.percentile))
        logger.info(
            "Fitted ΔClose z-score: median=%.4f MAD=%.4f P%s(|z|)=%.3f.",
            median,
            mad,
            self.percentile,
            self.threshold_,
        )
        return self

    def transform(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_fitted()
        close_arr = np.asarray(close, dtype=np.float64).reshape(-1)
        n = close_arr.size
        change = price_changes(close_arr)
        zscore = np.full(n, np.nan, dtype=np.float64)
        flags = np.zeros(n, dtype=np.int32)
        finite = np.isfinite(change)
        zscore[finite] = (change[finite] - self.median_) / self.scale_
        flags[finite] = (np.abs(zscore[finite]) > self.threshold_).astype(np.int32)
        return change, zscore, flags

    def to_dict(self) -> dict[str, float]:
        self._ensure_fitted()
        return {
            "percentile": self.percentile,
            "median": float(self.median_),
            "mad": float(self.mad_),
            "scale": float(self.scale_),
            "threshold": float(self.threshold_),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RobustReturnZScore:
        instance = cls(percentile=float(payload.get("percentile", 99.0)))
        instance.median_ = float(payload["median"])
        instance.mad_ = float(payload["mad"])
        instance.scale_ = float(payload["scale"])
        instance.threshold_ = float(payload["threshold"])
        return instance

    def _ensure_fitted(self) -> None:
        if not self.fitted:
            raise BaselineError("Z-score is not fitted. Call fit() or from_dict() first.")


class RollingVolBaseline:
    """P-percentile of trailing realized vol of ΔClose."""

    def __init__(self, lookback: int = 10, percentile: float = 99.0) -> None:
        if lookback < 2:
            raise BaselineError("lookback must be >= 2.")
        if not 0.0 < percentile < 100.0:
            raise BaselineError("percentile must be in (0, 100).")
        self.lookback = int(lookback)
        self.percentile = float(percentile)
        self.threshold_: float | None = None

    @property
    def fitted(self) -> bool:
        return self.threshold_ is not None

    def realized_vol(self, close: np.ndarray) -> np.ndarray:
        changes = price_changes(close)
        vol = np.full(changes.size, np.nan, dtype=np.float64)
        for t in range(self.lookback, changes.size):
            window = changes[t - self.lookback + 1 : t + 1]
            if np.isnan(window).any():
                continue
            vol[t] = float(np.std(window, ddof=1)) if window.size > 1 else 0.0
        return vol

    def fit(self, close: np.ndarray) -> RollingVolBaseline:
        vol = self.realized_vol(close)
        finite = vol[np.isfinite(vol)]
        if finite.size < 8:
            raise BaselineError("Not enough windows to fit the rolling-vol baseline.")
        self.threshold_ = float(np.percentile(finite, self.percentile))
        logger.info(
            "Fitted rolling-vol baseline: lookback=%s P%s=%.4f.",
            self.lookback,
            self.percentile,
            self.threshold_,
        )
        return self

    def transform(self, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_fitted()
        vol = self.realized_vol(close)
        flags = np.zeros(vol.size, dtype=np.int32)
        finite = np.isfinite(vol)
        flags[finite] = (vol[finite] > self.threshold_).astype(np.int32)
        return vol, flags

    def to_dict(self) -> dict[str, float]:
        self._ensure_fitted()
        return {
            "lookback": float(self.lookback),
            "percentile": self.percentile,
            "threshold": float(self.threshold_),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RollingVolBaseline:
        instance = cls(
            lookback=int(payload.get("lookback", 10)),
            percentile=float(payload.get("percentile", 99.0)),
        )
        instance.threshold_ = float(payload["threshold"])
        return instance

    def _ensure_fitted(self) -> None:
        if not self.fitted:
            raise BaselineError("Rolling-vol baseline is not fitted.")


def attach_baselines(
    scores: pd.DataFrame,
    *,
    close: np.ndarray,
    dates: pd.Series,
    zscore: RobustReturnZScore | None,
    vol: RollingVolBaseline | None,
) -> pd.DataFrame:
    """Join frozen baseline columns onto a Date-indexed score frame.

    Does not rewrite ``reconstruction_mse`` or ``anomaly``.
    """
    out = scores.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    parsed = pd.to_datetime(pd.Index(dates))
    table = pd.DataFrame({"Date": parsed.strftime("%Y-%m-%d")})
    if zscore is not None and zscore.fitted:
        change, z, z_flag = zscore.transform(close)
        table["price_change"] = change
        table["return_zscore"] = z
        table["zscore_anomaly"] = z_flag
    if vol is not None and vol.fitted:
        rolling, v_flag = vol.transform(close)
        table["rolling_vol"] = rolling
        table["vol_anomaly"] = v_flag
    drop = [c for c in ("price_change", "return_zscore", "zscore_anomaly", "rolling_vol", "vol_anomaly") if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    merged = out.merge(table, on="Date", how="left")
    for flag_col in ("zscore_anomaly", "vol_anomaly"):
        if flag_col in merged.columns:
            merged[flag_col] = merged[flag_col].fillna(0).astype(int)
    return merged


def comparison_summary(scores: pd.DataFrame) -> dict[str, Any]:
    """Overlap between LSTM flags and the two classical detectors."""
    if "anomaly" not in scores.columns:
        return {}
    lstm = scores["anomaly"].astype(int) == 1
    summary: dict[str, Any] = {"n_lstm_flags": int(lstm.sum())}
    if "zscore_anomaly" in scores.columns:
        zflag = scores["zscore_anomaly"].astype(int) == 1
        summary.update(
            {
                "n_zscore_flags": int(zflag.sum()),
                "n_lstm_and_zscore": int((lstm & zflag).sum()),
                "n_lstm_only_vs_zscore": int((lstm & ~zflag).sum()),
                "n_zscore_only": int((~lstm & zflag).sum()),
            }
        )
        abs_z = scores["return_zscore"].abs() if "return_zscore" in scores.columns else None
        if abs_z is not None and abs_z.notna().any():
            peak_i = abs_z.idxmax()
            summary["peak_abs_z_date"] = str(scores.loc[peak_i, "Date"])
            summary["peak_abs_z"] = float(abs_z.loc[peak_i])
    if "vol_anomaly" in scores.columns:
        vflag = scores["vol_anomaly"].astype(int) == 1
        union = (lstm | vflag).sum()
        summary.update(
            {
                "n_vol_flags": int(vflag.sum()),
                "n_lstm_and_vol": int((lstm & vflag).sum()),
                "n_lstm_only_vs_vol": int((lstm & ~vflag).sum()),
                "n_vol_only": int((~lstm & vflag).sum()),
                "jaccard_lstm_vol": float((lstm & vflag).sum() / union) if union else 0.0,
            }
        )
    return summary
