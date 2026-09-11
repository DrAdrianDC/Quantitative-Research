"""Causal feature builder, calibration split, and JSON feature-state I/O.

Replaces the previous RobustScaler + pickle path. The LSTM now sees
locally vol-normalized ΔClose windows. The frozen state is six numbers
and a handful of baseline stats — not a pickle — so a Hub download cannot
execute code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baseline import RobustReturnZScore, RollingVolBaseline
from src.features import (
    FeatureFrame,
    FeatureSpec,
    build_feature_frame,
    freeze_scale_floor,
)

logger = logging.getLogger(__name__)


class PreprocessingError(ValueError):
    """Raised when sequences cannot be built from the supplied series."""


class TimeSeriesPreprocessor:
    """Fit-once feature contract for the anomaly monitor."""

    def __init__(
        self,
        lookback: int = 10,
        vol_lookback: int = 60,
        min_scale_floor: float = 0.05,
        scale_floor_quantile: float = 0.05,
        max_train_vol_percentile: float = 95.0,
        target_column: str = "Close",
        date_column: str = "Date",
        scale_floor: float | None = None,
    ) -> None:
        if lookback < 2:
            raise PreprocessingError("lookback must be >= 2.")
        self.lookback = int(lookback)
        self.vol_lookback = int(vol_lookback)
        self.min_scale_floor = float(min_scale_floor)
        self.scale_floor_quantile = float(scale_floor_quantile)
        self.max_train_vol_percentile = float(max_train_vol_percentile)
        self.target_column = target_column
        self.date_column = date_column
        self.scale_floor = float(scale_floor) if scale_floor is not None else None
        self.quiet_vol_cutoff_: float | None = None
        self.return_zscore: RobustReturnZScore | None = None
        self.vol_baseline: RollingVolBaseline | None = None
        self._fitted = scale_floor is not None

    @property
    def spec(self) -> FeatureSpec:
        floor = self.scale_floor if self.scale_floor is not None else self.min_scale_floor
        return FeatureSpec(
            lookback=self.lookback,
            vol_lookback=self.vol_lookback,
            scale_floor=float(floor),
            date_column=self.date_column,
            target_column=self.target_column,
        )

    def _ordered_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.date_column not in frame.columns or self.target_column not in frame.columns:
            raise PreprocessingError(
                f"Frame must contain '{self.date_column}' and '{self.target_column}'."
            )
        ordered = frame.copy()
        ordered[self.date_column] = pd.to_datetime(ordered[self.date_column])
        return ordered.sort_values(self.date_column).reset_index(drop=True)

    def build_frame(self, frame: pd.DataFrame) -> FeatureFrame:
        ordered = self._ordered_frame(frame)
        close = pd.to_numeric(ordered[self.target_column], errors="coerce")
        if close.isna().any():
            raise PreprocessingError("Close contains non-numeric or null values.")
        return build_feature_frame(close.to_numpy(dtype=np.float64), ordered[self.date_column], self.spec)

    def fit(self, frame: pd.DataFrame) -> TimeSeriesPreprocessor:
        """Freeze the scale floor and quiet-vol cutoff on the calibration frame."""
        # First pass with the minimum floor so we can measure the distribution.
        probe = TimeSeriesPreprocessor(
            lookback=self.lookback,
            vol_lookback=self.vol_lookback,
            min_scale_floor=self.min_scale_floor,
            scale_floor_quantile=self.scale_floor_quantile,
            max_train_vol_percentile=self.max_train_vol_percentile,
            target_column=self.target_column,
            date_column=self.date_column,
            scale_floor=self.min_scale_floor,
        )
        features = probe.build_frame(frame)
        self.scale_floor = freeze_scale_floor(
            features.local_scale,
            quantile=self.scale_floor_quantile,
            min_floor=self.min_scale_floor,
        )
        features = self.build_frame(frame)
        valid = features.valid_mask
        scales = features.local_scale[features.end_index][valid]
        if scales.size < 16:
            raise PreprocessingError("Not enough valid windows to freeze the quiet-vol cutoff.")
        self.quiet_vol_cutoff_ = float(np.percentile(scales, self.max_train_vol_percentile))
        self._fitted = True
        logger.info(
            "Frozen scale_floor=%.4f quiet_vol_cutoff=%.4f (P%s of calibration local_scale).",
            self.scale_floor,
            self.quiet_vol_cutoff_,
            self.max_train_vol_percentile,
        )
        return self

    def quiet_mask(self, features: FeatureFrame) -> np.ndarray:
        """True on valid windows whose end-of-window local scale is below cutoff."""
        if self.quiet_vol_cutoff_ is None:
            raise PreprocessingError("quiet_vol_cutoff_ is not set. Call fit() first.")
        end_scale = features.local_scale[features.end_index]
        return features.valid_mask & (end_scale <= self.quiet_vol_cutoff_)

    def chronological_split(
        self,
        frame: pd.DataFrame,
        *,
        train_end_date: str | None,
        validation_fraction: float,
        train_start_date: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered = self._ordered_frame(frame)
        if train_end_date:
            cutoff = pd.Timestamp(train_end_date)
            train = ordered.loc[ordered[self.date_column] <= cutoff].copy()
            test = ordered.loc[ordered[self.date_column] > cutoff].copy()
        else:
            if not 0.0 < validation_fraction < 1.0:
                raise PreprocessingError("validation_fraction must be in (0, 1).")
            split_idx = int(len(ordered) * (1.0 - validation_fraction))
            train = ordered.iloc[:split_idx].copy()
            test = ordered.iloc[split_idx:].copy()
        if train_start_date:
            start = pd.Timestamp(train_start_date)
            train = train.loc[train[self.date_column] >= start].copy()
        if train.empty:
            raise PreprocessingError("Chronological split produced an empty train set.")
        logger.info(
            "Split %s rows → train=%s (%s → %s), tail=%s (%s → %s).",
            len(ordered),
            len(train),
            train[self.date_column].iloc[0].date() if len(train) else None,
            train[self.date_column].iloc[-1].date() if len(train) else None,
            len(test),
            test[self.date_column].iloc[0].date() if len(test) else None,
            test[self.date_column].iloc[-1].date() if len(test) else None,
        )
        return train, test

    def prepare_calibration(
        self,
        frame: pd.DataFrame,
        *,
        train_end_date: str | None,
        validation_fraction: float,
        fit_state: bool,
        train_start_date: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Quiet calibration windows for train / early-stopping.

        Early stopping uses the last ``validation_fraction`` of *quiet*
        calibration windows (still inside ``train_end_date``). Dates after
        ``train_end_date`` are not a val set: stopping on a crisis restores
        weights that reconstruct crises, which is the opposite of a frozen
        monitor.
        """
        calibration, _oos = self.chronological_split(
            frame,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            validation_fraction=validation_fraction,
        )
        if fit_state:
            self.fit(calibration)
        self._ensure_fitted()
        features = self.build_frame(calibration)
        quiet = self.quiet_mask(features)
        windows = features.windows[quiet]
        if windows.shape[0] < 32:
            raise PreprocessingError(
                f"Only {windows.shape[0]} quiet calibration windows; "
                "relax max_train_vol_percentile or extend train_end_date."
            )
        split_idx = int(len(windows) * (1.0 - validation_fraction))
        if split_idx < 16 or (len(windows) - split_idx) < 8:
            raise PreprocessingError("Quiet calibration split is too small.")
        x_train = windows[:split_idx]
        x_val = windows[split_idx:]
        meta = {
            "n_calibration_rows": int(len(calibration)),
            "n_valid_windows": int(features.valid_mask.sum()),
            "n_quiet_windows": int(quiet.sum()),
            "n_train_windows": int(x_train.shape[0]),
            "n_earlystop_windows": int(x_val.shape[0]),
            "scale_floor": float(self.scale_floor),
            "quiet_vol_cutoff": float(self.quiet_vol_cutoff_),
            "train_start_date": str(calibration[self.date_column].iloc[0].date()),
            "train_end_date": str(calibration[self.date_column].iloc[-1].date()),
        }
        logger.info(
            "Calibration: %s quiet windows → train=%s earlystop=%s (dropped %s loud windows).",
            int(quiet.sum()),
            x_train.shape[0],
            x_val.shape[0],
            int(features.valid_mask.sum() - quiet.sum()),
        )
        return x_train, x_val, meta

    def fit_baselines(self, frame: pd.DataFrame, percentile: float) -> None:
        """Fit the two classical detectors on the calibration Close series."""
        ordered = self._ordered_frame(frame)
        close = pd.to_numeric(ordered[self.target_column], errors="coerce").to_numpy(dtype=np.float64)
        self.return_zscore = RobustReturnZScore(percentile=percentile).fit(close)
        self.vol_baseline = RollingVolBaseline(lookback=self.lookback, percentile=percentile).fit(close)

    def three_way_split(
        self,
        windows: np.ndarray,
        *,
        tune_fraction: float,
        earlystop_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Chronological tune / early-stop / gate. Gate is never used in fit()."""
        n = int(windows.shape[0])
        n_tune = int(n * tune_fraction)
        n_es = int(n * earlystop_fraction)
        n_gate = n - n_tune - n_es
        if min(n_tune, n_es, n_gate) < 8:
            raise PreprocessingError(
                f"Three-way split too small (n={n}, tune={n_tune}, es={n_es}, gate={n_gate})."
            )
        x_tune = windows[:n_tune]
        x_es = windows[n_tune : n_tune + n_es]
        x_gate = windows[n_tune + n_es :]
        return x_tune, x_es, x_gate

    def save_state(self, path: str | Path) -> Path:
        self._ensure_fitted()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "lookback": self.lookback,
            "vol_lookback": self.vol_lookback,
            "min_scale_floor": self.min_scale_floor,
            "scale_floor_quantile": self.scale_floor_quantile,
            "max_train_vol_percentile": self.max_train_vol_percentile,
            "scale_floor": float(self.scale_floor),
            "quiet_vol_cutoff": None
            if self.quiet_vol_cutoff_ is None
            else float(self.quiet_vol_cutoff_),
            "target_column": self.target_column,
            "date_column": self.date_column,
            "feature": self.spec.to_dict(),
        }
        if self.return_zscore is not None and self.return_zscore.fitted:
            payload["return_zscore"] = self.return_zscore.to_dict()
        if self.vol_baseline is not None and self.vol_baseline.fitted:
            payload["vol_baseline"] = self.vol_baseline.to_dict()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote feature state to %s", destination)
        return destination

    @classmethod
    def load_state(cls, path: str | Path) -> TimeSeriesPreprocessor:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Feature state not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PreprocessingError(f"Feature state at {source} is not a JSON object.")
        instance = cls(
            lookback=int(payload.get("lookback", 10)),
            vol_lookback=int(payload.get("vol_lookback", 60)),
            min_scale_floor=float(payload.get("min_scale_floor", 0.05)),
            scale_floor_quantile=float(payload.get("scale_floor_quantile", 0.05)),
            max_train_vol_percentile=float(payload.get("max_train_vol_percentile", 95.0)),
            target_column=str(payload.get("target_column", "Close")),
            date_column=str(payload.get("date_column", "Date")),
            scale_floor=float(payload["scale_floor"]),
        )
        cutoff = payload.get("quiet_vol_cutoff")
        instance.quiet_vol_cutoff_ = None if cutoff is None else float(cutoff)
        instance._fitted = True
        if isinstance(payload.get("return_zscore"), dict):
            instance.return_zscore = RobustReturnZScore.from_dict(payload["return_zscore"])
        if isinstance(payload.get("vol_baseline"), dict):
            instance.vol_baseline = RollingVolBaseline.from_dict(payload["vol_baseline"])
        logger.info("Loaded feature state from %s", source)
        return instance

    # Backward-compatible aliases used by older call sites / tests.
    def save_scaler(self, path: str | Path) -> Path:
        return self.save_state(path)

    @classmethod
    def load_scaler(cls, path: str | Path) -> TimeSeriesPreprocessor:
        return cls.load_state(path)

    def _ensure_fitted(self) -> None:
        if not self._fitted or self.scale_floor is None:
            raise PreprocessingError("Feature state is not fitted. Call fit() or load_state() first.")
