"""Scaling and sliding-window construction for the LSTM autoencoder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

ScalerName = Literal["robust", "minmax", "standard"]

_SCALER_FACTORY: dict[str, type[BaseEstimator]] = {
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
}


class PreprocessingError(ValueError):
    """Raised when sequences cannot be built from the supplied series."""


class TimeSeriesPreprocessor:
    """Fit a scaler independently and emit 3D LSTM windows.

    The research notebook built overlapping lookback windows of length 10
    on the Close series after a ``RobustScaler`` transform. This class
    keeps that contract and persists the scaler as ``scaler.pkl`` so
    weekly fine-tuning stays in the same feature space.

    Output shape
    ------------
    ``(n_samples, lookback, n_features)`` with ``n_features = 1``.
    """

    def __init__(
        self,
        lookback: int = 10,
        scaler_type: ScalerName | str = "robust",
        target_column: str = "Close",
        date_column: str = "Date",
    ) -> None:
        if lookback < 2:
            raise PreprocessingError("lookback must be >= 2.")
        scaler_key = str(scaler_type).lower()
        if scaler_key not in _SCALER_FACTORY:
            raise PreprocessingError(
                f"Unknown scaler_type='{scaler_type}'. "
                f"Expected one of {sorted(_SCALER_FACTORY)}."
            )
        self.lookback = lookback
        self.scaler_type = scaler_key
        self.target_column = target_column
        self.date_column = date_column
        self.scaler: BaseEstimator = _SCALER_FACTORY[scaler_key]()
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> TimeSeriesPreprocessor:
        """Fit the scaler on ``target_column`` only (no leakage from the future)."""
        values = self._extract_target(frame)
        self.scaler.fit(values)
        self._fitted = True
        logger.info(
            "Fitted %s on %s observations.",
            self.scaler.__class__.__name__,
            len(values),
        )
        return self

    def transform_series(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the scaled 2D Close array ``(n_rows, 1)``."""
        self._ensure_fitted()
        values = self._extract_target(frame)
        scaled = np.asarray(self.scaler.transform(values), dtype=np.float32)
        return scaled

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Scale Close prices and build continuous lookback windows."""
        scaled = self.transform_series(frame)
        return self.create_sequences(scaled, self.lookback)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Fit on ``frame`` and return its 3D sequence tensor."""
        return self.fit(frame).transform(frame)

    def chronological_split(
        self,
        frame: pd.DataFrame,
        *,
        train_end_date: str | None = None,
        validation_fraction: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a sorted price frame without shuffling.

        If ``train_end_date`` is provided the split matches the notebook
        pattern (``Date <= train_end_date``). Otherwise the last
        ``validation_fraction`` of rows is held out — the production default.
        """
        ordered = self._ordered_frame(frame)
        if train_end_date:
            cutoff = pd.Timestamp(train_end_date)
            train = ordered.loc[ordered[self.date_column] <= cutoff].copy()
            test = ordered.loc[ordered[self.date_column] > cutoff].copy()
        else:
            if not 0.0 < validation_fraction < 1.0:
                raise PreprocessingError("validation_fraction must be in (0, 1).")
            split_idx = int(len(ordered) * (1.0 - validation_fraction))
            if split_idx < self.lookback:
                raise PreprocessingError(
                    "Training split is shorter than lookback; "
                    "reduce validation_fraction or lookback."
                )
            train = ordered.iloc[:split_idx].copy()
            test = ordered.iloc[split_idx:].copy()

        if train.empty or test.empty:
            raise PreprocessingError(
                "Chronological split produced an empty train or validation set."
            )
        logger.info(
            "Split %s rows → train=%s (%s → %s), val=%s (%s → %s).",
            len(ordered),
            len(train),
            train[self.date_column].iloc[0].date(),
            train[self.date_column].iloc[-1].date(),
            len(test),
            test[self.date_column].iloc[0].date(),
            test[self.date_column].iloc[-1].date(),
        )
        return train, test

    def prepare_train_val(
        self,
        frame: pd.DataFrame,
        *,
        train_end_date: str | None = None,
        validation_fraction: float = 0.2,
        fit_scaler: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X_train, X_val)`` tensors ready for Keras.

        The scaler is fitted on the training split only unless
        ``fit_scaler`` is ``False`` (retrain / fine-tune path).
        """
        train, val = self.chronological_split(
            frame,
            train_end_date=train_end_date,
            validation_fraction=validation_fraction,
        )
        if fit_scaler:
            self.fit(train)
        else:
            self._ensure_fitted()
        x_train = self.transform(train)
        x_val = self.transform(val)
        return x_train, x_val

    def save_scaler(self, path: str | Path) -> Path:
        """Serialize the fitted scaler independently of the Keras model."""
        self._ensure_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaler": self.scaler,
            "scaler_type": self.scaler_type,
            "lookback": self.lookback,
            "target_column": self.target_column,
            "date_column": self.date_column,
        }
        joblib.dump(payload, destination)
        logger.info("Saved scaler to %s", destination)
        return destination

    @classmethod
    def load_scaler(cls, path: str | Path) -> TimeSeriesPreprocessor:
        """Restore a preprocessor from ``scaler.pkl``."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Scaler artifact not found: {source}")
        payload = joblib.load(source)
        if isinstance(payload, dict) and "scaler" in payload:
            instance = cls(
                lookback=int(payload.get("lookback", 10)),
                scaler_type=str(payload.get("scaler_type", "robust")),
                target_column=str(payload.get("target_column", "Close")),
                date_column=str(payload.get("date_column", "Date")),
            )
            instance.scaler = payload["scaler"]
        else:
            # Backward-compatible load of a raw sklearn scaler object.
            instance = cls()
            instance.scaler = payload
        instance._fitted = True
        logger.info("Loaded scaler from %s", source)
        return instance

    @staticmethod
    def create_sequences(values: np.ndarray, lookback: int) -> np.ndarray:
        """Build continuous sliding windows for an LSTM.

        Parameters
        ----------
        values:
            1D array ``(n,)`` or 2D array ``(n, n_features)``.
        lookback:
            Window length in trading days.

        Returns
        -------
        numpy.ndarray
            Float32 tensor of shape ``(n - lookback + 1, lookback, n_features)``.
        """
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise PreprocessingError(
                f"Expected a 1D or 2D array, received shape {array.shape}."
            )
        n_rows = array.shape[0]
        if n_rows < lookback:
            raise PreprocessingError(
                f"Need at least {lookback} rows to build windows; received {n_rows}."
            )
        # Inclusive sliding window so the last observation is used.
        n_windows = n_rows - lookback + 1
        windows = np.stack(
            [array[i : i + lookback] for i in range(n_windows)],
            axis=0,
        )
        return windows.astype(np.float32, copy=False)

    def _extract_target(self, frame: pd.DataFrame) -> np.ndarray:
        ordered = self._ordered_frame(frame)
        if self.target_column not in ordered.columns:
            raise PreprocessingError(
                f"Missing target column '{self.target_column}'."
            )
        series = pd.to_numeric(ordered[self.target_column], errors="coerce")
        if series.isna().any():
            raise PreprocessingError(
                f"Target column '{self.target_column}' contains non-numeric "
                "or null values. Clean the frame in the data loader first."
            )
        return series.to_numpy(dtype=np.float64).reshape(-1, 1)

    def _ordered_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.date_column in frame.columns:
            return frame.sort_values(self.date_column).reset_index(drop=True)
        return frame.reset_index(drop=True)

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise PreprocessingError(
                "Scaler has not been fitted. Call fit() or load_scaler() first."
            )
