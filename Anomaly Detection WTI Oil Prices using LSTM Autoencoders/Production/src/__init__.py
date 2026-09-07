"""Production package for WTI crude-oil price anomaly detection.

Public modules
--------------
``src.config``
    Typed loader for ``config.yaml``.
``src.data_loader``
    yfinance download and integrity checks.
``src.preprocessor``
    Scaler + continuous lookback windows.
``src.model``
    LSTM autoencoder definition and persistence (``.keras``).
``src.plots``
    Portfolio figures (price series, anomalies, reconstruction error).
``src.pipeline``
    CLI orchestrator (``--mode train|retrain|evaluate``).
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
