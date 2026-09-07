---
license: mit
library_name: keras
tags:
  - time-series
  - anomaly-detection
  - lstm
  - autoencoder
  - wti
  - crude-oil
---

# WTI LSTM Autoencoder

Unsupervised anomaly detector for daily WTI futures (`CL=F`). An LSTM autoencoder reconstructs 10-day windows of RobustScaler-transformed close prices. Windows whose reconstruction MSE exceeds the 90th percentile of **training** MSE are flagged.

This card describes the **champion** stored in this Hub repo. Training and weekly fine-tuning live in the GitHub repository [DrAdrianDC/WTI-Prices-Anomaly-Detection](https://github.com/DrAdrianDC/WTI-Prices-Anomaly-Detection).

## Files

| File | Role |
| --- | --- |
| `lstm_autoencoder.keras` | Keras 3 weights + graph |
| `scaler.pkl` | Frozen `RobustScaler` + lookback metadata |
| `metadata.json` | Threshold, MAE, `last_date`, window counts |
| `reconstruction_scores.csv` | Per-day Close, MSE, flag |
| `detected_anomalies.csv` | Flagged days only |
| `plot-anomalies.png` | Price + anomaly markers |

## Contract

- Ticker: `CL=F` (unadjusted close, yfinance)
- Lookback: 10 trading days
- Input shape: `(batch, 10, 1)`
- Scaler: `RobustScaler`, fitted on the training split only; **never refit on fine-tune**
- Threshold: P90 of training reconstruction MSE
- Quality gate (weekly): challenger hold-out MAE must be ≤ champion MAE × 1.10

Replace the numbers below with the contents of `metadata.json` after each promoted retrain.

## Champion snapshot

- `last_date`: 2026-09-04
- Epochs trained (EarlyStopping): 34 / max 100
- Train MAE / hold-out MAE: 0.072 / 0.055
- Threshold: 0.0162
- Anomalies: 623 / 6,528 windows (9.5%)
- Strongest event: 2020-04-20, close −37.63 USD, MSE 0.270

## Use

```python
import joblib
from tensorflow import keras

model = keras.models.load_model("lstm_autoencoder.keras")
payload = joblib.load("scaler.pkl")
scaler = payload["scaler"] if isinstance(payload, dict) else payload
```

Build windows of shape `(n, 10, 1)` in the same scaled space, then compare per-window MSE to `metadata.json["threshold"]`.
