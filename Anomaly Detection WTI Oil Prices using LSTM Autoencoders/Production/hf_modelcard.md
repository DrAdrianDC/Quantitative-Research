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
  - monitoring
---

# WTI LSTM autoencoder

Unsupervised reconstruction-error model for daily WTI futures (`CL=F`). The network reconstructs **10-day windows of causally vol-normalized ΔClose** (USD/bbl). A window is flagged when its reconstruction MSE exceeds the 99th percentile of **quiet calibration** MSE (2010–2019). That threshold is frozen at **0.382**. 2008 is a historical holdout (not in the loss). 2020–present is out of sample.

This is a shape-break score, not a crisis classifier and not a forecast. A rolling 10-day vol rule and a one-day robust z-score are published next to it at the same P99 budget. Training and scoring are in this `Production/` folder (`src/pipeline.py`).

## Files

| File | Role |
| --- | --- |
| `lstm_autoencoder.keras` | Keras 3 weights + graph |
| `feature_state.json` | Frozen floor, lookbacks, P99 baselines. JSON, not pickle |
| `metadata.json` | Threshold, MAE, evaluation |
| `reconstruction_scores.csv` | Per-day scores + flags |
| `detected_anomalies.csv` | LSTM-flagged days |
| `plot-anomalies.png` | Price + flags |
| `plot-reconstruction-error.png` | Log MSE vs P99 |

## Contract

- Ticker: `CL=F`, unadjusted close
- Input: `(batch, 10, 1)` of \( x_t = \Delta\mathrm{Close}_t / \max(\mathrm{causal\ 60d\ MAD},\ 0.66) \)
- Percent/log returns are not used (negative print, 20 April 2020)
- Threshold: P99 of quiet 2010–2019 train MSE = **0.382**, frozen
- Weights are not updated on a schedule

## Champion snapshot

- `last_date`: 2026-09-10
- Calibration: 1,974 quiet train + 349 early-stop; 121 loud windows dropped
- Train MAE 0.187; early-stop MAE 0.287; best epoch 194 / 200
- Flags: 368 / 6,470 (5.7%); **OOS 2020–2026: 174 / 1,683 (10.3%)**
- OOS catalog (10 pre-registered events): LSTM **9 / 10**, rolling vol 8 / 10
- Spearman(MSE, 10-day vol): 0.33
- 20 April 2020: close −37.63 USD, \(M=2.44\); peak ringing 28 April \(M=2.78\)
- June 2022 liquidation: miss (\(M=-1.23\)). Vol baseline hits. Grind after local MAD has risen
- May–September 2026: 0 LSTM flags after the March–April episode

P99 is of quiet *train* MSE, not of live days. Rank \(M=\log_{10}(\mathrm{MSE}/\mathrm{threshold})\); do not treat the 0/1 as a 1% live rate.

## Use

```python
import json
from huggingface_hub import hf_hub_download
from tensorflow import keras

repo_id = "DrAdrianDC/wti-lstm-autoencoder"
model = keras.models.load_model(hf_hub_download(repo_id, "lstm_autoencoder.keras"))
state = json.loads(open(hf_hub_download(repo_id, "feature_state.json")).read())
threshold = json.load(open(hf_hub_download(repo_id, "metadata.json")))["threshold"]

# Build windows with the GitHub package (`src.features.build_feature_frame`)
# so the causal MAD matches training. Do not use a global scaler.
```

There is no Transformers `pipeline()` for this checkpoint.

## Limitations

- Not a forecast. The window has already ended.
- Not a crisis classifier. 2014–16 glut (in calibration) and June 2022 do not flag.
- Univariate unadjusted close only.
- Changing lookback, vol lookback, or activation invalidates the champion; run `train`.

## Intended use

Research and portfolio demonstration. Not for automated trading. If the flag fires on a live print, check CME before the model.
