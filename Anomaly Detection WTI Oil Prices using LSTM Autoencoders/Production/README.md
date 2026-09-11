> **Live repository:** [DrAdrianDC/WTI_Anomaly_Detection](https://github.com/DrAdrianDC/WTI_Anomaly_Detection). This `Production/` folder is a snapshot inside Quantitative-Research.

# WTI Crude Oil Price Anomaly Detection

Unsupervised LSTM autoencoder on daily WTI futures (`CL=F`). It reconstructs **10-day windows of locally vol-normalized ΔClose** (USD/bbl, not percent — the contract went negative). A day is flagged when reconstruction MSE exceeds the 99th percentile of *quiet* 2010–2019 error. That threshold is frozen. 2020–present is out of sample. 2008 is scored but was never in the loss.

The network never sees the WTI print. A causal 60-day MAD divides out local dollar scale before the LSTM, so $30 oil and $100 oil are comparable. Reconstructing raw Close made the score track 10-day volatility (Spearman 0.72). After this contract it is 0.33.

This is a shape-break monitor, not a crisis classifier and not a forecast. Slow grind selloffs (2016 glut, June 2022) look ordinary after local-vol normalization; the 20 April 2020 negative print does not. A 10-day rolling-vol rule and a one-day robust z-score run at the **same P99 budget** so you can see what the LSTM adds.

## Results

![WTI close with P99 flags](output_results/plot-anomalies.png)

![Reconstruction error vs frozen P99](output_results/plot-reconstruction-error.png)

Champion (quiet 2010–2019, tanh, dropout 0.1, best epoch 194 / 200). Scores through 10 September 2026.

| | |
| --- | --- |
| Train / early-stop windows | 1,974 / 349 quiet (121 loud dropped) |
| P99 threshold | 0.382, frozen |
| Flags, full tape | 368 / 6,470 (5.7%), 130 episodes |
| **OOS 2020–2026** | **174 / 1,683 (10.3%)**, 42 episodes |
| OOS catalog (10 pre-registered events) | LSTM **9 / 10**, rolling vol 8 / 10 |
| Spearman(MSE, 10-day vol) | 0.33 |
| 20 April 2020 | close −$37.63, \(M = 2.44\); peak ringing 28 April \(M = 2.78\) |
| After 21 April 2026 | May–September 2026: 0 LSTM flags |

P99 is of *quiet train* MSE, not of live days. The 10% OOS flag rate is the honest operating point; use \(M = \log_{10}(\mathrm{MSE}/\mathrm{threshold})\) to rank, not the 0/1.

| Date | Event | LSTM | Vol |
| --- | --- | --- | --- |
| 2020-03-09 | OPEC+ price war | hit | hit |
| 2020-03-18 | COVID demand collapse | hit | hit |
| 2020-04-20 | Negative settlement | hit | hit |
| 2020-04-21 | Post-negative bounce | hit | hit |
| 2021-11-26 | Omicron | hit | hit |
| 2022-03-08 | Russia–Ukraine spike | hit | hit |
| 2022-03-09 | Continuation | hit | hit |
| 2022-06-17 | Mid-year liquidation | **miss** | hit |
| 2023-04-03 | OPEC+ surprise cut | hit | miss |
| 2023-10-09 | Israel–Hamas spike | hit | miss |

June 2022 is \(M = -1.23\): a multi-week liquidation after local MAD has already risen. The 2014–16 glut sits inside calibration and is treated as ordinary tape. Both misses are the contract, not a silent bug. LSTM-only hits are the 2023 OPEC cut and Israel–Hamas.

Walk-forward (same recipe, expanding from 2010): `output_results/walkforward.csv`.

## Model

```
ΔClose_t / max(causal 60-day MAD, floor ≈ $0.66)   # floor frozen at train
  → windows (batch, 10, 1)
  → LSTM 64 + Dropout 0.1
  → LSTM 32 + Dropout 0.1          # bottleneck
  → RepeatVector(10)
  → LSTM 32 + Dropout 0.1
  → LSTM 64 + Dropout 0.1
  → TimeDistributed Dense(1)
```

Loss is MSE on quiet calibration windows. Early stopping uses a later quiet slice of 2010–2019, not 2020. Feature state is JSON (`feature_state.json`), not pickle. Weights and threshold stay frozen after `train`. Retrain is manual.

## Setup

Python 3.11 or 3.12. TensorFlow 2.16 wants `numpy<2`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python src/pipeline.py --mode train --skip-upload
python src/pipeline.py --mode evaluate
```

`--skip-walkforward` skips the extra folds. Knobs live in `config.yaml`. Modules: [DOCUMENTATION.md](DOCUMENTATION.md).

Prices: [yfinance](https://github.com/ranaroussi/yfinance) `CL=F`, unadjusted close. `output_results/price_snapshot.csv` is a research snapshot for offline review, not a redistribution license.

MIT. See [LICENSE](LICENSE).
