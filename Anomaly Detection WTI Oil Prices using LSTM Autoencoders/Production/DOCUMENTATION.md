# Documentation

Engineering notes for the WTI anomaly monitor. The [README](README.md) is the public surface.

## Contract

LSTM autoencoder on **locally vol-normalized 10-day ΔClose**. It answers: is this window unlike quiet 2010–2019 tape? It does not forecast and it does not classify crises.

- Calibrate on quiet windows from **2010-01-01** to **2019-12-31** (top 5% of local scale dropped from the loss).
- 2008 is scored, not trained. 2020–present is OOS.
- Freeze `scale_floor` (≈ $0.66), `quiet_vol_cutoff`, P99 threshold (**0.382**), and the two baselines.
- Do not update weights or the threshold on a cron.

`RollingVolBaseline` in `src/baseline.py` is the honest competitor at the same P99 budget.

## Features (`src/features.py`)

1. `ΔClose_t` in USD/bbl. No percent/log returns (20 April 2020).
2. `local_scale_t` = 1.4826 × MAD of `ΔClose_{t-60:t}` (**excludes today**). Look-ahead is tested.
3. `x_t = ΔClose_t / max(local_scale_t, scale_floor)`.
4. Inclusive windows of length 10.

Spearman(MSE, 10-day vol) was 0.72 on Close levels and is **0.33** here.

## `train`

1. Download `CL=F` (snapshot fallback if Yahoo is down).
2. Fit floor and quiet cutoff on the calibration frame.
3. Early stopping = last 15% of *quiet* calibration windows, not 2020.
4. Threshold = P99 of train MSE, written to `config.yaml`.
5. Fit P99 z-score and 10-day vol baselines on calibration Close.
6. Persist `feature_state.json` (not pickle). Score the full tape. LSTM ledger columns are then append-only.

`evaluate` appends new dates with frozen weights. `retrain` is manual: tune / early-stop / unseen gate on recent quiet windows; refuses if too few quiet windows; never moves the threshold. `walkforward` is expanding folds from 2010.

## Drift

| Signal | Measure | Action |
| --- | --- | --- |
| Data drift of `x_t` | PSI vs calibration (0.10 / 0.25) | Inspect the feed. Do not auto-retrain. |
| Data drift of `local_scale` | PSI | Vol regime. Expected; already divided out. |
| Concept drift | Median quiet-day MSE | Retrain only if quiet days, not a live episode, degraded. |

Champion last year: PSI(`x`) = 0.05 (stable), PSI(`local_scale`) = 2.88 (significant). Median quiet MSE 0.048 → 0.072. The last 252 days include March–April 2026; May–September 2026 flag rate is 0.

## Events

`EVENT_CATALOG` is pre-registered. Do not add dates after looking at scores. Metrics are episode-grained (`cluster_episodes`). Walk-forward uses `events_between(train_end, test_end)`.

OOS catalog (2020–2026): LSTM 9/10, vol 8/10. Miss for the LSTM: 17 June 2022 (\(M=-1.23\)). 2014–16 glut is inside calibration and is ordinary tape under this contract.

## Persistence

`lstm_autoencoder.keras`, `feature_state.json`, `metadata.json`, `drift_report.json`, `event_metrics.json`, `walkforward.csv`, `price_snapshot.csv` (research snapshot, not a redistribution license).

## CI

`.github/workflows/ci.yaml` — `pytest -q` on push/PR. No weekly job. No TensorFlow in the default suite.

## Do not

- Train on the full series.
- Lower the threshold to P90 to decorate a plot.
- Refit `scale_floor` or the threshold on a schedule.
- Add catalog events after seeing scores.
- Restore `scaler.pkl`.
