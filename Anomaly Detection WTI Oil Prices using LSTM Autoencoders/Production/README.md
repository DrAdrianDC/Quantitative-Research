# WTI Crude Oil Price Anomaly Detection

Production pipeline for unsupervised anomaly detection on daily WTI futures (`CL=F`).

An LSTM autoencoder learns to reconstruct 10-day windows of scaled close prices. Days whose reconstruction error exceeds a statistical threshold are flagged. The system is fully automated: data is pulled from Yahoo Finance, the model is trained or fine-tuned from the CLI, figures are written to `output_results/`, and a weekly GitHub Actions job can promote a new champion to the Hugging Face Hub only if a hold-out quality gate passes.

Abnormal WTI moves have a direct read-through to energy and financial risk. The model is unsupervised — it is never given event labels. It has to recover known stress regimes from reconstruction error alone: the 2008 collapse, the 2015–2016 oil bust, March 2022, and 20 April 2020, the first negative WTI settlement.

Module-level detail: [DOCUMENTATION.md](DOCUMENTATION.md).  
How to put this on GitHub, activate Actions, and seed Hugging Face: [PLAYBOOK.md](PLAYBOOK.md).

## Results

Close price with flagged anomalies:

![Anomaly Detection WTI Oil Price](output_results/plot-anomalies.png)

Reconstruction MSE versus the decision threshold:

![Reconstruction Error Over Time](output_results/plot-reconstruction-error.png)

Champion metrics (max 100 epochs, EarlyStopping at 34, scored through 2026-09-04):

| Metric | Value |
| --- | --- |
| Scored windows | 6,528 |
| Anomalies (P90 MSE) | 623 (9.5%) |
| Train MAE / hold-out MAE | 0.072 / 0.055 |
| Best validation loss | 0.0060 |
| Threshold | 0.0162 |
| Strongest event | 2020-04-20, close −$37.63, MSE 0.270 |

The top reconstruction errors fall on 20–30 April 2020. The model also clusters 2008 (101 flags), 2015–2016, and March 2022. Hold-out MAE is below train MAE, which is the signature of a model that generalizes rather than memorizes.

## Architecture

Daily unadjusted closes are downloaded with yfinance, scaled with `RobustScaler` (robust to fat tails and the 2020 negative print), and cut into continuous windows of shape `(n, 10, 1)`.

```
Input (batch, 10, 1)
  → LSTM 64 + Dropout 0.25
  → LSTM 32 + Dropout 0.25          # latent bottleneck
  → RepeatVector(10)
  → LSTM 32 + Dropout 0.25
  → LSTM 64 + Dropout 0.25
  → TimeDistributed Dense(1)
```

Training objective is MSE reconstruction. The anomaly score is per-window MSE. The threshold is the 90th percentile of training MSE. The scaler is fitted on the training split only and is never refit during weekly fine-tuning, so the feature space stays frozen.

## Project structure

```
config.yaml                      runtime knobs (ticker, lookback, epochs, Hub)
requirements.txt
DOCUMENTATION.md                 workflow and per-file reference
src/
  config.py                      typed loader for config.yaml
  data_loader.py                 yfinance download and validation
  preprocessor.py                scaler + 3D lookback windows
  model.py                       LSTM autoencoder (.keras)
  plots.py                       PNG / HTML figures
  pipeline.py                    train | retrain | evaluate
output_results/                  champion model, scores, plots
.github/workflows/retrain.yaml   Sunday 00:00 UTC fine-tune
```

## Setup

Python 3.11+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Known-good stack: NumPy 1.26, SciPy 1.14, TensorFlow 2.16.

## Usage

Train a champion on the full `CL=F` history and write model, CSVs and plots. Skip the Hub upload on a local machine:

```bash
python src/pipeline.py --mode train --skip-upload
```

Score the saved champion and refresh reports without training:

```bash
python src/pipeline.py --mode evaluate
```

Fine-tune on the latest window. The current champion is scored on a recent hold-out, a challenger is trained for a few epochs, and the challenger is promoted only if hold-out MAE does not degrade by more than 10%:

```bash
python src/pipeline.py --mode retrain
```

If no local model exists, `retrain` bootstraps a full `train` run.

## Configuration

All runtime parameters live in `config.yaml`: ticker (`CL=F`), lookback (10), scaler (`robust`), batch size, train vs retrain epochs, threshold percentile, Hugging Face `repo_id`, and every filename under `output_results/`.

Set your Hub repository before enabling uploads:

```yaml
huggingface:
  repo_id: "your-user/wti-lstm-autoencoder"
```

## MLOps

`.github/workflows/retrain.yaml` runs every Sunday at 00:00 UTC and on `workflow_dispatch`. It installs dependencies and runs:

```bash
python src/pipeline.py --mode retrain
```

Retrain is **weekly** (Sunday 00:00 UTC), not daily: WTI adds one trading bar per session, so a daily fine-tune is noise. Manual runs use `workflow_dispatch` after the Hub is seeded (see the playbook).

Configure the `HF_TOKEN` repository secret to publish. If the secret is missing, retrain still completes and the job stays green; only the Hub upload is skipped. A rejected challenger leaves the champion untouched.

## Data

Prices come from [yfinance](https://github.com/ranaroussi/yfinance): ticker `CL=F`, period `max`, unadjusted close. The loader retries transient Yahoo failures and rejects frames whose Close null ratio exceeds the configured cap.

## License

MIT. See [LICENSE](LICENSE).
