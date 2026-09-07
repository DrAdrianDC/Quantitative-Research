# Documentation

This document is the engineering reference for the WTI anomaly-detection pipeline: how a run moves through the system, what each file is responsible for, and which contracts must not be broken.

The [README](README.md) is the portfolio surface (problem, results, quick start). This file is what a teammate should read before changing code.

## System overview

```
                    config.yaml
                         │
                         ▼
                   src/config.py
                         │
                         ▼
                  src/pipeline.py
                   │     │     │
           train   │     │     │  evaluate
                   │  retrain  │
                   ▼     ▼     ▼
            ┌──────────┴──────────┐
            │  src/data_loader.py │  yfinance → Date, Close
            └──────────┬──────────┘
                       ▼
            ┌──────────┴──────────┐
            │ src/preprocessor.py │  RobustScaler + (n, 10, 1)
            └──────────┬──────────┘
                       ▼
            ┌──────────┴──────────┐
            │    src/model.py     │  LSTM autoencoder
            └──────────┬──────────┘
                       ▼
            ┌──────────┴──────────┐
            │    src/plots.py     │  PNG + HTML
            └──────────┬──────────┘
                       ▼
                 output_results/
                       │
                       ▼
              Hugging Face Hub
           (only if gate passes)
```

Three CLI modes share the same modules. They differ only in whether weights are created, updated, or left alone.

| Mode | Weights | Scaler | Data window | Gate | Writes plots |
| --- | --- | --- | --- | --- | --- |
| `train` | fit from scratch | fit on train split | full history | none | yes |
| `retrain` | fine-tune champion | **load, never refit** | last date − 365 days | hold-out MAE | yes, if promoted |
| `evaluate` | load only | load only | full history | none | yes |

Entry point:

```bash
python src/pipeline.py --mode {train|retrain|evaluate} [--skip-upload] [--epochs N] [--config PATH]
```

`src/pipeline.py` inserts the repo root on `sys.path`, so this command works from the project root without installing the package.

---

## End-to-end workflows

### 1. `train` — new champion

Use this for the first run, or whenever the architecture or scaler type changes.

1. Load and validate `config.yaml`.
2. Pin NumPy / TensorFlow seeds.
3. Download `CL=F` (`period=max` unless `start_date` is set). Validate ticker, dates, Close nulls.
4. Chronological split: last `validation_fraction` (default 20%) is hold-out. No shuffle.
5. Fit `RobustScaler` on the training split only. Build 3D windows `(n, lookback, 1)`.
6. Build the Sequential LSTM autoencoder and compile (Adam + `clipnorm`, MSE).
7. Fit with `EarlyStopping(monitor=val_loss, patience=10, restore_best_weights=True)`. Cap is 100 epochs; the current champion stopped at 34.
8. Compute train / hold-out MAE. Fit the anomaly threshold as P90 of **training** reconstruction MSE.
9. Score the full history. Align each window to its last trading day.
10. Write `output_results/`: `.keras`, `scaler.pkl`, `metadata.json`, CSVs, PNG, HTML, `training_history.csv`.
11. Upload to the Hub unless `--skip-upload` or `HF_TOKEN` is missing.

### 2. `retrain` — weekly fine-tune with a quality gate

This is the path GitHub Actions runs every Sunday.

1. If the local `.keras` is missing, try to pull champion artifacts from the Hub. If that also fails, fall back to `train`.
2. Refuse to continue if the model exists but `scaler.pkl` does not. Fine-tuning in a different feature space is a silent skew.
3. Read `last_date` from `metadata.json`. Download from `last_date − context_days` (365) so LSTM windows remain continuous.
4. Transform with the **existing** scaler. Split the recent frame chronologically.
5. Score the current champion on the hold-out → `baseline_mae`.
6. Fine-tune a few epochs (default 8, patience 3) on the tune split, validating on the same hold-out.
7. Score the challenger on that hold-out → `challenger_mae`.
8. Promote only if  
   `challenger_mae <= baseline_mae * (1 + max_degradation_ratio)`  
   with `max_degradation_ratio = 0.10`.
9. On promote: overwrite `.keras` (scaler unchanged), recompute threshold, rewrite reports and plots, upload.
10. On reject: raise `QualityGateRejected`. Disk is not overwritten. Process exits 0 so CI does not treat a conservative keep as an outage.

The gate compares both models on the **same** recent hold-out. Comparing a new MAE to a stale MAE from a different period is not a gate.

### 3. `evaluate` — reports only

Loads the champion and scaler, downloads the latest full history, scores, and rewrites CSVs and plots. Weights are not saved. Use this after a config change that only affects reporting, or to refresh figures for the README.

---

## Repository map

```
.
├── README.md
├── DOCUMENTATION.md          ← this file
├── config.yaml
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model.py
│   ├── plots.py
│   └── pipeline.py
├── output_results/
└── .github/workflows/retrain.yaml
```

---

## File reference

### `config.yaml`

Single source of runtime truth. Training, retraining, evaluate and CI all read this file. Do not hard-code lookback, epochs or paths in Python.

| Section | What it controls |
| --- | --- |
| `data` | Ticker `CL=F`, yfinance period, optional ISO date bounds, chronological split, null-ratio cap, download retries |
| `preprocessing` | `scaler_type` (`robust` / `minmax` / `standard`), `lookback` (10), column names |
| `model` | LSTM widths, dropout, activation, loss, Adam `clipnorm`, `threshold_percentile` |
| `training` | Max epochs (100), batch size (32), patience (10), seed, shuffle (false) |
| `retrain` | Fine-tune epochs (8), patience (3), `context_days` (365), `max_degradation_ratio` (0.10), `min_sequences` |
| `output_results` | Directory and every output filename |
| `huggingface` | `repo_id`, `repo_type`, `private` |

`RobustScaler` is the default because WTI has heavy tails and a negative-price event. MinMax / Standard would let April 2020 dominate the scale.

### `src/config.py`

Typed YAML loader. Parses the file into frozen dataclasses (`DataConfig`, `PreprocessingConfig`, `ModelConfig`, `TrainingConfig`, `RetrainConfig`, `OutputConfig`, `HuggingFaceConfig`) wrapped by `AppConfig`.

- `load_config(path=None)` — defaults to `<repo>/config.yaml`. Relative paths resolve against the repo root.
- `AppConfig.ensure_output_dir()` — creates `output_results/` if needed.
- `ConfigError` — missing keys or invalid YAML.

Python code should take `AppConfig`, not a raw dict. That keeps a renamed filename from silently writing to the wrong path.

### `src/data_loader.py`

Download and integrity layer. Nothing downstream should call yfinance directly.

**Public API**

- `download_wti_prices(ticker, start=None, end=None, period="max", ...)` → `DataFrame["Date", "Close"]`
- `last_available_date(frame)` → last observation as `Timestamp`
- `DataValidationError` / `DataDownloadError`

**Guarantees**

- Ticker must match `^[A-Za-z0-9=.\-]{1,20}$`.
- Dates must be real ISO `YYYY-MM-DD`; `start <= end`.
- Handles both flat and MultiIndex yfinance frames (`Close` / `Close, CL=F`).
- Coerces Close to numeric; drops nulls; rejects the frame if the null ratio exceeds `max_null_ratio`.
- Deduplicates on Date, sorts ascending, strips timezone.
- Retries transient Yahoo / network failures with exponential backoff.

### `src/preprocessor.py`

Feature contract for the LSTM.

**Class:** `TimeSeriesPreprocessor`

| Method | Role |
| --- | --- |
| `fit` / `transform` / `fit_transform` | Scale Close and emit 3D windows |
| `transform_series` | Scaled 2D array `(n, 1)` without windowing |
| `chronological_split` | Date cutoff **or** last `validation_fraction` |
| `prepare_train_val` | Split + optional scaler fit + both tensors |
| `save_scaler` / `load_scaler` | Independent `scaler.pkl` (joblib), includes lookback metadata |
| `create_sequences` | Inclusive sliding window: shape `(n − lookback + 1, lookback, 1)` |

The last observation **is** used (`n − lookback + 1` windows). Each anomaly flag is later aligned to the last timestamp of its window, so a row in `detected_anomalies.csv` is a calendar date, not a tensor index.

`fit_scaler=False` on the retrain path. The loaded scaler must stay frozen.

### `src/model.py`

LSTM autoencoder wrapper around Keras 3.

**Class:** `LSTMAutoencoder`

| Method | Role |
| --- | --- |
| `build` | Sequential model with a native `Input` layer (clean `.keras` serialization) |
| `fit` | Reconstructs `x` from `x`; EarlyStopping on `val_loss` |
| `reconstruct` | Forward pass |
| `reconstruction_errors` | Per-window MSE (threshold) or MAE (quality gate) |
| `mean_absolute_error` | Scalar MAE used by the retrain gate |
| `fit_threshold` / `detect_anomalies` | P90 MSE → 0/1 flags |
| `save` / `load` | Native `.keras` only; `.h5` is rejected |

`set_global_seeds(seed)` pins NumPy and TensorFlow.

Architecture (62,529 trainable parameters):

```
encoder_lstm_1 (64, relu, return_sequences=True)
encoder_dropout_1 (0.25)
encoder_lstm_2 (32, relu, return_sequences=False)   # bottleneck
encoder_dropout_2 (0.25)
latent_repeat (RepeatVector(lookback))
decoder_lstm_1 (32, relu, return_sequences=True)
decoder_dropout_1 (0.25)
decoder_lstm_2 (64, relu, return_sequences=True)
decoder_dropout_2 (0.25)
reconstruction (TimeDistributed Dense(1))
```

Adam uses `clipnorm=1.0` because ReLU LSTMs can explode. Time series are not shuffled.

### `src/plots.py`

Figure writer. Called after scores exist; it does not touch the model.

| Function | Output |
| --- | --- |
| `plot_price_series` | `plot.png` |
| `plot_anomalies` | `plot-anomalies.png` — price line + red markers |
| `plot_reconstruction_error` | `plot-reconstruction-error.png` — MSE + dashed threshold |
| `plot_training_loss` | `plot-training-loss.png` when history is available |
| `write_result_plots` | All of the above + Plotly HTML if Plotly is installed |

PNG is the source of truth (matplotlib, no kaleido). HTML is best-effort.

### `src/pipeline.py`

Orchestrator and CLI. This is the only file GitHub Actions should invoke.

**Responsibilities**

- Argument parsing (`--mode`, `--config`, `--epochs`, `--skip-upload`, `--verbose`).
- Logging (`wti.pipeline`).
- `run_train` / `run_retrain` / `run_evaluate`.
- `export_detection_report` — score the full series, write CSVs, call `write_result_plots`, enrich `metadata.json`.
- Hub pull / push (`HF_TOKEN`, non-interactive).
- Exit codes: `0` success or rejected gate, `1` operational failure, `2` bad config.

**Quality-gate types**

- `QualityGateRejected` — expected weekly outcome; champion stays.
- `PipelineError` — missing scaler, too few windows, corrupt metadata.

Hub uploads (when the token is set and the gate passes): `.keras`, `scaler.pkl`, `metadata.json`, both CSVs, `plot.png`, `plot-anomalies.png`, `plot-reconstruction-error.png`.

### `src/__init__.py`

Package marker and version (`1.0.0`). Intentionally does **not** import TensorFlow. Import `src.model` only when you need the network.

### `output_results/`

Review and runtime folder. Produced by `train`, `retrain` (on promote) and `evaluate`.

| File | Role |
| --- | --- |
| `lstm_autoencoder.keras` | Champion weights + graph |
| `scaler.pkl` | Frozen scaler + lookback metadata |
| `metadata.json` | Metrics, threshold, `last_date`, plot list |
| `training_history.csv` | Per-epoch `loss` / `val_loss` |
| `reconstruction_scores.csv` | Every scored day: Date, Close, MSE, flag |
| `detected_anomalies.csv` | Rows where `anomaly == 1` |
| `plot.png` | Price series |
| `plot-anomalies.png` | Price + anomaly markers (README hero figure) |
| `plot-reconstruction-error.png` | MSE vs threshold |
| `plot-training-loss.png` | Train / val loss |
| `plot-anomalies.html` | Interactive overlay |
| `plot-reconstruction-error.html` | Interactive MSE |

`.keras` and `.pkl` are gitignored (binaries). CSV, JSON and PNG are intended to be committed so the README renders on GitHub.

`metadata.json` fields used by the next `retrain`: `last_date`, `threshold`, `val_mae`.

### `.github/workflows/retrain.yaml`

```
cron: 0 0 * * 0          # Sunday 00:00 UTC
workflow_dispatch         # manual
```

Job on `ubuntu-latest`, Python 3.11, pip cache, 90-minute timeout:

1. Checkout
2. `pip install -r requirements.txt`
3. `python src/pipeline.py --mode retrain`
4. If `HF_TOKEN` is empty, log that upload was skipped and **leave the job green**

Environment: `HF_TOKEN` from Actions secrets, TensorFlow log level 2, Hub progress bars off. Concurrency group `wti-weekly-retrain` prevents overlapping Sunday runs from clobbering each other.

### `requirements.txt`

Pinned ranges that import cleanly together: `numpy>=1.26,<2.0`, `scipy>=1.11,<1.15`, `tensorflow>=2.16,<2.17`, plus pandas, yfinance, scikit-learn, PyYAML, huggingface_hub, joblib, matplotlib, plotly.

NumPy 2.x breaks this TensorFlow 2.16 wheel. Do not loosen that pin without re-testing the import.

### `.gitignore`

Ignores `.venv/`, `__pycache__/`, `.keras` / `.pkl` under `output_results/`, and `.h5`. Reports (CSV, JSON, PNG, HTML) stay visible.

---

## Data and scoring contract

- **Series:** unadjusted daily Close of `CL=F`.
- **Window:** 10 consecutive trading days (yfinance already drops weekends).
- **Alignment:** window `i` uses rows `[i, i+10)` and is assigned to the date of row `i+9`.
- **Threshold:** `percentile(train_mse, 90)`. Applied at inference to the full scored series.
- **Gate metric:** mean MAE on the hold-out windows, not MSE. MAE is the quantity compared across champion and challenger.

April 2020 (negative print), 2008, 2015–2016 and March 2022 are the qualitative acceptance set. If a new champion loses 20 April 2020 as the top error, treat that as a regression even if MAE improved.

---

## Operational notes

- Always run commands from the repository root.
- Local development should pass `--skip-upload` unless you intend to publish.
- Changing `lookback` or `scaler_type` invalidates the champion. Run `train`, not `retrain`.
- `retrain` with fewer than `min_sequences` (32) windows is a hard failure — there is not enough recent data to fine-tune or to gate.
- The Hub `repo_id` in `config.yaml` must be changed to your user or org before the first upload.
