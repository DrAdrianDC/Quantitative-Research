#!/usr/bin/env python3
"""Orchestrator for the WTI anomaly monitor.

``train``     calibrate on quiet windows in [train_start, train_end], freeze
              threshold + feature state, score the full tape, write reports.
``evaluate``  append new dates with the frozen model; publish a drift report.
              Weights are not updated on a schedule.
``retrain``   human-gated. Three-way split on recent *quiet* windows. Promotes
              only if gate MAE (never seen during fit) does not degrade.
              Threshold stays frozen. Refuses if the recent window is too loud
              to define a quiet calibration set.
``walkforward`` expanding-window OOS table.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, load_config  # noqa: E402
from src.data_loader import (  # noqa: E402
    DataDownloadError,
    DataValidationError,
    last_available_date,
    load_wti_prices,
    write_price_snapshot,
)
from src.events import EVENT_CATALOG  # noqa: E402
from src.ledger import (  # noqa: E402
    append_new_rows,
    assert_event_integrity,
    ledger_stats,
    load_ledger,
    write_ledger,
)
from src.model import LSTMAutoencoder, keras_model_to_dict, set_global_seeds  # noqa: E402
from src.plots import write_result_plots  # noqa: E402
from src.preprocessor import TimeSeriesPreprocessor  # noqa: E402
from src.validation import evaluation_payload, run_walkforward, score_prices  # noqa: E402

logger = logging.getLogger("wti.pipeline")

HF_TOKEN_ENV = "HF_TOKEN"


class PipelineError(RuntimeError):
    """Raised when a run cannot finish safely."""


class QualityGateRejected(RuntimeError):
    """Raised when the challenger fails the quiet-day MAE gate."""


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WTI LSTM anomaly monitor.")
    parser.add_argument(
        "--mode",
        choices=("train", "retrain", "evaluate", "walkforward"),
        required=True,
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-walkforward",
        action="store_true",
        help="Even if training.run_walkforward is true, skip the expanding-window table.",
    )
    return parser.parse_args(argv)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Corrupt JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"{path} must be a JSON object.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _decision_threshold(config: AppConfig, metadata: dict[str, Any]) -> float:
    if config.model.frozen_threshold is not None:
        return float(config.model.frozen_threshold)
    raw = metadata.get("threshold")
    if raw is None:
        raise PipelineError(
            "No frozen decision threshold. Run --mode train once, which writes "
            "model.frozen_threshold into config.yaml."
        )
    return float(raw)


def _write_frozen_threshold(config: AppConfig, threshold: float) -> None:
    """Persist the decision policy next to the weights. No manual copy-paste."""
    path = config.source_path
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s*frozen_threshold:\s*).*$", re.MULTILINE)
    replacement = rf"\g<1>{threshold:.16g}"
    if pattern.search(text):
        updated = pattern.sub(replacement, text, count=1)
    else:
        updated = text + f"\n  frozen_threshold: {threshold:.16g}\n"
    path.write_text(updated, encoding="utf-8")
    logger.info("Wrote model.frozen_threshold=%.8f into %s", threshold, path.name)


def _download(config: AppConfig, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    data_cfg = config.data
    frame = load_wti_prices(
        ticker=data_cfg.ticker,
        start=start if start is not None else data_cfg.start_date,
        end=end if end is not None else data_cfg.end_date,
        period=data_cfg.period,
        auto_adjust=data_cfg.auto_adjust,
        max_null_ratio=data_cfg.max_null_ratio,
        retries=data_cfg.download_retries,
        retry_backoff_seconds=data_cfg.retry_backoff_seconds,
        snapshot_path=data_cfg.snapshot_path,
    )
    write_price_snapshot(frame, data_cfg.snapshot_path)
    return frame


def _build_autoencoder(config: AppConfig) -> LSTMAutoencoder:
    return LSTMAutoencoder(
        lookback=config.preprocessing.lookback,
        n_features=1,
        encoder_units=config.model.encoder_units,
        decoder_units=config.model.decoder_units,
        dropout=config.model.dropout,
        activation=config.model.activation,
        loss=config.model.loss,
        optimizer=config.model.optimizer,
        clipnorm=config.model.clipnorm,
        threshold_percentile=config.model.threshold_percentile,
    )


def _new_preprocessor(config: AppConfig) -> TimeSeriesPreprocessor:
    return TimeSeriesPreprocessor(
        lookback=config.preprocessing.lookback,
        vol_lookback=config.preprocessing.vol_lookback,
        min_scale_floor=config.preprocessing.min_scale_floor,
        scale_floor_quantile=config.preprocessing.scale_floor_quantile,
        max_train_vol_percentile=config.preprocessing.max_train_vol_percentile,
        target_column=config.preprocessing.target_column,
        date_column=config.preprocessing.date_column,
    )


def _hf_token() -> str | None:
    token = os.environ.get(HF_TOKEN_ENV, "").strip()
    return token or None


def pull_artifacts_from_hub(config: AppConfig) -> bool:
    model_path = config.output.model_path
    if model_path.exists() and config.output.state_path.exists():
        return True
    token = _hf_token()
    if token is None:
        logger.warning("Local artifacts missing and %s is unset; cannot pull from the Hub.", HF_TOKEN_ENV)
        return model_path.exists()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PipelineError("huggingface_hub is required to restore artifacts from the Hub.") from exc
    config.ensure_output_dir()
    recovered = False
    for filename in (config.output.model_path.name, config.output.state_path.name, config.output.metadata_path.name):
        try:
            hf_hub_download(
                repo_id=config.huggingface.repo_id,
                repo_type=config.huggingface.repo_type,
                filename=filename,
                local_dir=str(config.output.directory),
                token=token,
            )
            recovered = True
            logger.info("Restored %s from Hugging Face.", filename)
        except Exception as exc:
            logger.warning("Could not download %s from the Hub: %s", filename, exc)
    return recovered and model_path.exists()


def push_artifacts_to_hub(config: AppConfig) -> None:
    token = _hf_token()
    if token is None:
        logger.warning("Skipping Hub upload: environment variable %s is not set.", HF_TOKEN_ENV)
        return
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise PipelineError("huggingface_hub is required to publish artifacts.") from exc

    api = HfApi(token=token)
    repo_id = config.huggingface.repo_id
    api.create_repo(
        repo_id=repo_id,
        repo_type=config.huggingface.repo_type,
        private=config.huggingface.private,
        exist_ok=True,
        token=token,
    )
    required = [
        config.output.model_path,
        config.output.state_path,
        config.output.metadata_path,
        config.output.scores_path,
        config.output.anomalies_path,
        config.output.anomalies_plot_path,
        config.output.price_plot_path,
        config.output.reconstruction_plot_path,
    ]
    optional = [
        config.output.zscore_plot_path,
        config.output.vol_plot_path,
        config.output.drift_plot_path,
        config.output.magnitude_plot_path,
        config.output.drift_path,
        config.output.events_path,
        config.output.walkforward_path,
        PROJECT_ROOT / "hf_modelcard.md",
    ]
    for artifact in required:
        if not artifact.exists():
            raise PipelineError(f"Cannot upload missing artifact: {artifact}")
        _upload_one(api, config, artifact, path_in_repo=artifact.name)
    for artifact in optional:
        if not artifact.exists():
            continue
        name = "README.md" if artifact.name == "hf_modelcard.md" else artifact.name
        _upload_one(api, config, artifact, path_in_repo=name)


def _upload_one(api: Any, config: AppConfig, artifact: Path, *, path_in_repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(artifact),
        path_in_repo=path_in_repo,
        repo_id=config.huggingface.repo_id,
        repo_type=config.huggingface.repo_type,
        token=_hf_token(),
    )
    logger.info("Uploaded %s → hf://%s/%s", artifact.name, config.huggingface.repo_id, path_in_repo)


def _write_reports(
    config: AppConfig,
    scores: pd.DataFrame,
    metadata: dict[str, Any],
    preprocessor: TimeSeriesPreprocessor,
    *,
    history: dict[str, Any] | None = None,
) -> pd.DataFrame:
    config.ensure_output_dir()
    ordered = write_ledger(config.output.scores_path, scores)
    detected = ordered.loc[ordered["anomaly"].astype(int) == 1].copy()
    detected.to_csv(config.output.anomalies_path, index=False)
    if history:
        pd.DataFrame(history).to_csv(config.output.history_path, index=False)

    payload = evaluation_payload(
        ordered,
        train_start_date=config.data.train_start_date,
        train_end_date=config.data.train_end_date,
        preprocessor=preprocessor,
        recent_days=config.drift.recent_days,
        psi_warn=config.drift.psi_warn,
        psi_alert=config.drift.psi_alert,
        quiet_mae_degradation=config.drift.quiet_mae_degradation,
        rolling_psi_window=config.drift.rolling_psi_window,
    )
    psi_frame = payload.pop("rolling_psi", pd.DataFrame())
    metadata.update(ledger_stats(ordered))
    metadata["evaluation"] = {
        key: value
        for key, value in payload.items()
        if key not in {"rolling_psi"}
    }
    _write_json(config.output.events_path, {
        "oos": payload.get("oos"),
        "calibration": payload.get("calibration"),
        "historical": payload.get("historical"),
        "honest_baseline": payload.get("honest_baseline"),
        "catalog": [
            {"date": e.date, "name": e.name, "era": e.era} for e in EVENT_CATALOG
        ],
    })
    drift_out = {
        "data_drift": payload.get("data_drift"),
        "concept_drift": payload.get("concept_drift"),
        "baselines": payload.get("baselines"),
        "baselines_oos": payload.get("baselines_oos"),
        "generated_at": _iso_now(),
    }
    _write_json(config.output.drift_path, drift_out)

    z_thr = preprocessor.return_zscore.threshold_ if preprocessor.return_zscore else None
    v_thr = preprocessor.vol_baseline.threshold_ if preprocessor.vol_baseline else None
    plots = write_result_plots(
        ordered,
        price_plot=config.output.price_plot_path,
        anomalies_plot=config.output.anomalies_plot_path,
        reconstruction_plot=config.output.reconstruction_plot_path,
        loss_plot=config.output.loss_plot_path,
        anomalies_html=config.output.anomalies_html_path,
        reconstruction_html=config.output.reconstruction_html_path,
        zscore_plot=config.output.zscore_plot_path,
        vol_plot=config.output.vol_plot_path,
        drift_plot=config.output.drift_plot_path,
        magnitude_plot=config.output.magnitude_plot_path,
        threshold=metadata.get("threshold"),
        zscore_threshold=z_thr,
        vol_threshold=v_thr,
        psi_frame=psi_frame if isinstance(psi_frame, pd.DataFrame) else None,
        psi_warn=config.drift.psi_warn,
        psi_alert=config.drift.psi_alert,
        history=history,
    )
    metadata["plots"] = {key: Path(value).name for key, value in plots.items()}
    _write_json(config.output.metadata_path, metadata)
    logger.info(
        "Wrote %s (%s flags, %s → %s).",
        config.output.scores_path.name,
        metadata.get("n_anomalies"),
        ordered["Date"].iloc[0],
        ordered["Date"].iloc[-1],
    )
    return ordered


def run_train(
    config: AppConfig,
    *,
    upload: bool,
    epochs: int | None = None,
    skip_walkforward: bool = False,
) -> dict[str, Any]:
    set_global_seeds(config.training.seed)
    frame = _download(config)
    preprocessor = _new_preprocessor(config)
    start, end = (
        (None, None)
        if config.training.full_history
        else (config.data.train_start_date, config.data.train_end_date)
    )
    x_train, x_val, cal_meta = preprocessor.prepare_calibration(
        frame,
        train_start_date=start,
        train_end_date=end,
        validation_fraction=config.data.validation_fraction,
        fit_state=True,
    )
    cal_frame, _ = preprocessor.chronological_split(
        frame,
        train_start_date=start,
        train_end_date=end,
        validation_fraction=config.data.validation_fraction,
    )
    preprocessor.fit_baselines(cal_frame, percentile=config.baseline.percentile)

    autoencoder = _build_autoencoder(config)
    autoencoder.build()
    history = autoencoder.fit(
        x_train,
        x_val,
        epochs=epochs if epochs is not None else config.training.epochs,
        batch_size=config.training.batch_size,
        patience=config.training.patience,
        shuffle=config.training.shuffle,
        min_delta=config.training.min_delta,
        verbose=2,
    )
    train_mae = autoencoder.mean_absolute_error(x_train)
    val_mae = autoencoder.mean_absolute_error(x_val)
    threshold = autoencoder.fit_threshold(x_train)
    _write_frozen_threshold(config, threshold)

    metadata: dict[str, Any] = {
        "mode": "train",
        "instrument": "anomaly_monitor",
        "ticker": config.data.ticker,
        "last_date": last_available_date(frame).strftime("%Y-%m-%d"),
        "lookback": config.preprocessing.lookback,
        "vol_lookback": config.preprocessing.vol_lookback,
        "scale_floor": preprocessor.scale_floor,
        "quiet_vol_cutoff": preprocessor.quiet_vol_cutoff_,
        "threshold": threshold,
        "threshold_frozen": True,
        "threshold_percentile": config.model.threshold_percentile,
        "train_mae": train_mae,
        "earlystop_mae": val_mae,
        "trained_at": _iso_now(),
        "model": keras_model_to_dict(autoencoder.model),
        "calibration": cal_meta,
        "epochs_trained": len(history.history.get("loss", [])),
        "best_earlystop_loss": float(min(history.history.get("val_loss", [float("nan")]))),
        "best_epoch": (
            int(min(range(len(history.history["val_loss"])), key=lambda i: history.history["val_loss"][i])) + 1
            if history.history.get("val_loss")
            else None
        ),
    }
    scores = score_prices(preprocessor, autoencoder, frame)
    config.ensure_output_dir()
    autoencoder.save(config.output.model_path)
    preprocessor.save_state(config.output.state_path)
    _write_reports(config, scores, metadata, preprocessor, history=history.history)

    if config.training.run_walkforward and not skip_walkforward:
        logger.info("Running walk-forward folds (this is the OOS evidence).")
        wf = run_walkforward(config, frame)
        wf.to_csv(config.output.walkforward_path, index=False)
        metadata["walkforward"] = wf.to_dict(orient="records")
        _write_json(config.output.metadata_path, metadata)

    if upload:
        push_artifacts_to_hub(config)
    logger.info("Training complete. threshold=%.6f last_date=%s", threshold, metadata["last_date"])
    return metadata


def run_evaluate(config: AppConfig, *, upload: bool = False) -> dict[str, Any]:
    if not config.output.model_path.exists() or not config.output.state_path.exists():
        raise PipelineError("evaluate requires a saved model and feature_state.json. Run --mode train first.")
    preprocessor = TimeSeriesPreprocessor.load_state(config.output.state_path)
    metadata = _read_json(config.output.metadata_path)
    threshold = _decision_threshold(config, metadata)
    autoencoder = LSTMAutoencoder.load(
        config.output.model_path,
        threshold_percentile=config.model.threshold_percentile,
        threshold=threshold,
    )
    ledger = load_ledger(config.output.scores_path)
    assert_event_integrity(ledger, config.retrain.acceptance_event_dates)

    last_scored = pd.to_datetime(ledger["Date"]).max()
    calendar_buffer = max(40, int(preprocessor.lookback + preprocessor.vol_lookback) * 2)
    start = (last_scored - pd.Timedelta(days=calendar_buffer)).strftime("%Y-%m-%d")
    prices = _download(config, start=start)
    scored = score_prices(preprocessor, autoencoder, prices)
    combined = append_new_rows(ledger, scored)

    metadata["mode"] = "evaluate"
    metadata["threshold"] = threshold
    metadata["threshold_frozen"] = True
    metadata["scores_mode"] = "append"
    metadata["last_date"] = str(combined["Date"].iloc[-1])
    metadata["evaluated_at"] = _iso_now()
    history = None
    if config.output.history_path.exists():
        history = pd.read_csv(config.output.history_path).to_dict(orient="list")
    _write_reports(config, combined, metadata, preprocessor, history=history)
    if upload:
        push_artifacts_to_hub(config)
    logger.info("Evaluation complete. n_anomalies=%s last_date=%s", metadata.get("n_anomalies"), metadata["last_date"])
    return metadata


def run_retrain(config: AppConfig, *, upload: bool, epochs: int | None = None) -> dict[str, Any]:
    """Fine-tune only on recent *quiet* windows. Three-way split. Frozen threshold.

    If the last year has too few quiet windows, we refuse — the recent
    tape is too loud to define a quiet calibration set.
    """
    set_global_seeds(config.training.seed)
    config.ensure_output_dir()
    if not config.output.model_path.exists():
        logger.info("Champion missing locally; attempting Hub restore.")
        pull_artifacts_from_hub(config)
    if not config.output.model_path.exists():
        logger.warning("No champion available. Bootstrapping with a full train run.")
        return run_train(config, upload=upload, epochs=epochs)

    if not config.output.state_path.exists():
        raise PipelineError(
            f"Found {config.output.model_path} but feature_state.json is missing. "
            "Refusing to fine-tune in a different feature space."
        )

    metadata = _read_json(config.output.metadata_path)
    last_date = metadata.get("last_date")
    start = None
    if last_date:
        start = (pd.Timestamp(str(last_date)) - pd.Timedelta(days=int(config.retrain.context_days))).strftime("%Y-%m-%d")
    frame = _download(config, start=start)
    preprocessor = TimeSeriesPreprocessor.load_state(config.output.state_path)
    features = preprocessor.build_frame(frame)
    quiet = features.windows[preprocessor.quiet_mask(features)]
    if quiet.shape[0] < config.retrain.min_sequences:
        raise PipelineError(
            f"Refusing to fine-tune: only {quiet.shape[0]} quiet windows in the "
            "retrain context (min_sequences="
            f"{config.retrain.min_sequences}). The recent tape is too loud "
            "to recalibrate safely."
        )
    x_tune, x_es, x_gate = preprocessor.three_way_split(
        quiet,
        tune_fraction=config.retrain.tune_fraction,
        earlystop_fraction=config.retrain.earlystop_fraction,
    )
    champion = LSTMAutoencoder.load(
        config.output.model_path,
        threshold_percentile=config.model.threshold_percentile,
        threshold=metadata.get("threshold"),
    )
    baseline_mae = champion.mean_absolute_error(x_gate)
    logger.info("Champion quiet-gate MAE (unseen) = %.6f", baseline_mae)
    history = champion.fit(
        x_tune,
        x_es,
        epochs=epochs if epochs is not None else config.retrain.epochs,
        batch_size=config.training.batch_size,
        patience=config.retrain.patience,
        shuffle=config.training.shuffle,
    )
    challenger_mae = champion.mean_absolute_error(x_gate)
    allowed = baseline_mae * (1.0 + config.retrain.max_degradation_ratio)
    logger.info(
        "Challenger quiet-gate MAE = %.6f (allowed <= %.6f).",
        challenger_mae,
        allowed,
    )
    if challenger_mae > allowed:
        raise QualityGateRejected(
            f"Quality gate rejected challenger: MAE {challenger_mae:.6f} "
            f"> allowed {allowed:.6f}. Champion artifacts were not overwritten."
        )

    threshold = _decision_threshold(config, metadata)
    champion.threshold_ = threshold
    ledger = load_ledger(config.output.scores_path)
    assert_event_integrity(ledger, config.retrain.acceptance_event_dates)
    scored = score_prices(preprocessor, champion, frame)
    combined = append_new_rows(ledger, scored)

    updated = {
        **metadata,
        "mode": "retrain",
        "threshold": threshold,
        "threshold_frozen": True,
        "last_date": str(combined["Date"].iloc[-1]),
        "gate_mae": challenger_mae,
        "baseline_gate_mae": baseline_mae,
        "n_tune_windows": int(x_tune.shape[0]),
        "n_earlystop_windows": int(x_es.shape[0]),
        "n_gate_windows": int(x_gate.shape[0]),
        "trained_at": _iso_now(),
        "gate_passed": True,
        "scores_mode": "append",
        "model": keras_model_to_dict(champion.model),
    }
    champion.save(config.output.model_path)
    preprocessor.save_state(config.output.state_path)
    _write_reports(config, combined, updated, preprocessor, history=history.history)
    if upload:
        push_artifacts_to_hub(config)
    logger.info("Retrain promoted. gate_mae=%.6f (was %.6f)", challenger_mae, baseline_mae)
    return updated


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        config = load_config(args.config)
    except Exception as exc:
        logger.error("Failed to load configuration: %s", exc)
        return 2
    logger.info("Starting pipeline mode=%s config=%s", args.mode, config.source_path)
    try:
        if args.mode == "train":
            run_train(
                config,
                upload=not args.skip_upload,
                epochs=args.epochs,
                skip_walkforward=args.skip_walkforward,
            )
        elif args.mode == "retrain":
            run_retrain(config, upload=not args.skip_upload, epochs=args.epochs)
        elif args.mode == "walkforward":
            set_global_seeds(config.training.seed)
            frame = _download(config)
            wf = run_walkforward(config, frame)
            config.ensure_output_dir()
            wf.to_csv(config.output.walkforward_path, index=False)
            logger.info("Walk-forward wrote %s", config.output.walkforward_path)
        else:
            run_evaluate(config, upload=not args.skip_upload)
    except QualityGateRejected as exc:
        logger.warning("%s", exc)
        return 0
    except (DataValidationError, DataDownloadError, PipelineError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected pipeline failure: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
