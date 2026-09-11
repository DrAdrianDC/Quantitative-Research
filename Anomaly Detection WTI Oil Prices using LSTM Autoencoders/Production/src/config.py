"""Typed loader for the project-level ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(ValueError):
    """Raised when ``config.yaml`` is missing, malformed or incomplete."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing '{context}.{key}' in configuration.")
    return mapping[key]


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _as_date_tuple(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item)
    raise ConfigError("acceptance_event_dates must be a string or a list of strings.")


def _as_folds(value: Any) -> tuple[tuple[str, str], ...]:
    if not value:
        return ()
    folds = []
    for item in value:
        folds.append((str(item["train_end"]), str(item["test_end"])))
    return tuple(folds)


@dataclass(frozen=True)
class DataConfig:
    ticker: str
    period: str
    start_date: str | None
    end_date: str | None
    auto_adjust: bool
    train_start_date: str | None
    train_end_date: str | None
    validation_fraction: float
    max_null_ratio: float
    download_retries: int
    retry_backoff_seconds: float
    snapshot_path: Path


@dataclass(frozen=True)
class PreprocessingConfig:
    lookback: int
    vol_lookback: int
    min_scale_floor: float
    scale_floor_quantile: float
    max_train_vol_percentile: float
    target_column: str
    date_column: str


@dataclass(frozen=True)
class ModelConfig:
    encoder_units: tuple[int, ...]
    decoder_units: tuple[int, ...]
    dropout: float
    activation: str
    loss: str
    optimizer: str
    clipnorm: float
    threshold_percentile: float
    frozen_threshold: float | None


@dataclass(frozen=True)
class BaselineConfig:
    percentile: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    patience: int
    min_delta: float
    seed: int
    shuffle: bool
    full_history: bool
    run_walkforward: bool


@dataclass(frozen=True)
class DriftConfig:
    psi_warn: float
    psi_alert: float
    quiet_mae_degradation: float
    rolling_psi_window: int
    recent_days: int


@dataclass(frozen=True)
class RetrainConfig:
    epochs: int
    patience: int
    context_days: int
    max_degradation_ratio: float
    min_sequences: int
    tune_fraction: float
    earlystop_fraction: float
    acceptance_event_dates: tuple[str, ...]


@dataclass(frozen=True)
class WalkforwardConfig:
    epochs: int
    patience: int
    folds: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    model_path: Path
    state_path: Path
    metadata_path: Path
    scores_path: Path
    anomalies_path: Path
    history_path: Path
    drift_path: Path
    events_path: Path
    walkforward_path: Path
    price_plot_path: Path
    anomalies_plot_path: Path
    reconstruction_plot_path: Path
    loss_plot_path: Path
    anomalies_html_path: Path
    reconstruction_html_path: Path
    zscore_plot_path: Path
    vol_plot_path: Path
    drift_plot_path: Path
    magnitude_plot_path: Path


@dataclass(frozen=True)
class HuggingFaceConfig:
    repo_id: str
    repo_type: str
    private: bool


@dataclass(frozen=True)
class AppConfig:
    """Immutable snapshot of the runtime configuration."""

    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    retrain: RetrainConfig
    baseline: BaselineConfig
    drift: DriftConfig
    walkforward: WalkforwardConfig
    output: OutputConfig
    huggingface: HuggingFaceConfig
    source_path: Path

    def ensure_output_dir(self) -> Path:
        self.output.directory.mkdir(parents=True, exist_ok=True)
        return self.output.directory


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Top-level configuration must be a mapping.")

    data_raw = _require(raw, "data", "root")
    prep_raw = _require(raw, "preprocessing", "root")
    model_raw = _require(raw, "model", "root")
    train_raw = _require(raw, "training", "root")
    retrain_raw = _require(raw, "retrain", "root")
    baseline_raw = raw.get("baseline") or {}
    drift_raw = raw.get("drift") or {}
    wf_raw = raw.get("walkforward") or {}
    out_raw = _require(raw, "output_results", "root")
    hf_raw = _require(raw, "huggingface", "root")

    output_dir = _as_path(_require(out_raw, "directory", "output_results"))
    snapshot_name = str(data_raw.get("snapshot_filename") or "price_snapshot.csv")

    output = OutputConfig(
        directory=output_dir,
        model_path=output_dir / _require(out_raw, "model_filename", "output_results"),
        state_path=output_dir / _require(out_raw, "state_filename", "output_results"),
        metadata_path=output_dir / _require(out_raw, "metadata_filename", "output_results"),
        scores_path=output_dir / _require(out_raw, "scores_filename", "output_results"),
        anomalies_path=output_dir / _require(out_raw, "anomalies_filename", "output_results"),
        history_path=output_dir / _require(out_raw, "history_filename", "output_results"),
        drift_path=output_dir / _require(out_raw, "drift_filename", "output_results"),
        events_path=output_dir / _require(out_raw, "events_filename", "output_results"),
        walkforward_path=output_dir / _require(out_raw, "walkforward_filename", "output_results"),
        price_plot_path=output_dir / _require(out_raw, "price_plot_filename", "output_results"),
        anomalies_plot_path=output_dir / _require(out_raw, "anomalies_plot_filename", "output_results"),
        reconstruction_plot_path=output_dir
        / _require(out_raw, "reconstruction_plot_filename", "output_results"),
        loss_plot_path=output_dir / _require(out_raw, "loss_plot_filename", "output_results"),
        anomalies_html_path=output_dir / _require(out_raw, "anomalies_html_filename", "output_results"),
        reconstruction_html_path=output_dir
        / _require(out_raw, "reconstruction_html_filename", "output_results"),
        zscore_plot_path=output_dir / _require(out_raw, "zscore_plot_filename", "output_results"),
        vol_plot_path=output_dir / _require(out_raw, "vol_plot_filename", "output_results"),
        drift_plot_path=output_dir / _require(out_raw, "drift_plot_filename", "output_results"),
        magnitude_plot_path=output_dir / _require(out_raw, "magnitude_plot_filename", "output_results"),
    )

    return AppConfig(
        data=DataConfig(
            ticker=str(_require(data_raw, "ticker", "data")),
            period=str(_require(data_raw, "period", "data")),
            start_date=data_raw.get("start_date"),
            end_date=data_raw.get("end_date"),
            auto_adjust=bool(_require(data_raw, "auto_adjust", "data")),
            train_start_date=data_raw.get("train_start_date"),
            train_end_date=data_raw.get("train_end_date"),
            validation_fraction=float(_require(data_raw, "validation_fraction", "data")),
            max_null_ratio=float(_require(data_raw, "max_null_ratio", "data")),
            download_retries=int(_require(data_raw, "download_retries", "data")),
            retry_backoff_seconds=float(_require(data_raw, "retry_backoff_seconds", "data")),
            snapshot_path=output_dir / snapshot_name,
        ),
        preprocessing=PreprocessingConfig(
            lookback=int(_require(prep_raw, "lookback", "preprocessing")),
            vol_lookback=int(_require(prep_raw, "vol_lookback", "preprocessing")),
            min_scale_floor=float(prep_raw.get("min_scale_floor", 0.05)),
            scale_floor_quantile=float(prep_raw.get("scale_floor_quantile", 0.05)),
            max_train_vol_percentile=float(prep_raw.get("max_train_vol_percentile", 95.0)),
            target_column=str(_require(prep_raw, "target_column", "preprocessing")),
            date_column=str(_require(prep_raw, "date_column", "preprocessing")),
        ),
        model=ModelConfig(
            encoder_units=tuple(int(u) for u in _require(model_raw, "encoder_units", "model")),
            decoder_units=tuple(int(u) for u in _require(model_raw, "decoder_units", "model")),
            dropout=float(_require(model_raw, "dropout", "model")),
            activation=str(_require(model_raw, "activation", "model")),
            loss=str(_require(model_raw, "loss", "model")),
            optimizer=str(_require(model_raw, "optimizer", "model")),
            clipnorm=float(_require(model_raw, "clipnorm", "model")),
            threshold_percentile=float(_require(model_raw, "threshold_percentile", "model")),
            frozen_threshold=(
                None
                if model_raw.get("frozen_threshold") in (None, "null")
                else float(model_raw["frozen_threshold"])
            ),
        ),
        training=TrainingConfig(
            epochs=int(_require(train_raw, "epochs", "training")),
            batch_size=int(_require(train_raw, "batch_size", "training")),
            patience=int(_require(train_raw, "patience", "training")),
            min_delta=float(train_raw.get("min_delta", 0.0)),
            seed=int(_require(train_raw, "seed", "training")),
            shuffle=bool(_require(train_raw, "shuffle", "training")),
            full_history=bool(train_raw.get("full_history", False)),
            run_walkforward=bool(train_raw.get("run_walkforward", False)),
        ),
        retrain=RetrainConfig(
            epochs=int(_require(retrain_raw, "epochs", "retrain")),
            patience=int(_require(retrain_raw, "patience", "retrain")),
            context_days=int(_require(retrain_raw, "context_days", "retrain")),
            max_degradation_ratio=float(_require(retrain_raw, "max_degradation_ratio", "retrain")),
            min_sequences=int(_require(retrain_raw, "min_sequences", "retrain")),
            tune_fraction=float(retrain_raw.get("tune_fraction", 0.60)),
            earlystop_fraction=float(retrain_raw.get("earlystop_fraction", 0.20)),
            acceptance_event_dates=_as_date_tuple(
                retrain_raw.get("acceptance_event_dates")
                or retrain_raw.get("acceptance_event_date")
            ),
        ),
        baseline=BaselineConfig(percentile=float(baseline_raw.get("percentile", 99.0))),
        drift=DriftConfig(
            psi_warn=float(drift_raw.get("psi_warn", 0.10)),
            psi_alert=float(drift_raw.get("psi_alert", 0.25)),
            quiet_mae_degradation=float(drift_raw.get("quiet_mae_degradation", 0.10)),
            rolling_psi_window=int(drift_raw.get("rolling_psi_window", 63)),
            recent_days=int(drift_raw.get("recent_days", 252)),
        ),
        walkforward=WalkforwardConfig(
            epochs=int(wf_raw.get("epochs", 40)),
            patience=int(wf_raw.get("patience", 12)),
            folds=_as_folds(wf_raw.get("folds")),
        ),
        output=output,
        huggingface=HuggingFaceConfig(
            repo_id=str(_require(hf_raw, "repo_id", "huggingface")),
            repo_type=str(_require(hf_raw, "repo_type", "huggingface")),
            private=bool(_require(hf_raw, "private", "huggingface")),
        ),
        source_path=config_path,
    )
