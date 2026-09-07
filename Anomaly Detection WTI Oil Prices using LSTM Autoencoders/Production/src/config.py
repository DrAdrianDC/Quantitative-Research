"""Typed loader for the project-level ``config.yaml``.

Keeping configuration out of source files lets training, weekly fine-tuning
and GitHub Actions share one contract without code changes.
"""

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


@dataclass(frozen=True)
class DataConfig:
    ticker: str
    period: str
    start_date: str | None
    end_date: str | None
    auto_adjust: bool
    train_end_date: str | None
    validation_fraction: float
    max_null_ratio: float
    download_retries: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class PreprocessingConfig:
    scaler_type: str
    lookback: int
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


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    patience: int
    seed: int
    shuffle: bool


@dataclass(frozen=True)
class RetrainConfig:
    epochs: int
    patience: int
    context_days: int
    max_degradation_ratio: float
    min_sequences: int


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    model_path: Path
    scaler_path: Path
    metadata_path: Path
    scores_path: Path
    anomalies_path: Path
    history_path: Path
    price_plot_path: Path
    anomalies_plot_path: Path
    reconstruction_plot_path: Path
    loss_plot_path: Path
    anomalies_html_path: Path
    reconstruction_html_path: Path


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
    output: OutputConfig
    huggingface: HuggingFaceConfig
    source_path: Path

    def ensure_output_dir(self) -> Path:
        """Create ``output_results/`` if it does not exist."""
        self.output.directory.mkdir(parents=True, exist_ok=True)
        return self.output.directory


def load_config(path: str | Path | None = None) -> AppConfig:
    """Parse YAML configuration into a typed ``AppConfig``.

    Parameters
    ----------
    path:
        Optional override. Defaults to ``<repo>/config.yaml``.
    """
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
    out_raw = _require(raw, "output_results", "root")
    hf_raw = _require(raw, "huggingface", "root")

    output_dir = _as_path(_require(out_raw, "directory", "output_results"))
    output = OutputConfig(
        directory=output_dir,
        model_path=output_dir / _require(out_raw, "model_filename", "output_results"),
        scaler_path=output_dir / _require(out_raw, "scaler_filename", "output_results"),
        metadata_path=output_dir / _require(out_raw, "metadata_filename", "output_results"),
        scores_path=output_dir / _require(out_raw, "scores_filename", "output_results"),
        anomalies_path=output_dir / _require(out_raw, "anomalies_filename", "output_results"),
        history_path=output_dir / _require(out_raw, "history_filename", "output_results"),
        price_plot_path=output_dir / _require(out_raw, "price_plot_filename", "output_results"),
        anomalies_plot_path=output_dir
        / _require(out_raw, "anomalies_plot_filename", "output_results"),
        reconstruction_plot_path=output_dir
        / _require(out_raw, "reconstruction_plot_filename", "output_results"),
        loss_plot_path=output_dir / _require(out_raw, "loss_plot_filename", "output_results"),
        anomalies_html_path=output_dir
        / _require(out_raw, "anomalies_html_filename", "output_results"),
        reconstruction_html_path=output_dir
        / _require(out_raw, "reconstruction_html_filename", "output_results"),
    )

    return AppConfig(
        data=DataConfig(
            ticker=str(_require(data_raw, "ticker", "data")),
            period=str(_require(data_raw, "period", "data")),
            start_date=data_raw.get("start_date"),
            end_date=data_raw.get("end_date"),
            auto_adjust=bool(_require(data_raw, "auto_adjust", "data")),
            train_end_date=data_raw.get("train_end_date"),
            validation_fraction=float(_require(data_raw, "validation_fraction", "data")),
            max_null_ratio=float(_require(data_raw, "max_null_ratio", "data")),
            download_retries=int(_require(data_raw, "download_retries", "data")),
            retry_backoff_seconds=float(
                _require(data_raw, "retry_backoff_seconds", "data")
            ),
        ),
        preprocessing=PreprocessingConfig(
            scaler_type=str(_require(prep_raw, "scaler_type", "preprocessing")).lower(),
            lookback=int(_require(prep_raw, "lookback", "preprocessing")),
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
            threshold_percentile=float(
                _require(model_raw, "threshold_percentile", "model")
            ),
        ),
        training=TrainingConfig(
            epochs=int(_require(train_raw, "epochs", "training")),
            batch_size=int(_require(train_raw, "batch_size", "training")),
            patience=int(_require(train_raw, "patience", "training")),
            seed=int(_require(train_raw, "seed", "training")),
            shuffle=bool(_require(train_raw, "shuffle", "training")),
        ),
        retrain=RetrainConfig(
            epochs=int(_require(retrain_raw, "epochs", "retrain")),
            patience=int(_require(retrain_raw, "patience", "retrain")),
            context_days=int(_require(retrain_raw, "context_days", "retrain")),
            max_degradation_ratio=float(
                _require(retrain_raw, "max_degradation_ratio", "retrain")
            ),
            min_sequences=int(_require(retrain_raw, "min_sequences", "retrain")),
        ),
        output=output,
        huggingface=HuggingFaceConfig(
            repo_id=str(_require(hf_raw, "repo_id", "huggingface")),
            repo_type=str(_require(hf_raw, "repo_type", "huggingface")),
            private=bool(_require(hf_raw, "private", "huggingface")),
        ),
        source_path=config_path,
    )
