"""Portfolio figures for the WTI anomaly monitor.

PNG is the source of truth (matplotlib, no kaleido). Interactive HTML is
written when Plotly is installed. Reconstruction error is drawn on a log
axis so isolated spikes remain readable against the quiet baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRICE_COLOR = "#1f4e79"
_ANOMALY_COLOR = "#c0392b"
_THRESHOLD_COLOR = "#c0392b"
_TRAIN_COLOR = "#1f4e79"
_VAL_COLOR = "#d35400"
_DRIFT_COLOR = "#6c3483"
_VOL_COLOR = "#1a5276"


class PlotError(RuntimeError):
    """Raised when a required figure cannot be written."""


def _year_span(dates: pd.Series) -> str:
    parsed = pd.to_datetime(dates, errors="coerce").dropna()
    if parsed.empty:
        return ""
    start = int(parsed.min().year)
    end = int(parsed.max().year)
    return f"{start}–{end}" if start != end else f"{start}"


def _prepare_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        raise PlotError("Cannot plot an empty scores frame.")
    required = {"Date", "Close", "reconstruction_mse", "anomaly"}
    missing = required - set(scores.columns)
    if missing:
        raise PlotError(f"Scores frame is missing columns: {sorted(missing)}")
    frame = scores.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame["reconstruction_mse"] = pd.to_numeric(frame["reconstruction_mse"], errors="coerce")
    frame["anomaly"] = pd.to_numeric(frame["anomaly"], errors="coerce").fillna(0).astype(int)
    frame = frame.dropna(subset=["Date", "Close", "reconstruction_mse"])
    if frame.empty:
        raise PlotError("Scores frame has no plottable rows after cleaning.")
    return frame.sort_values("Date").reset_index(drop=True)


def _save_png(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    return path


def plot_price_series(scores: pd.DataFrame, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    frame = _prepare_scores(scores)
    span = _year_span(frame["Date"])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame["Date"], frame["Close"], color=_PRICE_COLOR, linewidth=1.1, label="Close price")
    ax.set_title(f"WTI Crude Oil Price ({span})", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/bbl)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def plot_anomalies(scores: pd.DataFrame, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    frame = _prepare_scores(scores)
    detected = frame.loc[frame["anomaly"] == 1]
    span = _year_span(frame["Date"])
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(frame["Date"], frame["Close"], color=_PRICE_COLOR, linewidth=1.05, label="Close price", zorder=1)
    if not detected.empty:
        ax.scatter(
            detected["Date"],
            detected["Close"],
            color=_ANOMALY_COLOR,
            s=22,
            label="Anomaly (P99)",
            zorder=3,
        )
    ax.set_title(f"WTI anomaly flags {span}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/bbl)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def plot_reconstruction_error(
    scores: pd.DataFrame,
    path: str | Path,
    *,
    threshold: float | None,
) -> Path:
    import matplotlib.pyplot as plt

    frame = _prepare_scores(scores)
    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.plot(
        frame["Date"],
        frame["reconstruction_mse"],
        color=_PRICE_COLOR,
        linewidth=0.8,
        label="Reconstruction MSE",
    )
    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, color=_THRESHOLD_COLOR, linestyle="--", linewidth=1.2, label="P99 threshold")
    ax.set_yscale("log")
    ax.set_title("Reconstruction error (log scale)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("MSE (log)")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def plot_magnitude(
    scores: pd.DataFrame,
    path: str | Path,
) -> Path | None:
    if "magnitude" not in scores.columns:
        return None
    import matplotlib.pyplot as plt

    frame = scores.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["magnitude"] = pd.to_numeric(frame["magnitude"], errors="coerce")
    frame = frame.dropna(subset=["Date", "magnitude"]).sort_values("Date")
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot(frame["Date"], frame["magnitude"], color=_PRICE_COLOR, linewidth=0.8, label="Magnitude")
    ax.axhline(0.0, color=_THRESHOLD_COLOR, linestyle="--", linewidth=1.2, label="M = 0 (threshold)")
    ax.set_title("Magnitude  log10(MSE / threshold)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("M")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def plot_zscore(scores: pd.DataFrame, path: str | Path, *, threshold: float | None) -> Path | None:
    if "return_zscore" not in scores.columns:
        return None
    import matplotlib.pyplot as plt

    frame = scores.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["return_zscore"] = pd.to_numeric(frame["return_zscore"], errors="coerce")
    if "zscore_anomaly" in frame.columns:
        frame["zscore_anomaly"] = pd.to_numeric(frame["zscore_anomaly"], errors="coerce").fillna(0).astype(int)
    else:
        frame["zscore_anomaly"] = 0
    frame = frame.dropna(subset=["Date"]).sort_values("Date")
    if frame.empty:
        return None
    abs_z = frame["return_zscore"].abs()
    flagged = frame.loc[frame["zscore_anomaly"] == 1]
    span = _year_span(frame["Date"])
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot(frame["Date"], abs_z, color=_PRICE_COLOR, linewidth=0.9, label="|z| of daily ΔClose")
    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, color=_THRESHOLD_COLOR, linestyle="--", linewidth=1.2, label="P99 |z| (calibration)")
    if not flagged.empty:
        ax.scatter(flagged["Date"], flagged["return_zscore"].abs(), color=_ANOMALY_COLOR, s=18, zorder=3, label="Z-score flag")
    ax.set_title(f"One-day jump baseline ({span})", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("|z|")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def plot_vol_baseline(scores: pd.DataFrame, path: str | Path, *, threshold: float | None) -> Path | None:
    if "rolling_vol" not in scores.columns:
        return None
    import matplotlib.pyplot as plt

    frame = scores.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["rolling_vol"] = pd.to_numeric(frame["rolling_vol"], errors="coerce")
    if "vol_anomaly" in frame.columns:
        frame["vol_anomaly"] = pd.to_numeric(frame["vol_anomaly"], errors="coerce").fillna(0).astype(int)
    else:
        frame["vol_anomaly"] = 0
    frame = frame.dropna(subset=["Date", "rolling_vol"]).sort_values("Date")
    if frame.empty:
        return None
    flagged = frame.loc[frame["vol_anomaly"] == 1]
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot(frame["Date"], frame["rolling_vol"], color=_VOL_COLOR, linewidth=0.9, label="10-day realized vol of ΔClose")
    if threshold is not None and np.isfinite(threshold):
        ax.axhline(threshold, color=_THRESHOLD_COLOR, linestyle="--", linewidth=1.2, label="P99 vol (calibration)")
    if not flagged.empty:
        ax.scatter(flagged["Date"], flagged["rolling_vol"], color=_ANOMALY_COLOR, s=18, zorder=3, label="Vol flag")
    ax.set_title("Rolling volatility baseline", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/bbl")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def plot_drift(psi: pd.DataFrame, path: str | Path, *, warn: float, alert: float) -> Path | None:
    if psi is None or psi.empty:
        return None
    import matplotlib.pyplot as plt

    frame = psi.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "psi"]).sort_values("Date")
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot(frame["Date"], frame["psi"], color=_DRIFT_COLOR, linewidth=1.0, label="Rolling PSI of x_t")
    ax.axhline(warn, color="#b7950b", linestyle="--", linewidth=1.1, label=f"warn ({warn:.2f})")
    ax.axhline(alert, color=_THRESHOLD_COLOR, linestyle="--", linewidth=1.1, label=f"alert ({alert:.2f})")
    ax.set_title("Data drift — rolling PSI of vol-normalized returns vs calibration", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("PSI")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def plot_training_loss(history: Mapping[str, Any] | pd.DataFrame, path: str | Path) -> Path | None:
    import matplotlib.pyplot as plt

    if isinstance(history, pd.DataFrame):
        payload = {col: history[col].tolist() for col in history.columns}
    else:
        payload = dict(history)
    train_loss = payload.get("loss")
    if not train_loss:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color=_TRAIN_COLOR, linewidth=1.4, label="Training loss (quiet windows)")
    val_loss = payload.get("val_loss")
    if val_loss:
        ax.plot(epochs, val_loss, color=_VAL_COLOR, linewidth=1.4, label="Early-stop loss (quiet hold-out)")
    ax.set_title("Calibration loss (quiet windows only)", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def _write_plotly_html(
    scores: pd.DataFrame,
    anomalies_html: Path,
    reconstruction_html: Path,
    *,
    threshold: float | None,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.info("Plotly is not installed; skipping interactive HTML figures.")
        return

    frame = _prepare_scores(scores)
    detected = frame.loc[frame["anomaly"] == 1]
    span = _year_span(frame["Date"])

    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(x=frame["Date"], y=frame["Close"], mode="lines", name="Close price",
                   line={"color": _PRICE_COLOR, "width": 1.5})
    )
    if not detected.empty:
        price_fig.add_trace(
            go.Scatter(x=detected["Date"], y=detected["Close"], mode="markers", name="Anomaly",
                       marker={"color": _ANOMALY_COLOR, "size": 8})
        )
    price_fig.update_layout(
        title=f"WTI anomaly flags {span}",
        xaxis_title="Date",
        yaxis_title="Price (USD/bbl)",
        template="plotly_white",
        font={"size": 14},
        showlegend=True,
    )
    price_fig.write_html(anomalies_html, include_plotlyjs="cdn", full_html=True)

    error_fig = go.Figure()
    error_fig.add_trace(
        go.Scatter(x=frame["Date"], y=frame["reconstruction_mse"], name="Reconstruction MSE",
                   line={"color": _PRICE_COLOR, "width": 1.2})
    )
    if threshold is not None and np.isfinite(threshold):
        error_fig.add_trace(
            go.Scatter(x=frame["Date"], y=[threshold] * len(frame), name="P99 threshold",
                       line={"color": _THRESHOLD_COLOR, "dash": "dash"})
        )
    error_fig.update_layout(
        title="Reconstruction error (log scale)",
        xaxis_title="Date",
        yaxis_title="MSE",
        yaxis_type="log",
        template="plotly_white",
        font={"size": 14},
    )
    error_fig.write_html(reconstruction_html, include_plotlyjs="cdn", full_html=True)


def write_result_plots(
    scores: pd.DataFrame,
    *,
    price_plot: Path,
    anomalies_plot: Path,
    reconstruction_plot: Path,
    loss_plot: Path | None = None,
    anomalies_html: Path | None = None,
    reconstruction_html: Path | None = None,
    zscore_plot: Path | None = None,
    vol_plot: Path | None = None,
    drift_plot: Path | None = None,
    magnitude_plot: Path | None = None,
    threshold: float | None = None,
    zscore_threshold: float | None = None,
    vol_threshold: float | None = None,
    psi_frame: pd.DataFrame | None = None,
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
    history: Mapping[str, Any] | pd.DataFrame | None = None,
) -> dict[str, str]:
    written: dict[str, str] = {}
    written["price_plot"] = str(plot_price_series(scores, price_plot))
    written["anomalies_plot"] = str(plot_anomalies(scores, anomalies_plot))
    written["reconstruction_plot"] = str(
        plot_reconstruction_error(scores, reconstruction_plot, threshold=threshold)
    )
    if magnitude_plot is not None:
        mag = plot_magnitude(scores, magnitude_plot)
        if mag is not None:
            written["magnitude_plot"] = str(mag)
    if zscore_plot is not None:
        z_path = plot_zscore(scores, zscore_plot, threshold=zscore_threshold)
        if z_path is not None:
            written["zscore_plot"] = str(z_path)
    if vol_plot is not None:
        v_path = plot_vol_baseline(scores, vol_plot, threshold=vol_threshold)
        if v_path is not None:
            written["vol_plot"] = str(v_path)
    if drift_plot is not None and psi_frame is not None:
        d_path = plot_drift(psi_frame, drift_plot, warn=psi_warn, alert=psi_alert)
        if d_path is not None:
            written["drift_plot"] = str(d_path)
    if history is not None and loss_plot is not None:
        loss_path = plot_training_loss(history, loss_plot)
        if loss_path is not None:
            written["loss_plot"] = str(loss_path)
    if anomalies_html is not None and reconstruction_html is not None:
        _write_plotly_html(scores, anomalies_html, reconstruction_html, threshold=threshold)
        if anomalies_html.exists():
            written["anomalies_html"] = str(anomalies_html)
        if reconstruction_html.exists():
            written["reconstruction_html"] = str(reconstruction_html)
    logger.info("Wrote portfolio plots: %s", ", ".join(written))
    return written
