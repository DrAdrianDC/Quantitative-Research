"""Portfolio figures for the WTI anomaly detector.

The published notebook and
https://github.com/DrAdrianDC/Portfolio-Machine_Learning
ship two images a reviewer opens first: the raw price series and the
price series with anomaly markers. CSV scores without those plots are
not a complete deliverable.

PNG is the source of truth (matplotlib, no kaleido). Interactive HTML
is written when Plotly is installed.
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
    frame["reconstruction_mse"] = pd.to_numeric(
        frame["reconstruction_mse"], errors="coerce"
    )
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
    """WTI close series — counterpart of the notebook ``plot.png``."""
    import matplotlib.pyplot as plt

    frame = _prepare_scores(scores)
    span = _year_span(frame["Date"])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame["Date"], frame["Close"], color=_PRICE_COLOR, linewidth=1.1, label="Close price")
    ax.set_title(f"WTI Crude Oil Price ({span})", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def plot_anomalies(scores: pd.DataFrame, path: str | Path) -> Path:
    """Price series with red anomaly markers — the portfolio money plot.

    Matches ``plot-anomalies.png`` from the research notebook: line for
    Close, scatter for days whose reconstruction MSE exceeds the threshold.
    """
    import matplotlib.pyplot as plt

    frame = _prepare_scores(scores)
    detected = frame.loc[frame["anomaly"] == 1]
    span = _year_span(frame["Date"])
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(
        frame["Date"],
        frame["Close"],
        color=_PRICE_COLOR,
        linewidth=1.05,
        label="Close price",
        zorder=1,
    )
    if not detected.empty:
        ax.scatter(
            detected["Date"],
            detected["Close"],
            color=_ANOMALY_COLOR,
            s=22,
            label="Anomaly",
            zorder=3,
        )
    ax.set_title(f"Anomaly Detection WTI Oil Price {span}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
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
    """Reconstruction MSE over time with the decision threshold."""
    import matplotlib.pyplot as plt

    frame = _prepare_scores(scores)
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot(
        frame["Date"],
        frame["reconstruction_mse"],
        color=_PRICE_COLOR,
        linewidth=0.9,
        label="Reconstruction error",
    )
    if threshold is not None and np.isfinite(threshold):
        ax.axhline(
            threshold,
            color=_THRESHOLD_COLOR,
            linestyle="--",
            linewidth=1.2,
            label="Threshold",
        )
    ax.set_title("Reconstruction Error Over Time", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    destination = _save_png(fig, Path(path))
    plt.close(fig)
    return destination


def plot_training_loss(history: Mapping[str, Any] | pd.DataFrame, path: str | Path) -> Path | None:
    """Training vs validation loss. Returns ``None`` if history has no loss."""
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
    ax.plot(epochs, train_loss, color=_TRAIN_COLOR, linewidth=1.4, label="Training loss")
    val_loss = payload.get("val_loss")
    if val_loss:
        ax.plot(epochs, val_loss, color=_VAL_COLOR, linewidth=1.4, label="Validation loss")
    ax.set_title("Training and Validation Loss", fontsize=14)
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
        go.Scatter(
            x=frame["Date"],
            y=frame["Close"],
            mode="lines",
            name="Close price",
            line={"color": _PRICE_COLOR, "width": 1.5},
        )
    )
    if not detected.empty:
        price_fig.add_trace(
            go.Scatter(
                x=detected["Date"],
                y=detected["Close"],
                mode="markers",
                name="Anomaly",
                marker={"color": _ANOMALY_COLOR, "size": 8},
            )
        )
    price_fig.update_layout(
        title=f"Anomaly Detection WTI Oil Price {span}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        font={"size": 14},
        showlegend=True,
    )
    price_fig.write_html(anomalies_html, include_plotlyjs="cdn", full_html=True)

    error_fig = go.Figure()
    error_fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["reconstruction_mse"],
            name="Reconstruction Error",
            line={"color": _PRICE_COLOR, "width": 1.2},
        )
    )
    if threshold is not None and np.isfinite(threshold):
        error_fig.add_trace(
            go.Scatter(
                x=frame["Date"],
                y=[threshold] * len(frame),
                name="Threshold",
                line={"color": _THRESHOLD_COLOR, "dash": "dash"},
            )
        )
    error_fig.update_layout(
        title="Reconstruction Error Over Time",
        xaxis_title="Date",
        yaxis_title="MSE",
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
    threshold: float | None = None,
    history: Mapping[str, Any] | pd.DataFrame | None = None,
) -> dict[str, str]:
    """Write the full portfolio figure set and return the paths that landed."""
    written: dict[str, str] = {}
    written["price_plot"] = str(plot_price_series(scores, price_plot))
    written["anomalies_plot"] = str(plot_anomalies(scores, anomalies_plot))
    written["reconstruction_plot"] = str(
        plot_reconstruction_error(scores, reconstruction_plot, threshold=threshold)
    )
    if history is not None and loss_plot is not None:
        loss_path = plot_training_loss(history, loss_plot)
        if loss_path is not None:
            written["loss_plot"] = str(loss_path)
    if anomalies_html is not None and reconstruction_html is not None:
        _write_plotly_html(
            scores,
            anomalies_html,
            reconstruction_html,
            threshold=threshold,
        )
        if anomalies_html.exists():
            written["anomalies_html"] = str(anomalies_html)
        if reconstruction_html.exists():
            written["reconstruction_html"] = str(reconstruction_html)
    logger.info("Wrote portfolio plots: %s", ", ".join(written))
    return written
