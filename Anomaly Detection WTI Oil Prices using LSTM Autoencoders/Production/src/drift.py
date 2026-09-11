"""Data drift vs concept drift for a frozen anomaly monitor.

These are different failure modes and they demand different responses.

Data drift
    P(x) moved. The *inputs* the model sees — locally-normalized 10-day
    return windows — no longer match the calibration distribution.
    Typical cause: a new vol regime, a contract specification change, a
    feed that started interpolating holidays. Response: inspect the feed
    and the contract calendar. Do **not** silently fine-tune; adapting
    weights to a new input distribution without a human turns a frozen
    monitor into a trailing volatility filter.

Concept drift
    P(score | x) moved. Windows that still look like calibration-era
    quiet tape now reconstruct poorly (or the reverse). The notion of
    "normal 10-day shape" has shifted. Response: a human-gated retrain
    on quiet windows, threshold still frozen.

PSI thresholds follow the usual industry bands (Siddiqi / credit-risk
practice): <0.10 stable, 0.10–0.25 shift, >0.25 significant. They are
alarms, not p-values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class DriftError(ValueError):
    """Raised when a drift report cannot be computed."""


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI of ``actual`` vs ``expected`` using quantile bins of ``expected``."""
    exp = _finite(expected)
    act = _finite(actual)
    if exp.size < bins * 2 or act.size < 8:
        raise DriftError("Not enough finite observations to compute PSI.")
    edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return 0.0
    exp_pct = _bin_pct(exp, edges, epsilon)
    act_pct = _bin_pct(act, edges, epsilon)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (no p-value; n is large)."""
    exp = np.sort(_finite(expected))
    act = np.sort(_finite(actual))
    if exp.size < 8 or act.size < 8:
        raise DriftError("Not enough finite observations to compute KS.")
    grid = np.concatenate([exp, act])
    cdf_e = np.searchsorted(exp, grid, side="right") / exp.size
    cdf_a = np.searchsorted(act, grid, side="right") / act.size
    return float(np.max(np.abs(cdf_e - cdf_a)))


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _bin_pct(values: np.ndarray, edges: np.ndarray, epsilon: float) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    pct = counts.astype(np.float64) / max(values.size, 1)
    return np.maximum(pct, epsilon)


def psi_band(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "shift"
    return "significant"


def data_drift_report(
    *,
    expected_x: np.ndarray,
    actual_x: np.ndarray,
    expected_scale: np.ndarray,
    actual_scale: np.ndarray,
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
) -> dict[str, Any]:
    """Drift on the *inputs*: normalized returns and the local scale.

    A jump in ``local_scale`` with stable ``x`` is a vol-regime change
    (the tape got louder; the features already divided it out). A jump
    in ``x`` is a change in the distribution the network actually sees.
    """
    psi_x = population_stability_index(expected_x, actual_x)
    psi_scale = population_stability_index(expected_scale, actual_scale)
    report = {
        "kind": "data_drift",
        "psi_normalized_return": psi_x,
        "psi_normalized_return_band": psi_band(psi_x),
        "ks_normalized_return": ks_statistic(expected_x, actual_x),
        "psi_local_scale": psi_scale,
        "psi_local_scale_band": psi_band(psi_scale),
        "ks_local_scale": ks_statistic(expected_scale, actual_scale),
        "median_x_expected": float(np.median(_finite(expected_x))),
        "median_x_actual": float(np.median(_finite(actual_x))),
        "median_scale_expected": float(np.median(_finite(expected_scale))),
        "median_scale_actual": float(np.median(_finite(actual_scale))),
        "psi_warn": float(psi_warn),
        "psi_alert": float(psi_alert),
        "alert": bool(psi_x >= psi_alert),
        "warn": bool(psi_x >= psi_warn),
    }
    return report


def concept_drift_report(
    *,
    expected_quiet_mse: np.ndarray,
    actual_quiet_mse: np.ndarray,
    max_degradation_ratio: float = 0.10,
) -> dict[str, Any]:
    """Drift on the *score given quiet inputs*.

    Quiet days are the calibration reference. If those days start
    reconstructing worse, the mapping from shape to residual moved.
    Stormy days are *supposed* to reconstruct worse; they are not
    evidence of concept drift.
    """
    exp = _finite(expected_quiet_mse)
    act = _finite(actual_quiet_mse)
    if exp.size < 8 or act.size < 8:
        raise DriftError("Not enough quiet windows to judge concept drift.")
    exp_med = float(np.median(exp))
    act_med = float(np.median(act))
    allowed = exp_med * (1.0 + max_degradation_ratio)
    return {
        "kind": "concept_drift",
        "quiet_mse_expected_mean": float(np.mean(exp)),
        "quiet_mse_actual_mean": float(np.mean(act)),
        "quiet_mse_expected_median": exp_med,
        "quiet_mse_actual_median": act_med,
        "degradation_ratio": float(act_med / exp_med - 1.0) if exp_med > 0 else 0.0,
        "max_degradation_ratio": float(max_degradation_ratio),
        "allowed_median": allowed,
        "alert": bool(act_med > allowed),
        "n_expected": int(exp.size),
        "n_actual": int(act.size),
        "note": "Alert uses the median of quiet-day MSE. The mean is reported but is dominated by leftover ringing from overlapping windows.",
    }


def rolling_psi_series(
    expected_x: np.ndarray,
    dates: pd.Series,
    actual_x: np.ndarray,
    *,
    window: int = 63,
    bins: int = 10,
) -> pd.DataFrame:
    """Rolling PSI of normalized returns vs the frozen calibration sample."""
    parsed = pd.to_datetime(dates)
    x = np.asarray(actual_x, dtype=np.float64)
    if len(parsed) != x.size:
        raise DriftError("dates and actual_x must align.")
    rows = []
    for i in range(window - 1, x.size):
        sl = x[i - window + 1 : i + 1]
        if not np.isfinite(sl).any():
            continue
        try:
            psi = population_stability_index(expected_x, sl, bins=bins)
        except DriftError:
            continue
        rows.append({"Date": parsed.iloc[i] if hasattr(parsed, "iloc") else parsed[i], "psi": psi})
    return pd.DataFrame(rows)
