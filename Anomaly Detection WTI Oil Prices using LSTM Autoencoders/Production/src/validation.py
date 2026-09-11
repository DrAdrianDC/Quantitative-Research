"""Scoring, walk-forward, and out-of-sample reports. Imports TensorFlow."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.baseline import attach_baselines, comparison_summary
from src.config import AppConfig
from src.drift import concept_drift_report, data_drift_report, rolling_psi_series
from src.events import EVENT_CATALOG, events_between, events_for_era, summarize_detection
from src.model import LSTMAutoencoder, magnitude_score, set_global_seeds
from src.preprocessor import TimeSeriesPreprocessor

logger = logging.getLogger(__name__)


def score_prices(
    preprocessor: TimeSeriesPreprocessor,
    autoencoder: LSTMAutoencoder,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Per-day reconstruction MSE, magnitude, LSTM flag, and frozen baselines."""
    if autoencoder.threshold_ is None:
        raise RuntimeError("Decision threshold is not set; cannot flag windows.")
    features = preprocessor.build_frame(prices)
    valid = features.valid_mask
    windows = features.windows[valid]
    end_idx = features.end_index[valid]
    errors = autoencoder.reconstruction_errors(windows, metric="mse")
    flags = (errors > autoencoder.threshold_).astype(int)
    mag = magnitude_score(errors, autoencoder.threshold_)
    dates = pd.to_datetime(pd.Index(features.dates[end_idx]))
    scores = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Close": features.close[end_idx],
            "price_change": features.price_change[end_idx],
            "local_scale": features.local_scale[end_idx],
            "normalized_return": features.normalized_return[end_idx],
            "reconstruction_mse": errors,
            "magnitude": mag,
            "anomaly": flags.astype(int),
        }
    )
    scores = attach_baselines(
        scores,
        close=features.close,
        dates=pd.Series(features.dates),
        zscore=preprocessor.return_zscore,
        vol=preprocessor.vol_baseline,
    )
    return scores.sort_values("Date").reset_index(drop=True)


def split_calibration_window(
    scores: pd.DataFrame,
    train_end_date: str | None,
    train_start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split scores into historical (before start), calibration, and OOS.

    Historical rows are scored but were never in the loss. They are not
    the reported OOS set — that is strictly after ``train_end_date``.
    """
    empty = scores.iloc[0:0].copy()
    if not train_end_date:
        return empty, empty, scores.copy()
    dates = pd.to_datetime(scores["Date"])
    cutoff = pd.Timestamp(train_end_date)
    oos = scores.loc[dates > cutoff].copy()
    in_or_before = scores.loc[dates <= cutoff].copy()
    if not train_start_date:
        return empty, in_or_before, oos
    start = pd.Timestamp(train_start_date)
    in_dates = pd.to_datetime(in_or_before["Date"])
    historical = in_or_before.loc[in_dates < start].copy()
    cal = in_or_before.loc[in_dates >= start].copy()
    return historical, cal, oos


def split_by_train_end(scores: pd.DataFrame, train_end_date: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper: calibration ∪ historical, then OOS."""
    historical, cal, oos = split_calibration_window(scores, train_end_date)
    return pd.concat([historical, cal], ignore_index=True), oos


def evaluation_payload(
    scores: pd.DataFrame,
    *,
    train_end_date: str | None,
    preprocessor: TimeSeriesPreprocessor,
    recent_days: int,
    psi_warn: float,
    psi_alert: float,
    quiet_mae_degradation: float,
    rolling_psi_window: int,
    train_start_date: str | None = None,
) -> dict[str, Any]:
    """Event recall, baseline overlap, data vs concept drift."""
    historical, cal, oos = split_calibration_window(
        scores, train_end_date, train_start_date=train_start_date
    )
    payload: dict[str, Any] = {
        "full": summarize_detection(scores, events=EVENT_CATALOG),
        "historical": summarize_detection(historical, events=events_for_era("pre")) if len(historical) else {},
        "calibration": summarize_detection(cal, events=events_for_era("calibration")) if len(cal) else {},
        "oos": summarize_detection(oos, events=events_for_era("oos")) if len(oos) else {},
        "baselines": comparison_summary(scores),
        "baselines_oos": comparison_summary(oos) if len(oos) else {},
    }
    oos_hits = payload["oos"].get("events") or []
    payload["oos_event_table"] = [row for row in oos_hits if row.get("era") == "oos"]
    payload["calibration_event_table"] = [
        row for row in (payload["full"].get("events") or []) if row.get("era") == "calibration"
    ]
    payload["historical_event_table"] = [
        row for row in (payload["full"].get("events") or []) if row.get("era") == "pre"
    ]

    if preprocessor.quiet_vol_cutoff_ is not None and "local_scale" in scores.columns:
        quiet = scores["local_scale"] <= preprocessor.quiet_vol_cutoff_
    else:
        quiet = pd.Series(False, index=scores.index)

    cal_x = cal["normalized_return"].to_numpy() if "normalized_return" in cal.columns else np.array([])
    recent = scores.tail(int(recent_days))
    try:
        payload["data_drift"] = data_drift_report(
            expected_x=cal_x,
            actual_x=recent["normalized_return"].to_numpy(),
            expected_scale=cal["local_scale"].to_numpy(),
            actual_scale=recent["local_scale"].to_numpy(),
            psi_warn=psi_warn,
            psi_alert=psi_alert,
        )
    except Exception as exc:
        payload["data_drift"] = {"error": str(exc)}

    cal_quiet_mse = cal.loc[quiet.reindex(cal.index).fillna(False), "reconstruction_mse"].to_numpy()
    recent_quiet = recent.loc[
        quiet.reindex(recent.index).fillna(False), "reconstruction_mse"
    ].to_numpy()
    try:
        payload["concept_drift"] = concept_drift_report(
            expected_quiet_mse=cal_quiet_mse,
            actual_quiet_mse=recent_quiet,
            max_degradation_ratio=quiet_mae_degradation,
        )
    except Exception as exc:
        payload["concept_drift"] = {"error": str(exc)}

    try:
        payload["rolling_psi"] = rolling_psi_series(
            cal_x,
            scores["Date"],
            scores["normalized_return"].to_numpy(),
            window=rolling_psi_window,
        )
    except Exception:
        payload["rolling_psi"] = pd.DataFrame()

    # Detector disagreement on OOS events: LSTM vs rolling-vol baseline.
    payload["honest_baseline"] = _honest_event_comparison(oos)
    return payload


def _honest_event_comparison(
    oos: pd.DataFrame,
    events: Sequence | None = None,
) -> dict[str, Any]:
    if oos.empty:
        return {}
    from src.events import event_hit_mask

    catalog = list(events) if events is not None else list(events_for_era("oos"))
    if not catalog:
        return {"lstm_recall": None, "vol_recall": None}
    lstm = event_hit_mask(oos, catalog, flag_column="anomaly")
    result: dict[str, Any] = {"lstm": lstm.to_dict(orient="records")}
    if "vol_anomaly" in oos.columns:
        vol = event_hit_mask(oos, catalog, flag_column="vol_anomaly")
        result["rolling_vol"] = vol.to_dict(orient="records")
        result["lstm_recall"] = float(lstm["hit"].mean()) if len(lstm) else None
        result["vol_recall"] = float(vol["hit"].mean()) if len(vol) else None
        merged = lstm.merge(vol, on=["date", "name"], suffixes=("_lstm", "_vol"))
        result["lstm_only_events"] = merged.loc[
            (merged["hit_lstm"] == 1) & (merged["hit_vol"] == 0), "name"
        ].tolist()
        result["vol_only_events"] = merged.loc[
            (merged["hit_lstm"] == 0) & (merged["hit_vol"] == 1), "name"
        ].tolist()
        result["both_events"] = merged.loc[
            (merged["hit_lstm"] == 1) & (merged["hit_vol"] == 1), "name"
        ].tolist()
        result["neither_events"] = merged.loc[
            (merged["hit_lstm"] == 0) & (merged["hit_vol"] == 0), "name"
        ].tolist()
    return result


def run_walkforward(
    config: AppConfig,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Expanding-window folds. Each fold drops loud calibration windows and
    freezes its own P99 threshold. Reported numbers are test-block only.
    """
    rows = []
    for train_end, test_end in config.walkforward.folds:
        logger.info("Walk-forward fold train_end=%s test_end=%s", train_end, test_end)
        set_global_seeds(config.training.seed)
        preprocessor = TimeSeriesPreprocessor(
            lookback=config.preprocessing.lookback,
            vol_lookback=config.preprocessing.vol_lookback,
            min_scale_floor=config.preprocessing.min_scale_floor,
            scale_floor_quantile=config.preprocessing.scale_floor_quantile,
            max_train_vol_percentile=config.preprocessing.max_train_vol_percentile,
            target_column=config.preprocessing.target_column,
            date_column=config.preprocessing.date_column,
        )
        cutoff = pd.Timestamp(train_end)
        test_cut = pd.Timestamp(test_end)
        cal, _ = preprocessor.chronological_split(
            prices,
            train_start_date=config.data.train_start_date,
            train_end_date=train_end,
            validation_fraction=config.data.validation_fraction,
        )
        block = prices.loc[
            (prices[config.preprocessing.date_column] > cutoff)
            & (prices[config.preprocessing.date_column] <= test_cut)
        ].copy()
        if cal.empty or block.empty:
            logger.warning("Skipping empty fold %s → %s.", train_end, test_end)
            continue
        x_train, x_val, meta = preprocessor.prepare_calibration(
            prices,
            train_start_date=config.data.train_start_date,
            train_end_date=train_end,
            validation_fraction=config.data.validation_fraction,
            fit_state=True,
        )
        preprocessor.fit_baselines(cal, percentile=config.baseline.percentile)
        autoencoder = LSTMAutoencoder(
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
        autoencoder.build()
        autoencoder.fit(
            x_train,
            x_val,
            epochs=config.walkforward.epochs,
            batch_size=config.training.batch_size,
            patience=config.walkforward.patience,
            shuffle=config.training.shuffle,
            min_delta=config.training.min_delta,
            verbose=0,
        )
        autoencoder.fit_threshold(x_train)
        scored = score_prices(preprocessor, autoencoder, prices)
        test_scores = scored.loc[
            (pd.to_datetime(scored["Date"]) > cutoff)
            & (pd.to_datetime(scored["Date"]) <= test_cut)
        ]
        in_block = events_between(train_end, test_end)
        summary = summarize_detection(test_scores, events=in_block)
        honest = _honest_event_comparison(test_scores, events=in_block)
        rows.append(
            {
                "train_end": train_end,
                "test_end": test_end,
                "n_train_windows": meta["n_train_windows"],
                "threshold": autoencoder.threshold_,
                "n_test_days": int(len(test_scores)),
                "n_flags": summary["n_flags"],
                "flag_rate": summary["flag_rate"],
                "n_episodes": summary["n_episodes"],
                "n_events_in_block": len(in_block),
                "event_recall": (
                    float(sum(r["hit"] for r in (summary.get("events") or [])) / len(in_block))
                    if in_block
                    else None
                ),
                "events_hit": int(sum(r["hit"] for r in (summary.get("events") or []))),
                "lstm_only_events": "|".join(honest.get("lstm_only_events") or []),
                "vol_only_events": "|".join(honest.get("vol_only_events") or []),
                "vol_recall": honest.get("vol_recall"),
            }
        )
        logger.info(
            "Fold %s→%s flag_rate=%.3f recall=%s episodes=%s",
            train_end,
            test_end,
            summary["flag_rate"],
            summary["oos_event_recall"],
            summary["n_episodes"],
        )
    return pd.DataFrame(rows)
