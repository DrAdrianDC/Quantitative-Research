"""Append-only score ledger. Historical rows are a published record.

Weekly jobs may add dates after the cutoff. They may not recompute 2008
or 2020 because the weights changed. That is a model-risk contract, not
a convenience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

LEDGER_COLUMNS = (
    "Date",
    "Close",
    "price_change",
    "local_scale",
    "normalized_return",
    "reconstruction_mse",
    "magnitude",
    "anomaly",
    "return_zscore",
    "zscore_anomaly",
    "rolling_vol",
    "vol_anomaly",
)

REQUIRED_COLUMNS = ("Date", "reconstruction_mse", "anomaly")


class LedgerError(RuntimeError):
    """Raised when the published score history cannot be trusted."""


def load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise LedgerError(
            f"Score ledger missing at {path}. Run --mode train once to create it."
        )
    ledger = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(ledger.columns)
    if missing:
        raise LedgerError(f"Score ledger is missing columns: {sorted(missing)}")
    ledger["Date"] = pd.to_datetime(ledger["Date"]).dt.strftime("%Y-%m-%d")
    ledger = ledger.drop_duplicates(subset=["Date"], keep="first")
    ledger = ledger.sort_values("Date").reset_index(drop=True)
    if ledger.empty:
        raise LedgerError("Score ledger is empty.")
    return ledger


def assert_event_integrity(ledger: pd.DataFrame, event_dates: Sequence[str]) -> None:
    """Stress prints in the published record must stay present and flagged."""
    if not event_dates:
        return
    indexed = ledger.set_index("Date")
    missing = [day for day in event_dates if day not in indexed.index]
    dropped = [
        day
        for day in event_dates
        if day in indexed.index and int(indexed.loc[day, "anomaly"]) != 1
    ]
    if missing or dropped:
        parts = []
        if missing:
            parts.append(f"missing from ledger {missing}")
        if dropped:
            parts.append(f"unflagged in ledger {dropped}")
        raise LedgerError(
            "Score ledger failed historical event integrity (" + "; ".join(parts) + ")."
        )


def append_new_rows(ledger: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Keep the first copy of every date. New dates come from ``scored``."""
    last_scored = pd.to_datetime(ledger["Date"]).max()
    incoming = scored.copy()
    incoming["Date"] = pd.to_datetime(incoming["Date"]).dt.strftime("%Y-%m-%d")
    new_rows = incoming.loc[pd.to_datetime(incoming["Date"]) > last_scored].copy()
    if new_rows.empty:
        return ledger.copy()
    combined = pd.concat([ledger, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date"], keep="first")
    return combined.sort_values("Date").reset_index(drop=True)


def write_ledger(path: Path, scores: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = scores.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"]).dt.strftime("%Y-%m-%d")
    ordered = ordered.sort_values("Date").reset_index(drop=True)
    ordered.to_csv(path, index=False)
    return ordered


def ledger_stats(scores: pd.DataFrame) -> dict[str, Any]:
    ordered = scores.sort_values("Date") if "Date" in scores.columns else scores
    n = len(ordered)
    n_flags = int(ordered["anomaly"].astype(int).sum()) if n else 0
    stats: dict[str, Any] = {
        "n_windows_scored": int(n),
        "n_anomalies": n_flags,
        "anomaly_rate": float(n_flags / n) if n else 0.0,
    }
    if n:
        stats["first_date"] = str(ordered["Date"].iloc[0])
        stats["last_date"] = str(ordered["Date"].iloc[-1])
    flagged = ordered.loc[ordered["anomaly"].astype(int) == 1]
    if not flagged.empty:
        stats["first_anomaly_date"] = str(flagged["Date"].iloc[0])
        stats["last_anomaly_date"] = str(flagged["Date"].iloc[-1])
        peak_i = flagged["reconstruction_mse"].astype(float).idxmax()
        stats["max_anomaly_mse"] = float(flagged.loc[peak_i, "reconstruction_mse"])
        stats["max_anomaly_date"] = str(flagged.loc[peak_i, "Date"])
    return stats
