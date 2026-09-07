"""Robust download and validation of WTI futures prices from Yahoo Finance."""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_TICKER_PATTERN = re.compile(r"^[A-Za-z0-9=.\-]{1,20}$")
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class DataValidationError(ValueError):
    """Raised when downloaded market data fails integrity checks."""


class DataDownloadError(RuntimeError):
    """Raised when yfinance cannot be reached after retries."""


def _parse_optional_date(value: str | date | datetime | None, field: str) -> str | None:
    """Normalize a date-like value to ``YYYY-MM-DD`` or reject it."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{field} must be a non-empty ISO date string.")
    candidate = value.strip()
    if not _ISO_DATE_PATTERN.match(candidate):
        raise DataValidationError(
            f"{field}='{value}' is not a valid ISO date (expected YYYY-MM-DD)."
        )
    try:
        datetime.strptime(candidate, "%Y-%m-%d")
    except ValueError as exc:
        raise DataValidationError(f"{field}='{value}' is not a real calendar date.") from exc
    return candidate


def _validate_ticker(ticker: str) -> str:
    if not isinstance(ticker, str) or not ticker.strip():
        raise DataValidationError("ticker must be a non-empty string.")
    cleaned = ticker.strip()
    if not _TICKER_PATTERN.match(cleaned):
        raise DataValidationError(
            f"ticker='{ticker}' contains invalid characters. "
            "Expected a Yahoo Finance symbol such as 'CL=F'."
        )
    return cleaned


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """Return the Close series from both flat and MultiIndex yfinance frames."""
    if raw.empty:
        raise DataValidationError(f"yfinance returned an empty frame for ticker '{ticker}'.")

    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = raw.columns.get_level_values(0)
        if "Close" not in level_zero:
            raise DataValidationError(
                f"Downloaded frame for '{ticker}' has no 'Close' column."
            )
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            if ticker in close.columns:
                return close[ticker]
            return close.iloc[:, 0]
        return close

    if "Close" in raw.columns:
        return raw["Close"]

    raise DataValidationError(
        f"Could not locate a Close column in the yfinance payload for '{ticker}'."
    )


def _download_once(
    ticker: str,
    *,
    start: str | None,
    end: str | None,
    period: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    kwargs: dict[str, Any] = {
        "auto_adjust": auto_adjust,
        "progress": False,
        "threads": False,
    }
    if start is not None:
        kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
    else:
        kwargs["period"] = period

    logger.info(
        "Downloading %s from yfinance (start=%s, end=%s, period=%s).",
        ticker,
        start,
        end,
        period if start is None else None,
    )
    raw = yf.download(ticker, **kwargs)
    if raw is None or raw.empty:
        raise DataDownloadError(f"No rows returned for ticker '{ticker}'.")
    return raw


def download_wti_prices(
    ticker: str = "CL=F",
    *,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    period: str = "max",
    auto_adjust: bool = False,
    max_null_ratio: float = 0.05,
    retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> pd.DataFrame:
    """Download daily WTI (or any Yahoo) close prices with validation.

    Parameters
    ----------
    ticker:
        Yahoo Finance symbol. Production default is ``CL=F`` (WTI futures).
    start, end:
        Optional ISO dates. When ``start`` is set, ``period`` is ignored.
    period:
        yfinance period string used for a full-history pull.
    auto_adjust:
        Forwarded to yfinance. The research notebook used ``False``.
    max_null_ratio:
        Maximum allowed fraction of missing Close values after download.
    retries:
        Number of download attempts on transient network/API failures.
    retry_backoff_seconds:
        Base backoff; wait time doubles after each failed attempt.

    Returns
    -------
    pandas.DataFrame
        Two columns: ``Date`` (timezone-naive datetime64) and ``Close``
        (float64), sorted ascending and de-duplicated.

    Raises
    ------
    DataValidationError
        Invalid arguments or structurally broken market data.
    DataDownloadError
        yfinance failed after the configured number of retries.
    """
    ticker = _validate_ticker(ticker)
    start_iso = _parse_optional_date(start, "start")
    end_iso = _parse_optional_date(end, "end")

    if start_iso and end_iso and start_iso > end_iso:
        raise DataValidationError(
            f"start ({start_iso}) must be on or before end ({end_iso})."
        )
    if retries < 1:
        raise DataValidationError("retries must be >= 1.")
    if not 0.0 <= max_null_ratio <= 1.0:
        raise DataValidationError("max_null_ratio must be within [0, 1].")

    last_error: Exception | None = None
    raw: pd.DataFrame | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = _download_once(
                ticker,
                start=start_iso,
                end=end_iso,
                period=period,
                auto_adjust=auto_adjust,
            )
            break
        except (DataDownloadError, DataValidationError):
            raise
        except Exception as exc:  # yfinance raises a mix of OSError/JSON errors
            last_error = exc
            logger.warning(
                "yfinance download failed (attempt %s/%s): %s",
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))

    if raw is None:
        raise DataDownloadError(
            f"Failed to download '{ticker}' after {retries} attempts."
        ) from last_error

    close = _extract_close(raw, ticker)
    frame = close.to_frame(name="Close").reset_index()
    date_col = frame.columns[0]
    frame = frame.rename(columns={date_col: "Date"})

    frame["Date"] = pd.to_datetime(frame["Date"], utc=True, errors="coerce")
    if frame["Date"].isna().any():
        bad = int(frame["Date"].isna().sum())
        raise DataValidationError(f"{bad} rows have an unparseable Date value.")
    frame["Date"] = frame["Date"].dt.tz_convert(None).dt.normalize()

    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    null_ratio = float(frame["Close"].isna().mean())
    if null_ratio > max_null_ratio:
        raise DataValidationError(
            f"Close null ratio {null_ratio:.2%} exceeds max_null_ratio "
            f"{max_null_ratio:.2%} for ticker '{ticker}'."
        )

    dropped_nulls = int(frame["Close"].isna().sum())
    frame = frame.dropna(subset=["Close"]).drop_duplicates(subset=["Date"], keep="last")
    frame = frame.sort_values("Date").reset_index(drop=True)

    if frame.empty:
        raise DataValidationError(f"No usable Close observations remain for '{ticker}'.")

    logger.info(
        "Loaded %s rows for %s from %s to %s (dropped %s null Close values).",
        len(frame),
        ticker,
        frame["Date"].iloc[0].date(),
        frame["Date"].iloc[-1].date(),
        dropped_nulls,
    )
    return frame[["Date", "Close"]]


def last_available_date(frame: pd.DataFrame) -> pd.Timestamp:
    """Return the most recent observation date in a validated price frame."""
    if frame.empty or "Date" not in frame.columns:
        raise DataValidationError("Price frame is empty or missing a Date column.")
    return pd.Timestamp(frame["Date"].max())
