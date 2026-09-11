"""Pre-registered WTI stress episodes and episode-level metrics.

The catalog is the acceptance set. It was written from market history
(GFC, OPEC decisions, the April 2020 negative print, the 2022 invasion
spike) — not by inspecting reconstruction errors. Adding a date because
the model flagged it is circular and is not allowed.

A monitor is judged at **episode** grain, not day grain. Windowed
autoencoders ring for ~lookback days after a shock; a 10-day streak is
one episode, not ten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketEvent:
    date: str
    name: str
    era: str  # "pre" (historical holdout), "calibration", or "oos"

    def timestamp(self) -> pd.Timestamp:
        return pd.Timestamp(self.date)


# Chosen from public market history before scores were computed under the
# vol-normalized contract.
#   pre          — before train_start (2010-01-01). Historical holdout; not
#                  in the loss. Not the reported OOS recall.
#   calibration  — inside the 2010–2019 quiet-window train set.
#   oos          — after train_end (2019-12-31). Reported recall.
EVENT_CATALOG: tuple[MarketEvent, ...] = (
    MarketEvent("2001-09-17", "9/11 session reopen", "pre"),
    MarketEvent("2003-03-20", "Iraq invasion", "pre"),
    MarketEvent("2008-07-11", "WTI cycle high (~$147)", "pre"),
    MarketEvent("2008-10-10", "GFC crash week", "pre"),
    MarketEvent("2008-12-19", "GFC WTI trough", "pre"),
    MarketEvent("2011-05-05", "2011 risk-off dump", "calibration"),
    MarketEvent("2014-11-28", "OPEC Thanksgiving no-cut", "calibration"),
    MarketEvent("2015-08-24", "2015 China deval / risk-off", "calibration"),
    MarketEvent("2016-01-20", "2014–16 glut cycle low", "calibration"),
    MarketEvent("2016-02-11", "WTI sub-$30 trough", "calibration"),
    MarketEvent("2018-11-13", "Q4-18 inventory dump", "calibration"),
    MarketEvent("2018-12-24", "Q4-18 Christmas selloff", "calibration"),
    MarketEvent("2020-03-09", "OPEC+ price war", "oos"),
    MarketEvent("2020-03-18", "COVID demand collapse", "oos"),
    MarketEvent("2020-04-20", "WTI negative settlement", "oos"),
    MarketEvent("2020-04-21", "Post-negative bounce", "oos"),
    MarketEvent("2021-11-26", "Omicron risk-off", "oos"),
    MarketEvent("2022-03-08", "Russia–Ukraine spike", "oos"),
    MarketEvent("2022-03-09", "Post-invasion continuation", "oos"),
    MarketEvent("2022-06-17", "2022 mid-year liquidation", "oos"),
    MarketEvent("2023-04-03", "OPEC+ surprise cut", "oos"),
    MarketEvent("2023-10-09", "Israel–Hamas oil spike", "oos"),
)


def events_between(
    start: str,
    end: str,
    catalog: Sequence[MarketEvent] = EVENT_CATALOG,
) -> tuple[MarketEvent, ...]:
    """Events whose date sits in ``(start, end]``. Used by walk-forward so a
    2008–2011 test block is not scored against 2020 catalog entries.
    """
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return tuple(event for event in catalog if lo < event.timestamp() <= hi)


def events_for_era(era: str, catalog: Sequence[MarketEvent] = EVENT_CATALOG) -> tuple[MarketEvent, ...]:
    return tuple(event for event in catalog if event.era == era)


def cluster_episodes(
    dates: Sequence,
    flags: Sequence,
    *,
    max_gap_days: int = 1,
) -> pd.DataFrame:
    """Collapse 0/1 flags into episodes.

    ``max_gap_days`` is the number of *unflagged trading rows* allowed
    between two flags of the same episode. ``1`` merges a single quiet
    session (the ringing of one shock) without gluing unrelated months.
    """
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(list(dates)),
            "flag": np.asarray(flags, dtype=int),
        }
    ).sort_values("Date").reset_index(drop=True)
    flagged_pos = frame.index[frame["flag"] == 1].to_numpy()
    if flagged_pos.size == 0:
        return pd.DataFrame(columns=["start", "end", "n_days"])

    chunks: list[tuple[int, int]] = []
    start = prev = int(flagged_pos[0])
    for pos in flagged_pos[1:]:
        pos = int(pos)
        gap = pos - prev - 1
        if gap <= max_gap_days:
            prev = pos
        else:
            chunks.append((start, prev))
            start = prev = pos
    chunks.append((start, prev))

    rows = []
    for start, end in chunks:
        sl = frame.iloc[start : end + 1]
        rows.append(
            {
                "start": sl["Date"].iloc[0],
                "end": sl["Date"].iloc[-1],
                "n_days": int((sl["flag"] == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def event_hit_mask(
    scores: pd.DataFrame,
    events: Iterable[MarketEvent],
    *,
    flag_column: str = "anomaly",
    pre_days: int = 2,
    post_days: int = 5,
) -> pd.DataFrame:
    """One row per event: flag in the trading-day neighbourhood of the date.

    ``pre_days`` / ``post_days`` are in scored trading rows around the
    nearest session, not calendar days. A weekend-listed event still
    matches the Friday/Monday tape.
    """
    if scores.empty:
        return pd.DataFrame()
    ordered = scores.copy()
    ordered["Date"] = pd.to_datetime(ordered["Date"])
    ordered = ordered.sort_values("Date").reset_index(drop=True)
    rows = []
    for event in events:
        center = event.timestamp()
        offsets = (ordered["Date"] - center).abs()
        pos = int(offsets.to_numpy().argmin())
        lo_pos = max(0, pos - int(pre_days))
        hi_pos = min(len(ordered) - 1, pos + int(post_days))
        window = ordered.iloc[lo_pos : hi_pos + 1]
        hit = bool((window[flag_column].astype(int) == 1).any())
        rows.append(
            {
                "date": event.date,
                "name": event.name,
                "era": event.era,
                "nearest_trading_day": ordered.loc[pos, "Date"].strftime("%Y-%m-%d"),
                "hit": int(hit),
            }
        )
    return pd.DataFrame(rows)


def summarize_detection(
    scores: pd.DataFrame,
    *,
    flag_column: str = "anomaly",
    mse_column: str = "reconstruction_mse",
    events: Sequence[MarketEvent] = EVENT_CATALOG,
    max_gap_days: int = 1,
) -> dict[str, Any]:
    """Episode counts + recall by era. Unlabeled episodes are not precision."""
    hits = event_hit_mask(scores, events, flag_column=flag_column)
    episodes = cluster_episodes(scores["Date"], scores[flag_column], max_gap_days=max_gap_days)
    oos = hits.loc[hits["era"] == "oos"] if not hits.empty else hits
    cal = hits.loc[hits["era"] == "calibration"] if not hits.empty else hits
    summary: dict[str, Any] = {
        "n_days": int(len(scores)),
        "n_flags": int(scores[flag_column].astype(int).sum()),
        "flag_rate": float(scores[flag_column].astype(int).mean()) if len(scores) else 0.0,
        "n_episodes": int(len(episodes)),
        "median_episode_length": float(episodes["n_days"].median()) if len(episodes) else 0.0,
        "max_episode_length": int(episodes["n_days"].max()) if len(episodes) else 0,
        "oos_event_recall": float(oos["hit"].mean()) if len(oos) else None,
        "oos_events_hit": int(oos["hit"].sum()) if len(oos) else 0,
        "oos_events_total": int(len(oos)),
        "calibration_event_recall": float(cal["hit"].mean()) if len(cal) else None,
        "calibration_events_hit": int(cal["hit"].sum()) if len(cal) else 0,
        "calibration_events_total": int(len(cal)),
        "events": hits.to_dict(orient="records") if not hits.empty else [],
    }
    if mse_column in scores.columns and scores[flag_column].astype(int).any():
        flagged = scores.loc[scores[flag_column].astype(int) == 1]
        peak_i = flagged[mse_column].astype(float).idxmax()
        summary["peak_mse_date"] = str(pd.Timestamp(scores.loc[peak_i, "Date"]).date())
        summary["peak_mse"] = float(scores.loc[peak_i, mse_column])
    return summary
