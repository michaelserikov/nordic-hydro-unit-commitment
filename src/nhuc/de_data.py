"""German hourly input table for the unit commitment model (SMARD, cached in data/raw)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import merit, smard

TZ = "Europe/Berlin"


def load_hourly(start: str, end: str, raw: Path) -> pd.DataFrame:
    """Price, load, wind and solar for whole local calendar years start..end, with local year/month/day."""
    # Local midnight on 1 January is 23:00 UTC the day before: download one extra day.
    first_day = (pd.Timestamp(start) - pd.Timedelta(days=1)).date().isoformat()
    df = smard.load_all(first_day, end, raw)
    local = df["ts_utc"].dt.tz_convert(TZ)
    df["year"] = local.dt.year.astype("int32")
    df["month"] = local.dt.month.astype("int32")
    df["day"] = local.dt.date
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    return df[df["year"].isin(years)].reset_index(drop=True)


def thermal_table(start: str, end: str, raw: Path) -> pd.DataFrame:
    """Hourly table with generation by fuel, net imports and thermal demand (as in the merit-order model)."""
    hourly = load_hourly(start, end, raw)
    first_day = (pd.Timestamp(start) - pd.Timedelta(days=1)).date().isoformat()
    gen = smard.load_fuels(first_day, end, raw)
    return merit.thermal_demand(hourly, gen).reset_index(drop=True)
