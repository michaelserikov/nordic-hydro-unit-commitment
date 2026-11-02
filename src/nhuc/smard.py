"""Download hourly time series from SMARD (Bundesnetzagentur), cached on disk.

SMARD publishes each series as weekly JSON chunks:
  index:  https://www.smard.de/app/chart_data/{filter}/{region}/index_hour.json
  chunk:  https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_hour_{ts}.json
Timestamps are milliseconds since the epoch (UTC); values are EUR/MWh for prices and
MWh per hour for generation and load. Data licence: CC BY 4.0, Bundesnetzagentur | SMARD.de.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.smard.de/app/chart_data"
REGION = "DE-LU"

# SMARD filter ids for the series used here.
SERIES = {
    "price": 4169,          # day-ahead price, bidding zone DE-LU, EUR/MWh
    "wind_onshore": 4067,   # generation, MWh
    "wind_offshore": 1225,  # generation, MWh
    "solar": 4068,          # generation (photovoltaics), MWh
    "load": 410,            # grid load, MWh
}

# Generation by source and pumped-storage consumption (v0.4 dispatch model), MWh per hour.
FUEL_SERIES = {
    "lignite": 1223,
    "nuclear": 1224,        # ends 15 April 2023
    "hydro": 1226,
    "other_conv": 1227,     # oil, waste and other conventional
    "other_ren": 1228,
    "biomass": 4066,
    "hard_coal": 4069,
    "gas": 4071,
    "pumped_gen": 4070,
    "pumped_cons": 4387,
}

_session = requests.Session()
_session.headers["User-Agent"] = "nordic-hydro-unit-commitment (research; github.com/michaelserikov/nordic-hydro-unit-commitment)"


def _get_json(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        r = _session.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"failed: {url}")


def chunk_starts(filter_id: int, region: str = REGION) -> list[int]:
    """Start timestamps (ms, UTC) of all weekly chunks of a series."""
    return _get_json(f"{BASE}/{filter_id}/{region}/index_hour.json")["timestamps"]


def parse_chunk(doc: dict) -> pd.DataFrame:
    """One chunk document to a frame with columns ts_utc (tz-aware) and value; nulls dropped."""
    rows = [(t, v) for t, v in doc.get("series", []) if v is not None]
    df = pd.DataFrame(rows, columns=["ts_ms", "value"])
    df["ts_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["ts_utc", "value"]]


def load_series(name: str, start: str, end: str, cache_dir: Path, region: str = REGION) -> pd.DataFrame:
    """Hourly series `name` between start and end (inclusive dates, UTC), from cache or SMARD.

    A chunk is cached once it lies fully in the past, so a rerun downloads nothing new.
    """
    filter_id = {**SERIES, **FUEL_SERIES, **FORECAST_SERIES}[name]
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    week = pd.Timedelta(days=7)
    folder = cache_dir / f"{filter_id}_{region}"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for ts in chunk_starts(filter_id, region):
        t0 = pd.Timestamp(ts, unit="ms", tz="UTC")
        if t0 + week <= start_ts or t0 >= end_ts:
            continue
        path = folder / f"{ts}.json"
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
        else:
            doc = _get_json(f"{BASE}/{filter_id}/{region}/{filter_id}_{region}_hour_{ts}.json")
            if t0 + week < pd.Timestamp.now(tz="UTC"):
                path.write_text(json.dumps(doc), encoding="utf-8")
            time.sleep(0.2)  # keep the request rate low
        frames.append(parse_chunk(doc))
    if not frames:                  # series with no data in the window (nuclear after April 2023)
        return pd.DataFrame({"ts_utc": pd.Series(dtype="datetime64[ns, UTC]"), name: pd.Series(dtype=float)})
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts_utc")
    df = df[(df["ts_utc"] >= start_ts) & (df["ts_utc"] < end_ts)]
    return df.rename(columns={"value": name}).sort_values("ts_utc").reset_index(drop=True)


def load_all(start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """All series joined on the hour; every row has a price, other series may be missing."""
    out = load_series("price", start, end, cache_dir)
    for name in SERIES:
        if name != "price":
            out = out.merge(load_series(name, start, end, cache_dir), on="ts_utc", how="left")
    return out


# Day-ahead forecasts published by the TSOs (v0.5 price forecasting), MWh per hour.
FORECAST_SERIES = {
    "load_fc": 411,
    "onshore_fc": 123,
    "offshore_fc": 3791,
    "solar_fc": 125,
}


def load_forecasts(start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """TSO day-ahead forecasts (FORECAST_SERIES) joined on the hour; missing hours stay NaN."""
    out = None
    for name in FORECAST_SERIES:
        df = load_series(name, start, end, cache_dir)
        out = df if out is None else out.merge(df, on="ts_utc", how="outer")
    return out.sort_values("ts_utc").reset_index(drop=True)


def load_fuels(start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """Generation by source (FUEL_SERIES) joined on the hour; a missing hour of a source counts as zero."""
    out = None
    for name in FUEL_SERIES:
        df = load_series(name, start, end, cache_dir)
        out = df if out is None else out.merge(df, on="ts_utc", how="outer")
    return out.sort_values("ts_utc").fillna(0.0).reset_index(drop=True)
