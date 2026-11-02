"""Weekly data for the aggregate Norwegian hydro reservoir, cached in data/raw.

- Reservoir filling (TWh) and capacity per week, Norway total and per price area NO1-NO5: NVE
  reservoir statistics (magasinstatistikk), open API, 1995 onwards.
- Hourly generation of hydro reservoir plants in Norway and day-ahead prices of NO1-NO5:
  Energy-Charts API (Fraunhofer ISE), based on ENTSO-E transparency data, 2015 onwards.

Weekly inflow to the reservoirs is not published as such; it is implied by the water balance
    inflow[w] = filling[w] - filling[w-1] + reservoir generation[w]
which ignores spillage, evaporation and the water used by pumping, so it is energy that entered the
reservoirs net of those. Weekly price: average of the NO1-NO5 prices weighted by each area's
reservoir capacity.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

NVE_URL = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"
EC_URL = "https://api.energy-charts.info"
AREAS = ("NO1", "NO2", "NO3", "NO4", "NO5")
_HEADERS = {"User-Agent": "nordic-hydro-unit-commitment (research; github.com/michaelserikov/nordic-hydro-unit-commitment)"}


def _get(url: str, path: Path, retries: int = 6) -> object:
    """Download once and cache; the Energy-Charts API is rate limited, so wait and retry on 429."""
    if not path.exists():
        for attempt in range(retries):
            r = requests.get(url, headers=_HEADERS, timeout=180)
            if r.status_code == 429 and attempt < retries - 1:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
        time.sleep(3)
    return json.loads(path.read_text(encoding="utf-8"))


def reservoir(raw: Path) -> pd.DataFrame:
    """Weekly NVE statistics: ISO year/week, area ('NO' or NO1..NO5), filling and capacity in TWh."""
    d = pd.DataFrame(_get(NVE_URL, raw / "nve_magasinstatistikk.json"))
    d = d[d["omrType"].isin(["NO", "EL"])].copy()
    d["area"] = np.where(d["omrType"] == "NO", "NO", "NO" + d["omrnr"].astype(str))
    d["date"] = pd.to_datetime(d["dato_Id"])
    return (d.rename(columns={"iso_aar": "year", "iso_uke": "week", "fylling_TWh": "fill_twh",
                              "kapasitet_TWh": "capacity_twh"})
            [["area", "year", "week", "date", "fill_twh", "capacity_twh"]]
            .sort_values(["area", "date"]).reset_index(drop=True))


def _ec_year(kind: str, code: str, year: int, raw: Path) -> pd.DataFrame:
    q = "price?bzn=" if kind == "price" else "public_power?country="
    url = f"{EC_URL}/{q}{code}&start={year}-01-01&end={year + 1}-01-01"
    doc = _get(url, raw / "energy_charts" / f"{kind}_{code}_{year}.json")
    ts = pd.to_datetime(doc["unix_seconds"], unit="s", utc=True)
    if kind == "price":
        return pd.DataFrame({"ts_utc": ts, "value": pd.to_numeric(pd.Series(doc["price"]), errors="coerce")})
    pt = {p["name"]: p["data"] for p in doc["production_types"]}
    return pd.DataFrame({"ts_utc": ts, "value": pd.to_numeric(pd.Series(pt["Hydro water reservoir"]), errors="coerce")})


def weekly_series(years: range, raw: Path) -> pd.DataFrame:
    """Per ISO week: reservoir generation in Norway (TWh) and capacity-weighted NO price (EUR/MWh)."""
    res = reservoir(raw)
    cap = res[res["area"].isin(AREAS)].groupby("area")["capacity_twh"].last()
    w = (cap / cap.sum()).to_dict()
    gen, price = [], []
    for y in years:
        g = _ec_year("power", "no", y, raw)
        gen.append(g)
        p = None
        for a in AREAS:
            s = _ec_year("price", a, y, raw).rename(columns={"value": a})
            p = s if p is None else p.merge(s, on="ts_utc", how="outer")
        price.append(p)
    gen = pd.concat(gen).drop_duplicates("ts_utc").set_index("ts_utc").sort_index()
    price = pd.concat(price).drop_duplicates("ts_utc").set_index("ts_utc").sort_index()
    # energy: average MW in each hour x 1 h; works for hourly and 15-minute data alike
    gen_h = gen["value"].resample("1h").mean()
    price["no"] = sum(price[a] * w[a] for a in AREAS) / sum(w[a] for a in AREAS)
    price_h = price["no"].resample("1h").mean()
    # NVE weeks end on Sunday (dato_Id is the Sunday); aggregate Monday-Sunday in Oslo time
    def iso(s: pd.Series) -> pd.DataFrame:
        loc = s.index.tz_convert("Europe/Oslo")
        cal = loc.isocalendar()
        return pd.DataFrame({"year": cal.year.to_numpy(), "week": cal.week.to_numpy(), "v": s.to_numpy()})
    g = iso(gen_h).groupby(["year", "week"]).agg(gen_twh=("v", lambda v: v.sum() / 1e6), hours=("v", "count"))
    p = iso(price_h).groupby(["year", "week"]).agg(price=("v", "mean"))
    return g.join(p).reset_index()


def weekly_table(years: range, raw: Path) -> pd.DataFrame:
    """Joined weekly table for Norway: filling, capacity, generation, implied inflow and price."""
    res = reservoir(raw)
    no = res[res["area"] == "NO"].copy()
    no["fill_prev"] = no["fill_twh"].shift(1)
    ws = weekly_series(years, raw)
    t = no.merge(ws, on=["year", "week"], how="inner")
    t = t[t["hours"] >= 160]                          # complete weeks only
    t["inflow_twh"] = t["fill_twh"] - t["fill_prev"] + t["gen_twh"]
    return t.dropna(subset=["fill_prev", "price"]).reset_index(drop=True)
