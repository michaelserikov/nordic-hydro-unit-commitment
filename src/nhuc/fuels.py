"""Monthly fuel and carbon prices in EUR, from open sources, cached on disk.

- Natural gas, Europe (TTF) and Coal, South African (6000 kcal/kg, the closest open series to the
  API2 benchmark) from the World Bank Commodity Price Data ("Pink Sheet"), monthly, USD.
- EUR/USD monthly average reference rate from the ECB data API.
- EU emission allowances: clearing prices of the EEX primary auctions (daily), averaged by month.

Raw files are downloaded into data/raw/fuels/ and not committed; the derived monthly table is
written to results/. Conversions: 1 MWh = 3.412 MMBtu; coal 6000 kcal/kg = 25.12 GJ/t = 6.978 MWh/t.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

WB_URL = ("https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/"
          "related/CMO-Historical-Data-Monthly.xlsx")
ECB_URL = ("https://data-api.ecb.europa.eu/service/data/EXR/M.USD.EUR.SP00.A"
           "?format=csvdata&startPeriod={start}&endPeriod={end}")
EEX_URL = ("https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/"
           "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
           "emission-spot-primary-market-auction-report-2012-2025-data.zip")

MMBTU_PER_MWH = 3.412
COAL_MWH_PER_T = 6.978

_HEADERS = {"User-Agent": "nordic-hydro-unit-commitment (research; github.com/michaelserikov/nordic-hydro-unit-commitment)"}


def _cached(url: str, path: Path) -> bytes:
    if not path.exists():
        r = requests.get(url, headers=_HEADERS, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return path.read_bytes()


def world_bank(cache: Path) -> pd.DataFrame:
    """Monthly TTF gas (USD/MMBtu) and South African coal (USD/t)."""
    raw = pd.read_excel(io.BytesIO(_cached(WB_URL, cache / "wb_pink_sheet_monthly.xlsx")),
                        sheet_name="Monthly Prices", header=None)
    header = raw.iloc[4].tolist()
    gas_col = header.index("Natural gas, Europe")
    coal_col = next(i for i, h in enumerate(header) if isinstance(h, str) and h.startswith("Coal, South African"))
    rows = raw.iloc[6:, [0, gas_col, coal_col]]
    rows.columns = ["period", "gas_usd_mmbtu", "coal_usd_t"]
    rows = rows[rows["period"].astype(str).str.match(r"^\d{4}M\d{2}$")].copy()
    for c in ("gas_usd_mmbtu", "coal_usd_t"):  # the sheet marks missing months with a placeholder
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    rows["month"] = pd.PeriodIndex(rows["period"].str.replace("M", "-"), freq="M")
    return rows.set_index("month")[["gas_usd_mmbtu", "coal_usd_t"]]


def eur_usd(cache: Path, start: str, end: str) -> pd.Series:
    """Monthly average USD per EUR (ECB reference rate)."""
    raw = _cached(ECB_URL.format(start=start, end=end), cache / f"ecb_usd_eur_{start}_{end}.csv")
    df = pd.read_csv(io.BytesIO(raw))
    return pd.Series(df["OBS_VALUE"].to_numpy(float), index=pd.PeriodIndex(df["TIME_PERIOD"], freq="M"),
                     name="usd_per_eur")


def eua_auctions(cache: Path, years: range) -> pd.DataFrame:
    """Daily EEX primary-auction clearing prices of EU allowances (EUR/t)."""
    z = zipfile.ZipFile(io.BytesIO(_cached(EEX_URL, cache / "eex_eua_auctions_2012_2025.zip")))
    frames = []
    for year in years:
        name = next(n for n in z.namelist() if n.endswith(f"-{year}-data.xlsx"))
        raw = pd.read_excel(io.BytesIO(z.read(name)), header=None)
        header_row = next(i for i, v in enumerate(raw.iloc[:, 1]) if str(v).strip() == "Date")
        cols = [str(c).strip() for c in raw.iloc[header_row]]
        price_col = next(i for i, c in enumerate(cols) if c.startswith("Auction Price"))
        status_col = cols.index("Status")
        df = raw.iloc[header_row + 1:, [1, price_col, status_col]]
        df.columns = ["date", "price", "status"]
        df = df[[str(v).strip().lower() == "successful" for v in df["status"]]]
        frames.append(df[["date", "price"]])
    out = pd.concat(frames)
    out["date"] = pd.to_datetime(out["date"])
    out["price"] = out["price"].astype(float)
    return out.sort_values("date").reset_index(drop=True)


def monthly_prices(cache: Path, start: str = "2023-01", end: str = "2025-12") -> pd.DataFrame:
    """Monthly gas and coal (EUR per MWh thermal) and EUA (EUR/t) from start to end."""
    wb = world_bank(cache)
    fx = eur_usd(cache, start, end)
    years = range(int(start[:4]), int(end[:4]) + 1)
    eua = eua_auctions(cache, years)
    eua_m = eua.groupby(eua["date"].dt.to_period("M"))["price"].mean().rename("co2_eur_t")
    months = pd.period_range(start, end, freq="M")
    df = wb.reindex(months).join(fx).join(eua_m)
    df["gas_eur_mwh"] = df["gas_usd_mmbtu"] / df["usd_per_eur"] * MMBTU_PER_MWH
    df["coal_eur_mwh"] = df["coal_usd_t"] / df["usd_per_eur"] / COAL_MWH_PER_T
    if df[["gas_eur_mwh", "coal_eur_mwh", "co2_eur_t"]].isna().any().any():
        raise ValueError("missing fuel or carbon prices for some months")
    df.index.name = "month"
    return df
