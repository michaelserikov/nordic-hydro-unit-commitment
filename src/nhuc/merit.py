"""Merit-order dispatch of the German thermal fleet (baseline, from de-power-market-ppa v0.4).

Question: given hourly demand for thermal power and monthly fuel and carbon prices, which
lignite, hard-coal and gas plants run, and does the marginal plant's cost reproduce the
observed day-ahead price?

Thermal demand per hour is what load leaves for lignite, hard coal and gas after wind, solar,
hydro, biomass, other renewables, other conventional plants, nuclear (until April 2023), pumped
storage and net imports, all taken from the data. Net imports are the gap between load plus pumped
storage consumption and total domestic generation, so thermal demand equals observed thermal
output; the model decides only how it is split between fuels and what it costs at the margin.
That split is checked against observed generation by fuel, which the model does not see.

Fleet: for each fuel and month, available capacity is the maximum observed hourly output and
must-run output the 5th percentile (combined heat and power, system services, lignite inflexibility).
Capacity above must-run is cut into blocks whose efficiency falls linearly from the best to the
worst plants of that fuel. Marginal cost of a block = (fuel price + emission factor x CO2 price)
/ efficiency + variable O&M. Within an hour the least-cost dispatch is the merit order: must-run
first, then blocks in order of cost until demand is met. The hourly price is the cost of the last
block used, which is the dual of the demand constraint of the equivalent linear program (checked
against SciPy/HiGHS in the tests); if must-run output alone covers demand the price is set to 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

THERMAL = ("lignite", "hard_coal", "gas")


@dataclass(frozen=True)
class FuelTech:
    eta_best: float           # electrical efficiency of the best plants
    eta_worst: float          # ... and of the worst
    ef_t_per_mwh_th: float    # tCO2 per MWh of fuel (thermal)
    vom: float                # variable O&M, EUR per MWh electric


# Assumptions (typical literature ranges, documented in the README; sensitivity in the run).
TECH = {
    "lignite": FuelTech(eta_best=0.43, eta_worst=0.31, ef_t_per_mwh_th=0.364, vom=4.0),
    "hard_coal": FuelTech(eta_best=0.46, eta_worst=0.34, ef_t_per_mwh_th=0.338, vom=4.0),
    "gas": FuelTech(eta_best=0.60, eta_worst=0.33, ef_t_per_mwh_th=0.202, vom=2.0),
}
LIGNITE_FUEL_EUR_MWH = 4.0    # mine-mouth lignite, EUR per MWh thermal (assumption)
N_BLOCKS = 10


def thermal_demand(hourly: pd.DataFrame, fuels: pd.DataFrame) -> pd.DataFrame:
    """Join market and generation data; add net imports and thermal demand (MW = MWh per hour)."""
    df = hourly.merge(fuels, on="ts_utc", how="left").fillna({c: 0.0 for c in fuels.columns if c != "ts_utc"})
    df = df.dropna(subset=["load", "wind_onshore", "wind_offshore", "solar"]).copy()
    other = ["wind_onshore", "wind_offshore", "solar", "hydro", "biomass", "other_ren", "other_conv",
             "nuclear", "pumped_gen"]
    domestic = df[other + list(THERMAL)].sum(axis=1)
    df["net_imports"] = df["load"] + df["pumped_cons"] - domestic
    df["thermal_demand"] = df[list(THERMAL)].sum(axis=1)
    return df


MUST_RUN_Q = 0.05            # must-run output = this quantile of observed hourly output per month


def fleet(df: pd.DataFrame, q: float = MUST_RUN_Q) -> pd.DataFrame:
    """Per local month and fuel: available capacity (max output) and must-run (quantile q of output), MW."""
    g = df.groupby(["year", "month"])
    rows = []
    for (y, m), d in g:
        for f in THERMAL:
            rows.append({"year": y, "month": m, "fuel": f, "capacity": float(d[f].max()),
                         "must_run": float(d[f].quantile(q))})
    return pd.DataFrame(rows)


def block_costs(fuel_prices: pd.Series, tech: dict = TECH, lignite_fuel: float = LIGNITE_FUEL_EUR_MWH,
                n: int = N_BLOCKS) -> dict[str, np.ndarray]:
    """Marginal cost (EUR/MWh el) of n equal blocks per fuel, cheapest (most efficient) first."""
    fuel = {"lignite": lignite_fuel, "hard_coal": fuel_prices["coal_eur_mwh"], "gas": fuel_prices["gas_eur_mwh"]}
    out = {}
    for f, t in tech.items():
        eta = np.linspace(t.eta_best, t.eta_worst, n)
        out[f] = (fuel[f] + t.ef_t_per_mwh_th * fuel_prices["co2_eur_t"]) / eta + t.vom
    return out


def merit_order(demand: float, must_run: dict, blocks: list[tuple[float, str, float]]) -> tuple[float, dict]:
    """Dispatch one hour. blocks = (cost, fuel, size MW) sorted by cost. Returns (price, output by fuel)."""
    out = dict(must_run)
    rest = demand - sum(must_run.values())
    if rest <= 0:
        return 0.0, out
    price = blocks[-1][0]
    for cost, fuel, size in blocks:
        take = min(size, rest)
        out[fuel] += take
        rest -= take
        price = cost
        if rest <= 1e-9:
            break
    return float(price), out


def run(df: pd.DataFrame, fuel_prices: pd.DataFrame, tech: dict = TECH,
        lignite_fuel: float = LIGNITE_FUEL_EUR_MWH, must_run_q: float = MUST_RUN_Q) -> pd.DataFrame:
    """Hourly model price and dispatch by fuel for every row of df (from thermal_demand)."""
    fl = fleet(df, must_run_q).set_index(["year", "month", "fuel"])
    res = np.zeros((len(df), 1 + len(THERMAL)))
    for (y, m), idx in df.groupby(["year", "month"]).groups.items():
        costs = block_costs(fuel_prices.loc[pd.Period(f"{y}-{m:02d}", freq="M")], tech, lignite_fuel)
        must = {f: fl.loc[(y, m, f), "must_run"] for f in THERMAL}
        blocks = sorted((c, f, (fl.loc[(y, m, f), "capacity"] - must[f]) / len(costs[f]))
                        for f in THERMAL for c in costs[f])
        pos = df.index.get_indexer(idx)
        for i, d in zip(pos, df.loc[idx, "thermal_demand"].to_numpy()):
            p, out = merit_order(d, must, blocks)
            res[i] = [p] + [out[f] for f in THERMAL]
    out = pd.DataFrame(res, columns=["model_price"] + [f"model_{f}" for f in THERMAL], index=df.index)
    return pd.concat([df, out], axis=1)


def metrics(res: pd.DataFrame) -> pd.DataFrame:
    """Per year: mean prices, MAE, RMSE, hourly and daily correlation, error on negative-price hours."""
    rows = []
    for y, g in res.groupby("year"):
        e = g["model_price"] - g["price"]
        local_day = g["ts_utc"].dt.tz_convert("Europe/Berlin").dt.date
        daily = g.groupby(local_day)[["price", "model_price"]].mean()
        neg = g["price"] < 0
        rows.append({"year": y, "actual_mean": g["price"].mean(), "model_mean": g["model_price"].mean(),
                     "mae": e.abs().mean(), "rmse": float(np.sqrt((e ** 2).mean())),
                     "corr_hourly": g["price"].corr(g["model_price"]),
                     "corr_daily": daily["price"].corr(daily["model_price"]),
                     "mae_without_negative_hours": e[~neg].abs().mean(),
                     "negative_hours": int(neg.sum()),
                     "hours_above_200": int((g["price"] > 200).sum())})
    return pd.DataFrame(rows)


def generation_check(res: pd.DataFrame) -> pd.DataFrame:
    """Annual TWh by thermal fuel: observed vs model."""
    rows = []
    for y, g in res.groupby("year"):
        for f in THERMAL:
            rows.append({"year": y, "fuel": f, "actual_twh": g[f].sum() / 1e6,
                         "model_twh": g[f"model_{f}"].sum() / 1e6})
    out = pd.DataFrame(rows)
    out["error_pct"] = 100 * (out["model_twh"] / out["actual_twh"] - 1)
    return out
