"""Run the unit commitment model on 2023-2025 and compare it with the merit-order model.

    python -m nhuc.uc_run            # data from SMARD, World Bank, ECB and EEX (cached in data/raw)

Inputs shared with the merit-order model (de-power-market-ppa v0.4): lignite fuel cost 4 EUR/MWh
thermal and must-run quantile 0.15, both calibrated there on 2023. The one new parameter, the cost
of absorbing surplus inflexible output (dump cost, the negative price in such hours), is
calibrated on 2023 by the hourly MAE of the relaxed price; 2024 and 2025 are out of sample.
Years are solved in parallel, each with its own rolling horizon. Writes results/uc/ and figures/.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import de_data, fuels, merit, plots, uc

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
LIGNITE_FUEL = 4.0
MUST_RUN_Q = 0.15
DUMP_GRID = (0.0, 10.0, 25.0, 50.0)
CAL_YEAR = 2023
YEARS = (2023, 2024, 2025)


def _inputs(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = de_data.thermal_table(f"{year}-01-01", f"{year}-12-31", RAW)
    fp = fuels.monthly_prices(RAW / "fuels", "2023-01", "2025-12")
    return df, fp


def variant_units(name: str) -> dict:
    if name == "base":
        return uc.UNITS
    if name == "no_dynamics":     # minimum output only: no start-up cost, no minimum up/down time
        return {k: replace(u, startup_eur_per_mw=0.0, min_up=1, min_down=1) for k, u in uc.UNITS.items()}
    if name == "startup_x2":
        return {k: replace(u, startup_eur_per_mw=2 * u.startup_eur_per_mw) for k, u in uc.UNITS.items()}
    raise ValueError(name)


def job(year: int, dump_cost: float, variant: str = "base") -> str:
    tag = f"{year}_dump{dump_cost:g}_{variant}"
    path = CACHE / f"uc_{tag}.parquet"
    if not path.exists():
        df, fp = _inputs(year)
        res = uc.run(df, fp, must_run_q=MUST_RUN_Q, dump_cost=dump_cost, lignite_fuel=LIGNITE_FUEL,
                     units=variant_units(variant))
        CACHE.mkdir(parents=True, exist_ok=True)
        res.drop(columns=["day"]).assign(day=res["day"].astype(str)).to_parquet(path)
    return str(path)


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


def main() -> dict:
    out, figures = ROOT / "results" / "uc", ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)

    with ProcessPoolExecutor(max_workers=4) as ex:
        cal_paths = dict(zip(DUMP_GRID, ex.map(job, [CAL_YEAR] * len(DUMP_GRID), DUMP_GRID)))
    cal = []
    for dc, p in cal_paths.items():
        r = _load(p)
        for col in ("uc_price_relaxed", "uc_price_restricted"):
            m = uc.metrics(r, col).iloc[0]
            cal.append({"dump_cost": dc, "pricing": col.removeprefix("uc_price_"), "mae": m["mae"],
                        "corr_hourly": m["corr_hourly"], "bias": m["model_mean"] - m["actual_mean"]})
    cal = pd.DataFrame(cal)
    cal.to_csv(out / "calibration_2023.csv", index=False, float_format="%.3f")
    best = float(cal[cal["pricing"] == "relaxed"].set_index("dump_cost")["mae"].idxmin())

    other = [y for y in YEARS if y != CAL_YEAR]
    jobs = [(y, best, "base") for y in other] + [(2024, best, v) for v in ("no_dynamics", "startup_x2")]
    with ProcessPoolExecutor(max_workers=4) as ex:
        paths = list(ex.map(job, *zip(*jobs)))
    base = pd.concat([_load(cal_paths[best])] + [_load(p) for p in paths[:len(other)]], ignore_index=True)

    # merit-order baseline on the same inputs
    df_all = pd.concat([_inputs(y)[0] for y in YEARS], ignore_index=True)
    fp = fuels.monthly_prices(RAW / "fuels", "2023-01", "2025-12")
    mo = merit.run(df_all, fp, lignite_fuel=LIGNITE_FUEL, must_run_q=MUST_RUN_Q)
    mo["day"] = mo["ts_utc"].dt.tz_convert("Europe/Berlin").dt.date
    base = base.merge(mo[["ts_utc", "model_price"] + [f"model_{f}" for f in merit.THERMAL]], on="ts_utc", how="left")

    rows = []
    for label, col in (("merit order", "model_price"), ("UC restricted", "uc_price_restricted"),
                       ("UC relaxed", "uc_price_relaxed")):
        m = uc.metrics(base, col).assign(model=label)
        rows.append(m)
    met = pd.concat(rows, ignore_index=True)
    met["sample"] = met["year"].map(lambda y: "calibration" if y == CAL_YEAR else "out of sample")
    met.to_csv(out / "metrics.csv", index=False, float_format="%.3f")

    gen = pd.concat([uc.generation_check(base, "model_").assign(model="merit order"),
                     uc.generation_check(base, "uc_").assign(model="UC")], ignore_index=True)
    gen.to_csv(out / "generation_by_fuel.csv", index=False, float_format="%.3f")

    starts = (base.groupby("year")[[f"uc_unit_starts_{f}" for f in merit.THERMAL]].sum()
              .rename(columns=lambda c: c.removeprefix("uc_unit_starts_")))
    starts.to_csv(out / "unit_starts.csv", float_format="%.0f")

    sens = []
    for v, p in zip(("no_dynamics", "startup_x2"), paths[len(other):]):
        r = _load(p)
        for col in ("uc_price_relaxed", "uc_price_restricted"):
            m = uc.metrics(r, col).iloc[0]
            g = uc.generation_check(r, "uc_").set_index("fuel")["error_pct"]
            sens.append({"variant": v, "pricing": col.removeprefix("uc_price_"), "mae": m["mae"],
                         "corr_hourly": m["corr_hourly"], "model_mean": m["model_mean"],
                         **{f"gen_error_pct_{f}": g[f] for f in merit.THERMAL}})
    b24 = base[base["year"] == 2024]
    for col in ("uc_price_relaxed", "uc_price_restricted"):
        m = uc.metrics(b24, col).iloc[0]
        g = uc.generation_check(b24, "uc_").set_index("fuel")["error_pct"]
        sens.append({"variant": "base", "pricing": col.removeprefix("uc_price_"), "mae": m["mae"],
                     "corr_hourly": m["corr_hourly"], "model_mean": m["model_mean"],
                     **{f"gen_error_pct_{f}": g[f] for f in merit.THERMAL}})
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensitivity_2024.csv", index=False, float_format="%.3f")

    base[["ts_utc", "price", "model_price", "uc_price_restricted", "uc_price_relaxed", "uc_dump"]
         ].to_csv(out / "hourly_prices.csv", index=False, float_format="%.2f")
    plots.uc_duration(base, figures / "uc_price_duration.png")
    plots.uc_week(base, figures / "uc_sample_week.png")

    summary = {"calibration": {"year": CAL_YEAR, "dump_cost_eur_mwh": best, "lignite_fuel": LIGNITE_FUEL,
                               "must_run_quantile": MUST_RUN_Q},
               "units": {k: vars(u) for k, u in uc.UNITS.items()}, "clusters_per_fuel": uc.K_PER_FUEL,
               "lookahead_h": uc.LOOKAHEAD,
               "metrics": met.round(3).to_dict(orient="records"),
               "generation_by_fuel": gen.round(2).to_dict(orient="records"),
               "unit_starts": starts.round(0).to_dict(orient="index"),
               "sensitivity_2024": sens.round(3).to_dict(orient="records"),
               "shed_hours": int((base["uc_shed"] > 1e-6).sum()),
               "days_with_commitment_restart": {int(y): int(v) for y, v in base.groupby("year")["uc_restart_days"].first().items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
