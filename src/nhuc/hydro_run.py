"""Water values of Norwegian hydropower: fit on 2015-2025 and backtest each year out of sample.

    python -m nhuc.hydro_run         # data from NVE and Energy-Charts (cached in data/raw)

For every year 2015-2025 the SDP is trained on the other ten years and then operates the
reservoir through the held-out year on its realised inflows and prices. It is compared with
(a) the observed operation of Norwegian reservoir plants and (b) a perfect-foresight schedule that
knows the whole year in advance. End-of-year storage differences are valued at the model water
value, so all three are compared on the same terms. Writes results/hydro/ and figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import hydro_data, plots
from . import water_values as wvm

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
YEARS = range(2015, 2026)


def main() -> dict:
    out, figures = ROOT / "results" / "hydro", ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)

    t = wvm.to_52_weeks(hydro_data.weekly_table(YEARS, RAW))
    t.to_csv(out / "weekly_data.csv", index=False, float_format="%.4f")
    nve = hydro_data.reservoir(RAW)
    res = wvm.Reservoir(capacity=float(nve.query("area == 'NO'")["capacity_twh"].max()),
                        r_min=float(t["gen_twh"].min()), r_max=float(t["gen_twh"].max()),
                        s_min=wvm.guide_curve(nve, range(1995, 2015)))

    inflow_all, price_all = wvm.scenarios(t, YEARS)
    full = wvm.solve(inflow_all, price_all, res)
    plots.water_values(full["water_values"], full["levels"], res.capacity, figures / "water_values.png",
                       observed=t[["year", "week", "fill_twh"]], floor=res.s_min)

    rows, weekly = [], []
    for y in YEARS:
        train = [x for x in YEARS if x != y]
        inflow, price = wvm.scenarios(t, train)
        sol = wvm.solve(inflow, price, res)
        g = t[t["year"] == y].set_index("week").reindex(range(1, wvm.WEEKS + 1))
        if g[["inflow_twh", "price"]].isna().any().any():
            continue
        mean_p = float(g["price"].mean())
        rel = (g["price"] / mean_p).to_numpy()
        I = g["inflow_twh"].to_numpy()
        s0 = float(g["fill_prev"].iloc[0])
        s_end_obs = float(g["fill_twh"].iloc[-1])
        wv_end = wvm.water_value_at(sol, wvm.WEEKS, s_end_obs) * mean_p      # EUR/MWh

        sim = wvm.simulate(sol, s0, I, rel, res)
        pf_obj, pf_r, pf_s = wvm.perfect_foresight(s0, I, rel, res, terminal_value=wv_end / mean_p)
        p = g["price"].to_numpy()

        def value(release, s_end):     # million EUR: revenue + end storage difference at the water value
            return float((p * release).sum() + (s_end - s_end_obs) * wv_end)
        obs_v = value(g["gen_twh"].to_numpy(), s_end_obs)
        sdp_v = value(sim["release"].to_numpy(), float(sim["fill"].iloc[-1]))
        pf_v = value(pf_r, float(pf_s[-1]))
        wv_obs = np.array([wvm.water_value_at(sol, w, f) for w, f in zip(range(1, 53), g["fill_prev"])]) * mean_p
        rows.append({"year": y, "mean_price": mean_p, "inflow_twh": I.sum(), "obs_gen_twh": g["gen_twh"].sum(),
                     "sdp_gen_twh": sim["release"].sum(), "pf_gen_twh": pf_r.sum(),
                     "sdp_spill_twh": sim["spill"].sum(),
                     "value_observed_meur": obs_v, "value_sdp_meur": sdp_v, "value_pf_meur": pf_v,
                     "sdp_vs_observed_pct": 100 * (sdp_v / obs_v - 1), "pf_vs_sdp_pct": 100 * (pf_v / sdp_v - 1),
                     "corr_wv_price": float(np.corrcoef(wv_obs, p)[0, 1]),
                     "corr_release_obs_sdp": float(np.corrcoef(sim["release"], g["gen_twh"])[0, 1]),
                     "fill_mae_twh": float(np.abs(sim["fill"].to_numpy() - g["fill_twh"].to_numpy()).mean()),
                     "cycles": sol["cycles"]})
        weekly.append(pd.DataFrame({"year": y, "week": range(1, 53), "date": g["date"].to_numpy(),
                                    "price": p, "wv_obs": wv_obs, "fill_obs": g["fill_twh"].to_numpy(),
                                    "fill_sdp": sim["fill"].to_numpy(), "fill_pf": pf_s,
                                    "gen_obs": g["gen_twh"].to_numpy(), "gen_sdp": sim["release"].to_numpy(),
                                    "gen_pf": pf_r}))
    bt = pd.DataFrame(rows)
    wk = pd.concat(weekly, ignore_index=True)
    bt.to_csv(out / "backtest_by_year.csv", index=False, float_format="%.3f")
    wk.to_csv(out / "backtest_weekly.csv", index=False, float_format="%.4f")
    plots.hydro_backtest(wk[wk["year"] >= 2021], figures / "hydro_backtest.png")

    tot = bt[["value_observed_meur", "value_sdp_meur", "value_pf_meur"]].sum()
    summary = {"reservoir": {k: (v.round(2).tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(res).items()}, "years": list(YEARS), "discount_rate": 0.05,
               "inflow_twh_per_year": {int(k): round(v, 1) for k, v in t.groupby("year")["inflow_twh"].sum().items()},
               "full_sample_cycles": full["cycles"],
               "totals_meur": tot.round(0).to_dict(),
               "sdp_vs_observed_pct_total": round(100 * (tot["value_sdp_meur"] / tot["value_observed_meur"] - 1), 2),
               "pf_vs_sdp_pct_total": round(100 * (tot["value_pf_meur"] / tot["value_sdp_meur"] - 1), 2),
               "corr_wv_price_median": round(float(bt["corr_wv_price"].median()), 3),
               "corr_release_median": round(float(bt["corr_release_obs_sdp"].median()), 3),
               "by_year": bt.round(3).to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
