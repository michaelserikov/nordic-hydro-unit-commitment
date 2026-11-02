"""Clustered unit commitment of the German thermal fleet (MILP) and the prices it implies.

The merit-order model (merit.py) splits hourly thermal demand between lignite, hard coal and gas
and prices each hour at the cost of the last block. It ignores that plants have a minimum stable
output, minimum up and down times and start-up costs. This module adds them.

Fleet. As in the merit-order model, per fuel and month the available capacity is the maximum
observed hourly output and must-run output (heat, system services) a quantile of it; must-run runs
at zero marginal cost. The capacity above must-run is split into K efficiency classes per fuel
(clusters). Each cluster consists of N identical units of a typical size; the model decides per
hour how many units are online (integer u), how many start (s) and stop (d), and their output p.

Clustered formulation (integer unit counts per cluster, as in Palmintier and Webster 2014):
    u[t] - u[t-1] = s[t] - d[t]
    pmin * size * u[t] <= p[t] <= size * u[t]
    u[t] >= sum of s over the last min_up hours         (a started unit stays on)
    N - u[t] >= sum of d over the last min_down hours   (a stopped unit stays off)
    sum_k p[k, t] - dump[t] + shed[t] = demand[t] - must_run[t]
Cost: marginal cost x output + start-up cost x starts + dump and shed penalties. dump is surplus
inflexible output that has to be absorbed (renewable curtailment or exports at a negative price),
priced at -dump_cost; shed is unserved thermal demand at a high penalty (it should never bind).

Days are solved in sequence (rolling horizon): each day is optimised over 24 + LOOKAHEAD hours,
the first 24 are kept and their commitment and start history carry over to the next day.

Prices. A MILP has no dual prices, so two standard constructions are reported:
- restricted: fix the commitment (u, s, d) at the MILP solution and take the dual of the demand
  constraint of the remaining LP: the marginal cost of the marginal online unit (in the rare hours
  where all online units run at full output this dual is degenerate; the dearest online unit's cost
  is used);
- relaxed: duals of the LP relaxation (integrality dropped), which spreads start-up costs and
  minimum-output constraints into the price (close to convex-hull pricing in many cases).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .merit import TECH, THERMAL, LIGNITE_FUEL_EUR_MWH


@dataclass(frozen=True)
class UnitType:
    size_mw: float          # typical unit size
    pmin: float             # minimum stable output, share of size
    min_up: int             # hours
    min_down: int           # hours
    startup_eur_per_mw: float   # cost of one (warm) start per MW of unit size, fuel and wear


# Typical values (assumptions, documented in the README; sensitivity in the run).
UNITS = {
    "lignite": UnitType(500.0, 0.40, 8, 6, 70.0),
    "hard_coal": UnitType(400.0, 0.35, 6, 4, 55.0),
    "ccgt": UnitType(400.0, 0.40, 4, 2, 45.0),
    "ocgt": UnitType(100.0, 0.20, 1, 1, 25.0),
}
K_PER_FUEL = 4          # efficiency classes per fuel; for gas the two best are CCGT, the rest OCGT
LOOKAHEAD = 12          # hours beyond the day
SHED_COST = 10000.0     # EUR/MWh, high enough that the MILP always starts a unit instead


@dataclass
class Cluster:
    name: str
    fuel: str
    unit: UnitType
    eta: float


def clusters(k: int = K_PER_FUEL, units: dict | None = None) -> list[Cluster]:
    units = UNITS if units is None else units
    out = []
    for f in THERMAL:
        t = TECH[f]
        for i, eta in enumerate(np.linspace(t.eta_best, t.eta_worst, k)):
            if f == "gas":
                ut = units["ccgt"] if i < k / 2 else units["ocgt"]
            else:
                ut = units[f]
            out.append(Cluster(f"{f}_{i}", f, ut, float(eta)))
    return out


def marginal_costs(cl: list[Cluster], fuel_prices: pd.Series, lignite_fuel: float = LIGNITE_FUEL_EUR_MWH) -> np.ndarray:
    fuel = {"lignite": lignite_fuel, "hard_coal": fuel_prices["coal_eur_mwh"], "gas": fuel_prices["gas_eur_mwh"]}
    return np.array([(fuel[c.fuel] + TECH[c.fuel].ef_t_per_mwh_th * fuel_prices["co2_eur_t"]) / c.eta
                     + TECH[c.fuel].vom for c in cl])


@dataclass
class State:
    """Commitment carried from one day to the next: units online and recent starts and stops."""
    u: np.ndarray                 # (K,)
    s_hist: np.ndarray            # (K, H) starts in the last H hours, oldest first
    d_hist: np.ndarray            # (K, H)
    n_prev: np.ndarray            # (K,) units available on the previous day (capacity can change monthly)


HIST = 24


def solve_window(cost: np.ndarray, n_units: np.ndarray, size: np.ndarray, demand: np.ndarray,
                 cl: list[Cluster], state: State | None, dump_cost: float,
                 time_limit: float = 30.0) -> dict:
    """Unit commitment over T hours for K clusters.

    cost (T, K) EUR/MWh; n_units (T, K) units available; size (T, K) MW per unit; demand (T,)
    is the thermal demand left after must-run (may be negative). Returns MILP solution and the
    restricted and relaxed hourly prices.
    """
    T, K = cost.shape
    nv_k = 4 * T                                   # u, s, d, p per cluster
    def iu(k, t): return k * nv_k + t
    def is_(k, t): return k * nv_k + T + t
    def id_(k, t): return k * nv_k + 2 * T + t
    def ip(k, t): return k * nv_k + 3 * T + t
    idump = K * nv_k
    ished = idump + T
    nv = ished + T

    c = np.zeros(nv)
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    integ = np.zeros(nv)
    pmin = np.array([x.unit.pmin for x in cl])
    su = np.array([x.unit.startup_eur_per_mw for x in cl])
    mu = np.array([x.unit.min_up for x in cl])
    md = np.array([x.unit.min_down for x in cl])
    n_max = n_units.max(axis=0)
    if state is not None:
        n_max = np.maximum(n_max, np.maximum(state.u, state.n_prev))
    for k in range(K):
        for t in range(T):
            ub[iu(k, t)] = n_units[t, k]
            ub[is_(k, t)] = n_max[k]
            ub[id_(k, t)] = n_max[k]
            integ[[iu(k, t), is_(k, t), id_(k, t)]] = 1
            c[ip(k, t)] = cost[t, k]
            c[is_(k, t)] = su[k] * size[t, k]
    if state is None:                              # first day: free initial commitment
        for k in range(K):
            ub[is_(k, 0)] = 0
            ub[id_(k, 0)] = 0
    c[idump:idump + T] = dump_cost
    c[ished:ished + T] = SHED_COST

    rows, cols, vals, rlo, rhi = [], [], [], [], []
    r = 0
    def add(entries, lo, hi):
        nonlocal r
        for j, v in entries:
            rows.append(r); cols.append(j); vals.append(v)
        rlo.append(lo); rhi.append(hi)
        r += 1

    bal_rows = []
    for t in range(T):
        bal_rows.append(r)
        add([(ip(k, t), 1.0) for k in range(K)] + [(idump + t, -1.0), (ished + t, 1.0)], demand[t], demand[t])
    for k in range(K):
        for t in range(T):
            # transition
            if t == 0:
                if state is not None:
                    add([(iu(k, 0), 1.0), (is_(k, 0), -1.0), (id_(k, 0), 1.0)], state.u[k], state.u[k])
            else:
                add([(iu(k, t), 1.0), (iu(k, t - 1), -1.0), (is_(k, t), -1.0), (id_(k, t), 1.0)], 0.0, 0.0)
            # output limits
            add([(ip(k, t), 1.0), (iu(k, t), -size[t, k])], -np.inf, 0.0)
            add([(ip(k, t), 1.0), (iu(k, t), -pmin[k] * size[t, k])], 0.0, np.inf)
            # minimum up time: u[t] - sum_{tau=t-mu+1..t} s[tau] >= past starts in window
            past_s = past_d = 0.0
            ent = [(iu(k, t), 1.0)]
            for tau in range(t - mu[k] + 1, t + 1):
                if tau >= 0:
                    ent.append((is_(k, tau), -1.0))
                elif state is not None:
                    past_s += state.s_hist[k, HIST + tau]
            add(ent, min(past_s, n_units[t, k]), np.inf)   # capacity may have dropped at a month start
            # minimum down time: u[t] + sum d <= n_max - past stops
            ent = [(iu(k, t), 1.0)]
            for tau in range(t - md[k] + 1, t + 1):
                if tau >= 0:
                    ent.append((id_(k, tau), 1.0))
                elif state is not None:
                    past_d += state.d_hist[k, HIST + tau]
            add(ent, -np.inf, n_max[k] - past_d)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    rlo, rhi = np.array(rlo), np.array(rhi)
    res = milp(c, constraints=LinearConstraint(A, rlo, rhi), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-3, "disp": False})
    if res.x is None:
        raise RuntimeError(f"unit commitment failed: {res.message}")
    x = np.round(res.x, 6)
    x[integ == 1] = np.round(x[integ == 1])

    def lp_prices(fix: bool) -> np.ndarray:
        lb2, ub2 = lb.copy(), ub.copy()
        if fix:
            lb2[integ == 1] = x[integ == 1]
            ub2[integ == 1] = x[integ == 1]
        eq = rlo == rhi
        le_hi = (~eq) & np.isfinite(rhi)
        ge_lo = (~eq) & np.isfinite(rlo)
        A_ub = sparse.vstack([A[le_hi], -A[ge_lo]]).tocsr()
        b_ub = np.concatenate([rhi[le_hi], -rlo[ge_lo]])
        lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A[eq], b_eq=rhi[eq],
                     bounds=list(zip(lb2, np.where(np.isinf(ub2), None, ub2))), method="highs")
        if lp.status != 0:
            raise RuntimeError(f"pricing LP failed: {lp.message}")
        eq_idx = np.flatnonzero(eq)
        pos = {row: i for i, row in enumerate(eq_idx)}
        return np.array([lp.eqlin.marginals[pos[b]] for b in bal_rows])

    u = np.array([[x[iu(k, t)] for k in range(K)] for t in range(T)])
    price_r = lp_prices(True)
    # If every online unit runs at full output, the dual of the restricted LP is degenerate: any value
    # between the cost of the dearest online unit and the shed penalty is valid. Use the former.
    degenerate = price_r > 0.5 * SHED_COST
    for t in np.flatnonzero(degenerate):
        online = u[t] > 0.5
        price_r[t] = cost[t][online].max() if online.any() else SHED_COST
    s = np.array([[x[is_(k, t)] for k in range(K)] for t in range(T)])
    d = np.array([[x[id_(k, t)] for k in range(K)] for t in range(T)])
    p = np.array([[x[ip(k, t)] for k in range(K)] for t in range(T)])
    return {"u": u, "s": s, "d": d, "p": p, "dump": x[idump:idump + T], "shed": x[ished:ished + T],
            "cost": float(res.fun), "gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
            "price_restricted": price_r, "price_relaxed": lp_prices(False)}


def next_state(sol: dict, keep: int, prev: State | None, n_units: np.ndarray) -> State:
    K = sol["u"].shape[1]
    s_all = np.concatenate([prev.s_hist if prev else np.zeros((K, HIST)), sol["s"][:keep].T], axis=1)
    d_all = np.concatenate([prev.d_hist if prev else np.zeros((K, HIST)), sol["d"][:keep].T], axis=1)
    return State(u=sol["u"][keep - 1].copy(), s_hist=s_all[:, -HIST:], d_hist=d_all[:, -HIST:],
                 n_prev=n_units[:keep].max(axis=0))


def run(df: pd.DataFrame, fuel_prices: pd.DataFrame, must_run_q: float, dump_cost: float,
        lignite_fuel: float = LIGNITE_FUEL_EUR_MWH, k_per_fuel: int = K_PER_FUEL,
        units: dict | None = None, lookahead: int = LOOKAHEAD) -> pd.DataFrame:
    """Rolling daily unit commitment over df (from de_data.thermal_table). Adds model columns."""
    cl = clusters(k_per_fuel, units)
    K = len(cl)
    from .merit import fleet
    fl = fleet(df, must_run_q).set_index(["year", "month", "fuel"])
    ym = list(zip(df["year"], df["month"]))
    fd = {key: (float(r["capacity"]), float(r["must_run"])) for key, r in fl.iterrows()}
    must_f = {f: np.array([fd[(y, m, f)][1] for y, m in ym]) for f in THERMAL}
    must = sum(must_f.values())
    cap = np.zeros((len(df), K))
    for j, c in enumerate(cl):
        cap[:, j] = [(fd[(y, m, c.fuel)][0] - fd[(y, m, c.fuel)][1]) / k_per_fuel for y, m in ym]
    n_units = np.maximum(np.round(cap / np.array([c.unit.size_mw for c in cl])), 0)
    size = np.where(n_units > 0, cap / np.maximum(n_units, 1), 0.0)
    months = sorted(set(ym))
    mc = {(y, m): marginal_costs(cl, fuel_prices.loc[pd.Period(f"{y}-{m:02d}", freq="M")], lignite_fuel)
          for y, m in months}
    cost = np.array([mc[k] for k in ym])
    demand = df["thermal_demand"].to_numpy() - must

    out = {k: np.zeros(len(df)) for k in ("price_restricted", "price_relaxed", "dump", "shed", "starts")}
    gen = np.zeros((len(df), K))
    starts = np.zeros((len(df), K))
    days = df.groupby("day", sort=True).indices
    state = None
    restarts = 0
    for day in sorted(days):
        idx = np.sort(days[day])
        i0, n = idx[0], len(idx)
        w = np.arange(i0, min(i0 + n + lookahead, len(df)))
        try:
            sol = solve_window(cost[w], n_units[w], size[w], demand[w], cl, state, dump_cost)
        except RuntimeError:            # history inconsistent with new capacity: restart the commitment
            sol = solve_window(cost[w], n_units[w], size[w], demand[w], cl, None, dump_cost)
            restarts += 1
        keep = slice(0, n)
        out["price_restricted"][idx] = sol["price_restricted"][keep]
        out["price_relaxed"][idx] = sol["price_relaxed"][keep]
        out["dump"][idx] = sol["dump"][keep]
        out["shed"][idx] = sol["shed"][keep]
        out["starts"][idx] = (sol["s"][keep] * size[idx]).sum(axis=1)   # MW started
        gen[idx] = sol["p"][keep]
        starts[idx] = sol["s"][keep]
        state = next_state(sol, n, state, n_units[w])

    res = df.copy()
    for k, v in out.items():
        res[f"uc_{k}"] = v
    for f in THERMAL:
        cols = [j for j, c in enumerate(cl) if c.fuel == f]
        res[f"uc_{f}"] = gen[:, cols].sum(axis=1) + must_f[f]
        res[f"uc_unit_starts_{f}"] = starts[:, cols].sum(axis=1)
    for j, c in enumerate(cl):
        res[f"uc_gen_{c.name}"] = gen[:, j]
    res.attrs["restarts"] = restarts
    res["uc_restart_days"] = restarts
    return res


def metrics(res: pd.DataFrame, price_col: str) -> pd.DataFrame:
    rows = []
    for y, g in res.groupby("year"):
        e = g[price_col] - g["price"]
        daily = g.groupby("day")[["price", price_col]].mean()
        neg = g["price"] < 0
        rows.append({"year": y, "actual_mean": g["price"].mean(), "model_mean": g[price_col].mean(),
                     "mae": e.abs().mean(), "rmse": float(np.sqrt((e ** 2).mean())),
                     "corr_hourly": g["price"].corr(g[price_col]),
                     "corr_daily": daily["price"].corr(daily[price_col]),
                     "share_hours_at_or_below_zero_actual": float((g["price"] <= 0).mean()),
                     "share_hours_at_or_below_zero_model": float((g[price_col] <= 0).mean()),
                     "mae_negative_hours": float(e[neg].abs().mean()) if neg.any() else np.nan})
    return pd.DataFrame(rows)


def generation_check(res: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for y, g in res.groupby("year"):
        for f in THERMAL:
            rows.append({"year": y, "fuel": f, "actual_twh": g[f].sum() / 1e6, "model_twh": g[f"{prefix}{f}"].sum() / 1e6})
    out = pd.DataFrame(rows)
    out["error_pct"] = 100 * (out["model_twh"] / out["actual_twh"] - 1)
    return out
