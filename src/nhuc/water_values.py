"""Water values of the aggregate Norwegian reservoir by stochastic dynamic programming (SDP).

Model. One reservoir with the capacity of all Norwegian reservoirs (NVE, about 87 TWh), weekly
stages. In week w the operator knows this week's inflow and price (hazard-decision), not the
future ones, and chooses the release R (between a minimum and the turbine capacity) and spillage:
    S[w+1] = S[w] + I[w] - R[w] - spill[w],  0 <= S <= capacity
    V_w(S) = E_y [ max_R  p_w^y R + beta V_{w+1}(S[w+1]) ]
The expectation runs over the historical years y (inflow and price of the same week and year
together, so their correlation within a week is kept; weeks are treated as independent). The year
is periodic: the backward recursion is repeated over several years until the value function of
week 1 stops changing (discounted at 5% a year). The water value is the slope dV/dS: what one more
MWh in the reservoir is worth in week w at filling S.

Price scaling. A price-taking producer's optimal policy does not change if all prices are
multiplied by the same constant, so the model is trained on prices divided by their annual mean
(price shape) and water values come out in units of the annual mean price. For the backtest of a
year they are multiplied by that year's mean price, so the comparison with observed prices tests
timing and shape, not the price level (which no hydro model can know a year ahead).

Operating floor. Norwegian producers keep reservoirs well above empty for security of supply (and
NVE watches the spring minimum). The model may not go below the lowest end-of-week filling
observed in 1995-2014, the twenty years before the sample, so the floor uses no test-year data.

Limitations: one aggregate reservoir (no cascade, no area constraints), price-taker (Norwegian
hydro sets much of its own price), inflow implied by the water balance, weeks independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

BETA = (1 / 1.05) ** (1 / 52)
WEEKS = 52


@dataclass
class Reservoir:
    capacity: float      # TWh
    r_min: float         # TWh per week
    r_max: float         # TWh per week
    s_min: np.ndarray | None = None   # (52,) lowest allowed filling at the end of each week, TWh
    step: float = 0.25   # grid step, TWh

    def floor(self, w: int) -> float:
        return 0.0 if self.s_min is None else float(self.s_min[w])


def to_52_weeks(t: pd.DataFrame) -> pd.DataFrame:
    """Drop ISO week 53 (2015 and 2020); every backtest year starts from its observed filling."""
    return t[t["week"] <= WEEKS].reset_index(drop=True)


def guide_curve(nve: pd.DataFrame, years) -> np.ndarray:
    """Lowest observed end-of-week filling (TWh) per week of year over `years`: an operating floor."""
    g = nve[(nve["area"] == "NO") & nve["year"].isin(list(years)) & (nve["week"] <= WEEKS)]
    return g.groupby("week")["fill_twh"].min().reindex(range(1, WEEKS + 1)).interpolate().to_numpy()


def scenarios(t: pd.DataFrame, years) -> tuple[np.ndarray, np.ndarray]:
    """Inflow (Y, 52) in TWh and price relative to the annual mean (Y, 52)."""
    inflow, price = [], []
    for y in years:
        g = t[t["year"] == y].set_index("week").reindex(range(1, WEEKS + 1))
        if g[["inflow_twh", "price"]].isna().any().any():
            continue
        inflow.append(g["inflow_twh"].to_numpy())
        price.append((g["price"] / g["price"].mean()).to_numpy())
    return np.array(inflow), np.array(price)


def _stage(V_next: np.ndarray, levels: np.ndarray, inflow: np.ndarray, price: np.ndarray,
           res: Reservoir, floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """One backward step. Returns expected value per level and the optimal next level per (y, level)."""
    S = levels[:, None]                 # current level
    Sn = levels[None, :]                # candidate next level
    Y = len(inflow)
    vals = np.empty((Y, len(levels)))
    choice = np.empty((Y, len(levels)), dtype=int)
    for y in range(Y):
        avail = S + inflow[y] - Sn                       # water to release or spill
        R = np.minimum(avail, res.r_max)
        ok = (avail >= res.r_min - 1e-9) & (Sn >= floor - 1e-9)
        # if even the lowest next level cannot give r_min, release everything (allowed, rare)
        q = price[y] * R + BETA * V_next[None, :]
        q = np.where(ok, q, -np.inf)
        none_ok = ~ok.any(axis=1)
        if none_ok.any():               # floor or minimum release cannot be met: release r_min if possible
            for i in np.flatnonzero(none_ok):
                j = int(np.clip(np.searchsorted(levels, S[i, 0] + inflow[y] - res.r_min, side="right") - 1,
                                0, len(levels) - 1))
                q[i, j] = price[y] * max(S[i, 0] + inflow[y] - levels[j], 0.0) + BETA * V_next[j]
        choice[y] = q.argmax(axis=1)
        vals[y] = q.max(axis=1)
    return vals.mean(axis=0), choice


def solve(inflow: np.ndarray, price: np.ndarray, res: Reservoir, max_cycles: int = 40,
          tol: float = 1e-4) -> dict:
    """Periodic SDP. Returns value functions V (53, L) and water values (52, L) per level."""
    levels = np.arange(0, res.capacity + 1e-9, res.step)
    V = np.zeros((WEEKS + 1, len(levels)))
    prev_wv = None
    for cycle in range(max_cycles):
        V[WEEKS] = V[0] - V[0].mean() if cycle else 0.0
        for w in range(WEEKS - 1, -1, -1):
            V[w], _ = _stage(V[w + 1], levels, inflow[:, w], price[:, w], res, res.floor(w))
        wv = np.gradient(V[:WEEKS], res.step, axis=1)
        if prev_wv is not None and np.nanmax(np.abs(wv - prev_wv)) < tol:
            break
        prev_wv = wv
    return {"levels": levels, "V": V, "water_values": wv, "cycles": cycle + 1}


def simulate(sol: dict, s0: float, inflow: np.ndarray, price: np.ndarray, res: Reservoir) -> pd.DataFrame:
    """Operate one year with the SDP policy on realised inflow and (relative) price, week by week."""
    levels, V = sol["levels"], sol["V"]
    S = s0
    rows = []
    for w in range(WEEKS):
        Sn = levels
        avail = S + inflow[w] - Sn
        R = np.minimum(avail, res.r_max)
        ok = (avail >= res.r_min - 1e-9) & (Sn >= res.floor(w) - 1e-9)
        q = np.where(ok, price[w] * R + BETA * V[w + 1], -np.inf)
        if not ok.any():
            j = int(np.clip(np.searchsorted(levels, S + inflow[w] - res.r_min, side="right") - 1, 0, len(levels) - 1))
            r, spill = max(S + inflow[w] - levels[j], 0.0), 0.0
        else:
            j = int(q.argmax())
            r = float(R[j])
            spill = float(avail[j] - R[j])
        rows.append({"week": w + 1, "release": r, "spill": spill, "fill": float(levels[j])})
        S = float(levels[j])
    return pd.DataFrame(rows)


def water_value_at(sol: dict, week: int, fill: float) -> float:
    return float(np.interp(fill, sol["levels"], sol["water_values"][week - 1]))


def perfect_foresight(s0: float, inflow: np.ndarray, price: np.ndarray, res: Reservoir,
                      terminal_value: float) -> tuple[float, np.ndarray, np.ndarray]:
    """LP with known inflows and prices for the whole year; end storage valued at terminal_value.

    Variables per week: release R, spill X, storage at the end of the week S. Returns objective
    (revenue in price units x TWh, plus terminal value x S_end), release and storage paths.
    """
    n = len(inflow)
    # x = [R(n), X(n), S(n)]
    c = np.concatenate([-price, np.zeros(n), np.zeros(n)])
    c[3 * n - 1] = -terminal_value
    A = np.zeros((n, 3 * n))
    b = np.zeros(n)
    for w in range(n):
        A[w, w] = 1.0          # R
        A[w, n + w] = 1.0      # X
        A[w, 2 * n + w] = 1.0  # S[w]
        if w > 0:
            A[w, 2 * n + w - 1] = -1.0
            b[w] = inflow[w]
        else:
            b[w] = s0 + inflow[w]
    floors = [(min(res.floor(w), res.capacity), res.capacity) for w in range(n)]
    bounds = [(res.r_min, res.r_max)] * n + [(0, None)] * n + floors
    lp = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if lp.status != 0:
        # floor or minimum release infeasible in a dry year: relax the floor
        bounds = [(res.r_min, res.r_max)] * n + [(0, None)] * n + [(0, res.capacity)] * n
        lp = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    return -lp.fun, lp.x[:n], lp.x[2 * n:]
