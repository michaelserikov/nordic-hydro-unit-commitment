"""Water value SDP on synthetic reservoirs."""

import numpy as np

from nhuc import water_values as wv


def _res(**kw):
    base = dict(capacity=20.0, r_min=0.0, r_max=2.0, step=0.5)
    base.update(kw)
    return wv.Reservoir(**base)


def test_water_values_fall_with_filling():
    rng = np.random.default_rng(0)
    inflow = rng.uniform(0.5, 1.5, (6, wv.WEEKS))
    price = 1 + 0.4 * np.cos(np.linspace(0, 2 * np.pi, wv.WEEKS))[None, :] + rng.normal(0, 0.05, (6, wv.WEEKS))
    sol = wv.solve(inflow, price, _res())
    w = sol["water_values"][10, 2:-2]
    # concave value function: water values do not increase with filling (up to grid noise)
    assert np.all(np.diff(w) <= 0.05)


def test_deterministic_sdp_matches_perfect_foresight():
    inflow = np.full((1, wv.WEEKS), 1.0)
    price = (1 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, wv.WEEKS)))[None, :]
    res = _res()
    sol = wv.solve(inflow, price, res)
    sim = wv.simulate(sol, 10.0, inflow[0], price[0], res)
    end = float(sim["fill"].iloc[-1])
    term = wv.water_value_at(sol, wv.WEEKS, end)
    pf_obj, pf_r, pf_s = wv.perfect_foresight(10.0, inflow[0], price[0], res, terminal_value=term)
    sdp_obj = float((price[0] * sim["release"]).sum() + term * end)
    # one known scenario: the SDP policy is the perfect-foresight schedule up to grid error
    assert abs(sdp_obj - pf_obj) / pf_obj < 0.02


def test_floor_is_respected():
    inflow = np.full((3, wv.WEEKS), 0.5)
    price = np.ones((3, wv.WEEKS))
    floor = np.full(wv.WEEKS, 8.0)
    res = _res(s_min=floor)
    sol = wv.solve(inflow, price, res)
    sim = wv.simulate(sol, 12.0, inflow[0], price[0], res)
    assert sim["fill"].min() >= 8.0 - 1e-9
