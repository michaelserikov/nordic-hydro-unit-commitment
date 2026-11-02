"""Unit commitment on small synthetic cases (no data needed)."""

import numpy as np

from nhuc import uc


def _cl(pmin=0.0, up=1, down=1, su=0.0, size=100.0, eta=0.5, fuel="gas"):
    return uc.Cluster("c", fuel, uc.UnitType(size, pmin, up, down, su), eta)


def _off(k=1):
    """All units offline, no recent starts or stops."""
    return uc.State(u=np.zeros(k), s_hist=np.zeros((k, uc.HIST)), d_hist=np.zeros((k, uc.HIST)), n_prev=np.ones(k))


def _solve(demand, cost, n, cl, dump_cost=0.0, state=None):
    T = len(demand)
    K = len(cl)
    cost = np.tile(np.asarray(cost, float), (T, 1))
    n_units = np.tile(np.asarray(n, float), (T, 1))
    size = np.tile(np.array([c.unit.size_mw for c in cl]), (T, 1))
    return uc.solve_window(cost, n_units, size, np.asarray(demand, float), cl, state, dump_cost)


def test_without_dynamics_prices_equal_merit_order():
    cl = [_cl(), _cl()]
    sol = _solve([50, 150, 250], [20.0, 40.0], [2, 2], cl)
    # 50 and 150 MW are served by the cheap cluster (200 MW), 250 MW needs the dearer one
    np.testing.assert_allclose(sol["price_restricted"], [20, 20, 40], atol=1e-6)
    np.testing.assert_allclose(sol["price_relaxed"], [20, 20, 40], atol=1e-6)
    assert sol["shed"].sum() == 0


def test_minimum_output_forces_dump_at_negative_price():
    cl = [_cl(pmin=0.5, up=3, su=1000.0)]
    # an online 100 MW unit cannot go below 50 MW; at 20 MW demand the surplus is dumped
    sol = _solve([100, 20, 100], [30.0], [1], cl, dump_cost=15.0)
    assert sol["u"][:, 0].tolist() == [1, 1, 1]
    assert abs(sol["dump"][1] - 30.0) < 1e-6
    assert abs(sol["price_restricted"][1] + 15.0) < 1e-6


def test_minimum_up_time_keeps_unit_online():
    cl = [_cl(pmin=0.2, up=4, su=10.0)]
    sol = _solve([0, 80, 0, 0, 0, 0], [30.0], [1], cl, dump_cost=1.0, state=_off())
    u = sol["u"][:, 0]
    start = int(np.argmax(u))
    assert u[start:start + 4].tolist() == [1, 1, 1, 1]


def test_start_up_cost_raises_relaxed_price_in_peak():
    cl = [_cl(su=0.0), _cl(su=500.0, size=50.0)]
    sol = _solve([100, 140, 100], [20.0, 40.0], [1, 1], cl)
    # the peak unit's start-up cost is spread into the relaxed price of the peak hour
    assert sol["price_relaxed"][1] > sol["price_restricted"][1] - 1e-9
    assert sol["price_restricted"][1] == 40.0 or abs(sol["price_restricted"][1] - 40.0) < 1e-6


def test_state_carries_commitment():
    cl = [_cl(pmin=0.3, up=6, su=100.0)]
    first = _solve([60, 60], [25.0], [1], cl, state=_off())
    st = uc.next_state(first, 2, _off(), np.ones((2, 1)))
    # the unit started at hour 0 must stay on for 6 hours, i.e. 4 more hours into the next window
    nxt = _solve([0, 0, 0, 0, 0], [25.0], [1], cl, dump_cost=1.0, state=st)
    assert nxt["u"][:4, 0].tolist() == [1, 1, 1, 1]
