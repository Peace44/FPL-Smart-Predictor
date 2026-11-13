"""Synthetic player market shared by the tests (no network, no real data)."""
import numpy as np
import pandas as pd
import pytest

from fpl_smart.my_team import TeamState
from fpl_smart.solver import SolverOptions

SMALL = {1: 6, 2: 14, 3: 14, 4: 9}        # candidate pool per position: keeps solves around a second


def make_market(seed: int = 0, n_teams: int = 20, gws=(1, 2, 3, 4)):
    rng = np.random.default_rng(seed)
    rows, pid = [], 1
    base = {1: (40, 55), 2: (40, 65), 3: (45, 120), 4: (45, 130)}
    for team in range(1, n_teams + 1):
        for pos, n in ((1, 2), (2, 5), (3, 5), (4, 3)):
            for _ in range(n):
                lo, hi = base[pos]
                cost = int(rng.integers(lo, hi + 1))
                rows.append({"id": pid, "name": f"P{pid}", "team": f"T{team:02d}", "team_id": team, "pos": pos,
                             "now_cost": cost, "price": cost / 10, "quality": 1.2 + cost / 28 + rng.normal(0, 0.5)})
                pid += 1
    players = pd.DataFrame(rows).set_index("id")
    xp = pd.DataFrame({g: (players["quality"] + rng.normal(0, 0.4, len(players))).clip(lower=0.1) for g in gws})
    return players, xp


@pytest.fixture(scope="session")
def market():
    return make_market()


@pytest.fixture()
def opts():
    return SolverOptions(per_pos_keep=dict(SMALL), time_limit=30, use_chips=False, threads=1)


def empty_state(gw: int = 1, bank: int = 1000) -> TeamState:
    return TeamState(team_id=0, next_gw=gw, squad=[], buy_price={}, sell_price={}, bank=bank, free_transfers=1)


def initial_plan(players, xp, opts, gw: int = 1, horizon: int = 3):
    from dataclasses import replace
    from fpl_smart.solver import solve
    o = replace(opts, chips=[{"name": "wildcard", "gws": [gw]}], wildcard_gw=gw, use_chips=True)
    return solve(xp, players, empty_state(gw), horizon, o)


@pytest.fixture(scope="session")
def squad_state(market):
    """A sensible 15-man squad with one free transfer and a little money in the bank."""
    players, xp = market
    plan = initial_plan(players, xp, SolverOptions(per_pos_keep=dict(SMALL), time_limit=30, threads=1))
    w = plan.weeks[0]
    cost = players["now_cost"].to_dict()
    return TeamState(0, 1, list(w.squad), {p: cost[p] for p in w.squad}, {p: cost[p] for p in w.squad},
                     w.bank, 1)


def check_squad(week, players, budget_ok=True):
    pos = players["pos"]
    assert len(week.squad) == len(set(week.squad)) == 15
    assert pos[week.squad].value_counts().to_dict() == {2: 5, 3: 5, 1: 2, 4: 3}
    assert players.loc[week.squad, "team_id"].value_counts().max() <= 3
    assert len(week.lineup) == 11 and set(week.lineup) <= set(week.squad)
    n = pos[week.lineup].value_counts()
    assert n[1] == 1 and 3 <= n[2] <= 5 and 2 <= n[3] <= 5 and 1 <= n[4] <= 3
    assert sorted(week.lineup + week.bench) == sorted(week.squad)
    assert pos[week.bench[0]] == 1 and all(pos[p] != 1 for p in week.bench[1:])
    assert week.captain in week.lineup and week.vice in week.lineup and week.captain != week.vice
    assert week.bank >= 0
