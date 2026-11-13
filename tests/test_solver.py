from dataclasses import replace

import pytest

from fpl_smart.my_team import TeamState
from fpl_smart.solver import best_eleven, chip_windows, hold_value, solve

from conftest import check_squad, empty_state, initial_plan


def test_initial_squad_is_legal_and_affordable(market, opts):
    players, xp = market
    plan = initial_plan(players, xp, opts)
    w = plan.weeks[0]
    check_squad(w, players)
    assert players.loc[w.squad, "now_cost"].sum() + w.bank == 1000
    assert w.hits == 0 and w.chip == "wildcard"
    for later in plan.weeks[1:]:
        check_squad(later, players)


def test_captain_is_best_projected_starter(market, opts):
    players, xp = market
    w = initial_plan(players, xp, opts).weeks[0]
    assert xp.loc[w.captain, 1] == pytest.approx(xp.loc[w.lineup, 1].max())


def test_hold_makes_no_transfers(market, opts, squad_state):
    players, xp = market
    plan = hold_value(xp, players, squad_state, 3, opts)
    for w in plan.weeks:
        assert not w.transfers_in and not w.transfers_out and w.hits == 0
        assert sorted(w.squad) == sorted(squad_state.squad)


def _with_duds(market, state, n, stars=False):
    """Copy of the projections where `n` owned players are worthless (midfielders, or the best players)."""
    players, xp = market
    if stars:
        duds = list(xp.loc[state.squad, 1].nlargest(n).index)
    else:
        duds = [p for p in state.squad if players.at[p, "pos"] == 3][:n]
    xp = xp.copy()
    xp.loc[duds] = 0.0
    return xp, duds


def test_free_transfer_used_without_a_hit(market, opts, squad_state):
    players, _ = market
    xp, duds = _with_duds(market, squad_state, 1)
    plan = solve(xp, players, squad_state, 3, opts)
    w = plan.weeks[0]
    assert duds[0] in w.transfers_out and w.hits == 0 and len(w.transfers_in) == 1
    check_squad(w, players)


def test_hit_taken_only_when_it_pays(market, opts, squad_state):
    players, _ = market
    xp, duds = _with_duds(market, squad_state, 6, stars=True)       # too many to hide on the bench
    plan = solve(xp, players, squad_state, 3, opts)
    assert sum(w.hits for w in plan.weeks) >= 1
    assert {p for w in plan.weeks for p in w.transfers_out} <= set(duds)
    for w in plan.weeks:
        assert w.hits == max(len(w.transfers_in) - w.free_transfers, 0)
    none = solve(xp, players, squad_state, 3, replace(opts, max_hits_per_gw=0))
    assert all(w.hits == 0 and len(w.transfers_in) <= w.free_transfers for w in none.weeks)
    assert plan.objective > none.objective
    # a small upgrade is not worth four points
    players2, xp2 = market
    calm = solve(xp2, players2, squad_state, 1, opts)
    assert calm.weeks[0].hits == 0


def test_free_transfers_roll_over_and_cap(market, opts, squad_state):
    players, xp = market
    plan = solve(xp, players, replace(squad_state, free_transfers=4), 4, replace(opts, no_transfers=True))
    assert [w.free_transfers for w in plan.weeks] == [4, 5, 5, 5]


def test_money_is_conserved(market, opts, squad_state):
    players, _ = market
    xp, _ = _with_duds(market, squad_state, 2)
    plan = solve(xp, players, squad_state, 3, opts)
    cost = players["now_cost"]
    bank = squad_state.bank
    for w in plan.weeks:
        bank += sum(squad_state.sell_price.get(p, cost[p]) for p in w.transfers_out) - cost[w.transfers_in].sum()
        assert w.bank == bank and bank >= 0


def test_banned_and_locked(market, opts, squad_state):
    players, _ = market
    xp, duds = _with_duds(market, squad_state, 1)
    free = solve(xp, players, squad_state, 3, opts)
    target = free.weeks[0].transfers_in[0]
    banned = solve(xp, players, squad_state, 3, replace(opts, banned=(target,)))
    assert all(target not in w.transfers_in for w in banned.weeks)
    locked = solve(xp, players, squad_state, 3, replace(opts, locked=(duds[0],)))
    assert all(duds[0] in w.squad for w in locked.weeks)


def test_wildcard_week_is_free_and_keeps_banked_transfers(market, opts, squad_state):
    players, _ = market
    xp, _ = _with_duds(market, squad_state, 4)
    st = replace(squad_state, free_transfers=2)
    o = replace(opts, use_chips=True, chips=[{"name": "wildcard", "gws": [1, 2, 3]}], wildcard_gw=1)
    plan = solve(xp, players, st, 3, o)
    w = plan.weeks[0]
    assert w.chip == "wildcard" and w.hits == 0 and len(w.transfers_in) >= 4
    assert plan.weeks[1].free_transfers == 2
    check_squad(w, players)


def test_free_hit_squad_lasts_one_week(market, opts, squad_state):
    players, xp = market
    o = replace(opts, use_chips=True, chips=[{"name": "freehit", "gws": [1, 2, 3]}], freehit_gw=2, no_transfers=True)
    plan = solve(xp, players, squad_state, 3, o)
    own = sorted(squad_state.squad)
    assert sorted(plan.weeks[0].squad) == own and sorted(plan.weeks[2].squad) == own
    fh = plan.weeks[1]
    assert fh.chip == "freehit" and sorted(fh.squad) != own
    check_squad(fh, players)
    budget = squad_state.bank + sum(squad_state.sell_price.values())
    assert players.loc[fh.squad, "now_cost"].sum() <= budget
    assert plan.fh_gain > 0
    assert plan.weeks[2].free_transfers == plan.weeks[1].free_transfers


def test_expiring_chips_are_played_once_and_counted(market, opts, squad_state):
    players, xp = market
    chips = [{"name": "bboost", "gws": [1, 2, 3]}, {"name": "3xc", "gws": [1, 2, 3]}]
    plan = solve(xp, players, squad_state, 3, replace(opts, use_chips=True, chips=chips, no_transfers=True))
    assert sorted(plan.chips) == ["3xc", "bboost"] and plan.chips["3xc"] != plan.chips["bboost"]
    for w in plan.weeks:
        g = w.gw
        expect = xp.loc[w.lineup, g].sum() + xp.loc[w.captain, g] * (2 if w.chip == "3xc" else 1)
        if w.chip == "bboost":
            expect += xp.loc[w.bench, g].sum()
        assert w.xp == pytest.approx(expect, abs=0.02)


def test_chip_held_when_it_outlives_the_plan_and_gain_is_small(market, opts, squad_state):
    players, xp = market
    chips = [{"name": "bboost", "gws": list(range(1, 20))}]
    o = replace(opts, use_chips=True, chips=chips, no_transfers=True, chip_reserve={"bboost": 1000.0})
    assert solve(xp, players, squad_state, 3, o).chips == {}


def test_invalid_chip_weeks_are_rejected(market, opts, squad_state):
    players, xp = market
    with pytest.raises(ValueError):
        solve(xp, players, squad_state, 3, replace(opts, wildcard_gw=1))            # no wildcard available
    chips = [{"name": "wildcard", "gws": [1, 2]}, {"name": "freehit", "gws": [1, 2]}]
    with pytest.raises(ValueError):
        solve(xp, players, squad_state, 3, replace(opts, use_chips=True, chips=chips, wildcard_gw=1, freehit_gw=1))


def test_warm_start_never_makes_things_worse(market, opts, squad_state):
    players, _ = market
    xp, _ = _with_duds(market, squad_state, 2)
    a = solve(xp, players, squad_state, 3, opts)
    b = solve(xp, players, squad_state, 3, replace(opts, pool=a.pool), warm=a.values)
    assert b.objective >= a.objective - 1e-6


def test_chip_windows_uses_the_open_copy():
    avail = [{"name": "wildcard", "gws": list(range(11, 20))}, {"name": "wildcard", "gws": list(range(20, 39))}]
    win = chip_windows(avail, list(range(11, 23)))
    assert win["wildcard"] == {"gws": list(range(11, 20)), "expires": True}
    assert win["freehit"] == {"gws": [], "expires": False}
    later = chip_windows(avail[1:], list(range(18, 27)))
    assert later["wildcard"]["gws"] == list(range(20, 27)) and not later["wildcard"]["expires"]


def test_best_eleven_respects_formation():
    pos = {**{i: 1 for i in (1, 2)}, **{i: 2 for i in range(3, 8)}, **{i: 3 for i in range(8, 13)},
           **{i: 4 for i in range(13, 16)}}
    xp = {p: 1.0 for p in pos}
    xp.update({8: 9, 9: 8, 10: 7, 11: 6, 12: 5, 13: 9, 14: 8, 15: 7})     # strong mids and forwards
    xp.update({3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1})                   # hopeless defenders: still need three
    # keeper, three defenders, then the seven best of the eight attackers; captain doubled
    assert best_eleven(list(pos), xp, pos) == pytest.approx(1 + 0.3 + (35 - 5) + 24 + 9)
