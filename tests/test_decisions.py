"""Chip search, robustness, report and season replay on the synthetic market."""
from dataclasses import replace

import numpy as np
import pandas as pd

from fpl_smart import backtest, optimise, planner, report
from fpl_smart.solver import Plan, hold_value, solve
from fpl_smart.xp import Projection

from conftest import check_squad

CHIPS = [{"name": n, "gws": [1, 2, 3]} for n in ("wildcard", "freehit", "bboost", "3xc")]


def _search(market, opts, state, **kw):
    players, xp = market
    o = replace(opts, use_chips=True, chips=CHIPS, time_limit=20)
    return optimise.chip_search(xp, players, state, 3, o, workers=1, search_time=10, polish_time=5, top=2, **kw)


def test_chip_search_tries_every_week_and_ranks(market, opts, squad_state):
    cs = _search(market, opts, squad_state)
    tried = {(p.wildcard_gw, p.freehit_gw) for p in cs.scenarios}
    assert {(None, None), (1, None), (2, None), (3, None), (None, 1), (None, 2), (None, 3)} <= tried
    assert any(w and f for w, f in tried)                       # combinations were explored
    scores = [p.objective for p in cs.scenarios]
    assert scores == sorted(scores, reverse=True)
    assert cs.best.freehit_gw and cs.best.fh_gain > 0           # an expiring free hit is worth playing
    assert {"freehit", "bboost", "3xc"} <= set(cs.best.chips)
    assert len(set(cs.best.chips.values())) == len(cs.best.chips)   # never two chips in one gameweek
    assert cs.best.objective >= cs.without_squad_chips().objective


def test_choose_prefers_keeping_chips_on_a_near_tie():
    def fake(score, wc, fh):
        week = type("W", (), {"gw": 5})()
        return Plan([week], score, "Optimal", 0.0, 0, wildcard_gw=wc, freehit_gw=fh)
    cs = optimise.ChipSearch([fake(100.0, 5, None), fake(99.2, 7, None), fake(90.0, None, None)], {})
    assert cs.choose(margin=1.5).wildcard_gw == 7          # close enough: keep the chip this week
    assert cs.choose(margin=0.5).wildcard_gw == 5          # clear gain: play it


def test_perturb_keeps_shape_and_scale(market):
    players, xp = market
    rng = np.random.default_rng(1)
    draws = [optimise.perturb(xp, players, rng) for _ in range(30)]
    assert draws[0].shape == xp.shape and (draws[0].index == xp.index).all()
    ratio = np.mean([d.to_numpy().sum() for d in draws]) / xp.to_numpy().sum()
    assert 0.97 < ratio < 1.03
    assert not np.allclose(draws[0].to_numpy(), draws[1].to_numpy())


def test_robustness_reports_shares(market, opts, squad_state):
    players, xp = market
    xp = xp.copy()
    dud = [p for p in squad_state.squad if players.at[p, "pos"] == 4][0]
    xp.loc[dud] = 0.0                                           # an obvious sale
    plan = solve(xp, players, squad_state, 3, opts)
    rb = optimise.robustness(xp, players, squad_state, plan, 3, opts, n=6, workers=1, time_limit=10)
    assert rb.n == 6 and rb.sold.index[0] == dud and rb.sold.iloc[0] >= 0.5
    assert abs(rb.captain.sum() - 1) < 1e-9 and 0 <= rb.base_kept <= 1
    assert (rb.squad <= 1).all() and abs(rb.squad.sum() - 15) < 1e-9


def _projection(market):
    players, xp = market
    pl = players.assign(status="a", chance=np.nan, news="", own=5.0)
    teams = sorted(pl["team"].unique())
    rows = []
    for g in xp.columns:
        for i, t in enumerate(teams):
            opp = teams[(i + g) % len(teams)] if teams[(i + g) % len(teams)] != t else teams[(i + 1) % len(teams)]
            rows.append({"team": t, "opp": opp, "gw": g, "home": i % 2 == 0, "has_odds": g == 1,
                         "kickoff": pd.Timestamp("2026-08-15", tz="UTC") + pd.Timedelta(days=7 * g)})
    return Projection(pl, xp, pd.DataFrame(), pd.DataFrame(rows), None, 1)


def test_report_renders_every_section(market, opts, squad_state):
    players, xp = market
    P = _projection(market)
    cs = _search(market, opts, squad_state)
    plan = cs.choose()
    hold = hold_value(xp, players, squad_state, 3, opts)
    rb = optimise.robustness(xp, players, squad_state, plan, 3, replace(opts, use_chips=True, chips=CHIPS),
                             n=3, workers=1, time_limit=5)
    rec = planner.Recommendation(P, squad_state, plan, hold, 3, cs, rb, pd.Timestamp("2026-08-21 17:30", tz="UTC"),
                                 {"team_name": "Test XI", "generated": "2026-08-14 09:00+00:00"})
    text = report.render(rec)
    for heading in ("# Gameweek 1 plan for Test XI", "## What to do", "## How sure is this", "## The weeks ahead",
                    "## Chip timing", "## Your squad", "## Best projected players you do not own", "## Notes"):
        assert heading in text
    assert "nan" not in text.lower() and "None" not in text
    assert report.summary(rec).startswith("GW1:")


def test_season_replay_runs_and_keeps_the_rules(market):
    players, xp = market
    rng = np.random.default_rng(3)
    data = {}
    for g in (1, 2, 3, 4):
        pts = (xp[g] + rng.normal(0, 2.5, len(xp))).round().clip(lower=0)
        mins = pd.Series(np.where(rng.random(len(xp)) < 0.9, 90, 0), index=xp.index)
        data[g] = backtest.GWData(g, xp.loc[:, [c for c in xp.columns if c >= g]], players,
                                  (pts * (mins > 0)).to_dict(), mins.to_dict())
    from conftest import SMALL
    from fpl_smart.solver import SolverOptions
    base = SolverOptions(per_pos_keep=dict(SMALL))
    res = {s: backtest.simulate(data, "test", s, horizon=3, opt=base, initial_horizon=3) for s in backtest.STRATEGIES}
    for r in res.values():
        assert list(r.weeks.index) == [1, 2, 3, 4] and r.total == r.weeks["points"].sum()
        assert (r.weeks["value"] >= 0).all() and r.weeks.loc[1, "transfers"] == 0
    assert res["hold"].weeks["transfers"].sum() == 0 and res["hold"].weeks["hits"].sum() == 0
    assert "strategy" in backtest.render(list(res.values()))
