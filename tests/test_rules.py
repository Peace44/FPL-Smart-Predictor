"""Small pure functions: prices, free transfers, odds maths, availability, names."""
import numpy as np
import pandas as pd
import pytest

from fpl_smart import market, my_team, planner, xp
from fpl_smart.my_team import TeamState


def test_selling_price_keeps_half_the_profit_rounded_down():
    assert my_team.selling_price(50, 50) == 50
    assert my_team.selling_price(50, 51) == 50
    assert my_team.selling_price(50, 52) == 51
    assert my_team.selling_price(50, 53) == 51
    assert my_team.selling_price(50, 47) == 47


def _hist(weeks, chips=()):
    return {"current": [{"event": g, "event_transfers": n, "event_transfers_cost": c} for g, n, c in weeks],
            "chips": [{"event": g, "name": n} for g, n in chips]}


def test_free_transfers_replayed_from_history():
    f = my_team.free_transfers_from_history
    assert f(_hist([(1, 0, 0)]), 1, 2) == 1                                  # nothing banked in week one
    assert f(_hist([(1, 0, 0), (2, 0, 0), (3, 0, 0)]), 1, 4) == 3            # two rolled
    assert f(_hist([(1, 0, 0), (2, 1, 0), (3, 0, 0)]), 1, 4) == 2            # used one, rolled one
    assert f(_hist([(1, 0, 0), (2, 3, 8)]), 1, 3) == 1                       # took a hit: back to one
    assert f(_hist([(1, 0, 0)] + [(g, 0, 0) for g in range(2, 12)]), 1, 12) == 5      # capped at five
    assert f(_hist([(1, 0, 0), (2, 0, 0), (3, 9, 0), (4, 0, 0)], [(3, "wildcard")]), 1, 5) == 3


def test_devig_and_implied_goals_reproduce_the_odds():
    p = market.devig(1.8, 3.9, 4.6)
    assert sum(p) == pytest.approx(1.0)
    lh, la = market.implied_lambdas(p[0], p[2], 0.55)
    h, d, a = market._outcome_probs(lh, la)
    assert h - a == pytest.approx(p[0] - p[2], abs=1e-4)
    from scipy.stats import poisson
    assert 1 - poisson.cdf(2, lh + la) == pytest.approx(0.55, abs=1e-4)
    assert lh > la > 0


def test_return_date_parsing():
    now = pd.Timestamp("2026-11-13", tz="UTC")
    assert xp.parse_return_date("Knee injury - Expected back 05 Dec", now) == pd.Timestamp("2026-12-05", tz="UTC")
    assert xp.parse_return_date("Suspended until 14 Jan", now) == pd.Timestamp("2027-01-14", tz="UTC")
    assert xp.parse_return_date("Knock - 75% chance of playing", now) is None


def test_availability_from_flags():
    now = pd.Timestamp("2026-11-13", tz="UTC")
    dates = {11: pd.Timestamp("2026-11-21", tz="UTC"), 12: pd.Timestamp("2026-11-28", tz="UTC"),
             13: pd.Timestamp("2026-12-05", tz="UTC")}
    el = pd.DataFrame([
        {"status": "a", "news": "", "chance_of_playing_next_round": None},
        {"status": "d", "news": "Knock - 75% chance of playing", "chance_of_playing_next_round": 75},
        {"status": "i", "news": "Hamstring - Expected back 01 Dec", "chance_of_playing_next_round": 0},
        {"status": "u", "news": "Has joined another club", "chance_of_playing_next_round": 0},
        {"status": "s", "news": "Suspended for one match", "chance_of_playing_next_round": 0},
    ], index=[1, 2, 3, 4, 5])
    a = xp.availability(el, dates, now)
    assert a.loc[1].tolist() == [1.0, 1.0, 1.0]
    assert a.at[2, 11] == 0.75 and a.at[2, 11] < a.at[2, 12] < a.at[2, 13] < 1
    assert a.loc[3, [11, 12]].tolist() == [0.0, 0.0] and 0 < a.at[3, 13] < 1
    assert a.loc[4].tolist() == [0.0, 0.0, 0.0]
    assert a.at[5, 11] == 0.0 and a.at[5, 12] == 1.0


def test_unflagged_uplift_stays_coherent():
    mins = pd.DataFrame({"p_play": [0.95, 0.6, 0.2], "p60": [0.9, 0.4, 0.05], "emin": [80.0, 40.0, 6.0]})
    out = xp._unflagged_uplift(mins, np.array([True, True, True]), np.array([1, 1, 1]))
    eps = 1e-9
    assert (out["p_play"] >= mins["p_play"] - eps).all() and (out["p_play"] <= 1).all()
    assert (out["p60"] <= out["p_play"] + eps).all() and (out["emin"] <= 90 * out["p_play"] + eps).all()
    assert out.at[0, "p_play"] > mins.at[0, "p_play"] + 0.01          # a nailed starter with no flag
    assert out.at[2, "p_play"] == pytest.approx(mins.at[2, "p_play"])  # a fringe player gains nothing


def test_resolve_players():
    pl = pd.DataFrame({"name": ["Ødegaard", "Martinez", "Martinez", "Groß"], "team": ["ARS", "AVL", "MUN", "BHA"]},
                      index=[10, 20, 30, 40])
    assert planner.resolve_players(["odegaard", "Gross", "30"], pl) == [10, 40, 30]
    assert planner.resolve_players(["Martinez (MUN)"], pl) == [30]
    with pytest.raises(ValueError, match="several"):
        planner.resolve_players(["Martinez"], pl)
    with pytest.raises(ValueError, match="did you mean"):
        planner.resolve_players(["Marti"], pl)


def test_auto_horizon_stretches_past_an_expiring_chip():
    def st(gw, chips):
        return TeamState(0, gw, [], {}, {}, 0, 1, chips)
    first = [{"name": "wildcard", "gws": list(range(11, 20))}, {"name": "wildcard", "gws": list(range(20, 39))}]
    assert planner.auto_horizon(st(11, first)) == 12
    assert planner.auto_horizon(st(15, [{"name": "wildcard", "gws": list(range(15, 20))}])) == 9
    assert planner.auto_horizon(st(11, [])) == 9
    assert planner.auto_horizon(st(36, [])) == 3
