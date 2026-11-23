"""The gameweek review's arithmetic, the limiting-factor choice, the tuned-settings store and the
improver's adoption rule. No network."""
import json

import numpy as np
import pandas as pd
import pytest

from fpl_smart import evaluate as ev
from fpl_smart import evidence, gwreview, improve, tuning, xp
from fpl_smart.scoring import score_gw

# ids 1-2 keepers, 3-7 defenders, 8-12 midfielders, 13-15 forwards
POS = {i: 1 if i <= 2 else 2 if i <= 7 else 3 if i <= 12 else 4 for i in range(1, 16)}
LINEUP = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
BENCH = [2, 7, 12, 15]


def picks(captain=13, vice=8):
    return [{"element": p, "position": i + 1, "is_captain": p == captain, "is_vice_captain": p == vice}
            for i, p in enumerate(LINEUP + BENCH)]


def points(**over):
    pts = {p: 2 for p in POS}
    pts.update({int(k[1:]): v for k, v in over.items()})
    return pts


def minutes(pts, absent=()):
    return {p: (0 if p in absent else 90) for p in pts}


def test_best_xi_respects_formation():
    pts = points(p2=9, p3=1, p4=1, p5=1, p6=1, p7=1, p13=10, p14=10, p15=10)
    total, ids = gwreview.best_xi(list(POS), pts, POS)
    n = pd.Series([POS[p] for p in ids]).value_counts().to_dict()
    assert len(ids) == 11 and n[1] == 1 and n[2] >= 3 and n[3] >= 2 and 1 <= n[4] <= 3
    assert 2 in ids and total == 9 + 3 * 1 + 4 * 2 + 3 * 10   # keeper 9, the three defenders the rules need


@pytest.mark.parametrize("chip", [None, "bboost", "3xc"])
def test_parts_add_up_to_the_score(chip):
    pts = points(p13=12, p15=7, p7=5, p2=3)
    mins = minutes(pts, absent=(9,))
    t = gwreview.team_parts(picks(), chip, 4, pts, mins, POS)
    sc = score_gw(LINEUP, BENCH, 13, 8, pts, mins, POS, chip)
    assert t["net"] == pytest.approx(sc.points - 4)
    assert t["net"] == pytest.approx(t["best_xi"] - t["selection"] + t["captain_extra"] + t["chip_extra"] - t["hits"])
    assert t["selection"] >= 0
    if chip == "bboost":
        assert t["selection"] == 0 and t["chip_extra"] == pytest.approx(sum(pts.values()) - t["best_xi"])
    if chip == "3xc":
        assert t["chip_extra"] == t["captain_extra"] == 12


def test_vice_takes_the_armband_and_captain_loss_is_hindsight():
    pts = points(p8=6, p14=11)
    t = gwreview.team_parts(picks(captain=13, vice=8), None, 0, pts, minutes(pts, absent=(13,)), POS)
    assert t["armband"] == 8 and t["captain_extra"] == 6
    assert t["best_captain"] == 11 and t["captain_loss"] == 5


def test_deficits_sum_to_the_gap():
    pts = points(p13=12, p15=7, p7=5)
    mins = minutes(pts)
    ours = gwreview.team_parts(picks(), None, 4, pts, mins, POS)
    elite = [gwreview.team_parts(picks(captain=c), ch, h, pts, mins, POS)
             for c, ch, h in ((13, None, 0), (8, "bboost", 0), (14, "3xc", 8))]
    d = gwreview.deficits(ours, elite)
    assert sum(d[f] for f in gwreview.FACTORS) == pytest.approx(d["total"])
    assert d["total"] == pytest.approx(np.mean([e["net"] for e in elite]) - ours["net"])


def test_limiting_factor_choice():
    win = [{"gw": 1, "deficits": {"squad": 1, "selection": 5, "captaincy": 2, "chips": 0, "hits": 0}},
           {"gw": 2, "deficits": {"squad": 9, "selection": -1, "captaincy": 2, "chips": 0, "hits": 0}}]
    assert gwreview.choose_factor(win)[0] == "squad"
    assert gwreview.choose_factor(win, operations_lost=4)[0] == "operations"
    assert gwreview.choose_factor([])[0] is None


def test_window_skips_weeks_the_engine_did_not_manage():
    reviews = [{"gw": g, "engine_managed": g != 3, "deficits": {"squad": 1}} for g in range(1, 10)]
    assert [r["gw"] for r in gwreview.window(reviews, 8, n=3)] == [6, 7, 8]
    assert [r["gw"] for r in gwreview.window(reviews, 4, n=6)] == [1, 2, 4]


def test_review_data_round_trip_keeps_notes():
    rv = gwreview.Review(7, True, {"gw": 7, "final": True, "x": np.float64(1.5), "ids": {1, 2}}, ["# Gameweek 7 review"])
    text = gwreview.render(rv, "Bench order looked odd.\nCheck the keeper.")
    assert gwreview.parse(text)["x"] == 1.5
    assert gwreview.existing_notes(text) == "Bench order looked odd.\nCheck the keeper."
    again = gwreview.render(rv, gwreview.existing_notes(text))
    assert gwreview.existing_notes(again) == gwreview.existing_notes(text)


def test_milestones_track_progress_towards_first_place():
    """The user asked for milestones in every review; this keeps them from being cleaned away."""
    def h(gw, pts, rank, overall, cost=0):
        return {"event": gw, "points": pts, "event_transfers_cost": cost, "rank": rank, "overall_rank": overall}
    hist = [h(1, 60, 900_000, 900_000), h(2, 40, 3_000_000, 1_200_000), h(3, 70, 400_000, 800_000, cost=4),
            h(4, 80, 90_000, 450_000)]
    averages = {1: 50, 2: 45, 3: 60, 4: 55}

    def engine(gw, net, avg, elite):
        return {"gw": gw, "engine_managed": True, "summary": {"net": net, "average": avg},
                "elite": {"avg": {"net": elite}}}
    earlier = [engine(3, 66, 60, 70), engine(5, 99, 50, 60)]      # GW5 is later than the week reviewed
    lines = gwreview.milestones(hist, 4, earlier, engine(4, 80, 55, 75), averages)
    assert lines == ["New season-best overall rank: 450,000.",
                     "Entered the top 500,000.",
                     "Green arrows in a row: 2.",
                     "Next milestone: the top 250,000 (200,000 places to go).",
                     "Best gameweek rank of the season: 90,000.",
                     "Beat the game average in 3 of 4 gameweeks.",
                     "Since the engine took over (2 gameweeks): 146 points, +31 against the average, +1 against the top 50."]
    down = gwreview.milestones(hist[:2], 2, [], {}, averages)
    assert down == ["Season-best overall rank: 900,000 (GW1).", "Dropped out of the top 1,000,000.",
                    "Next milestone: the top 1,000,000 (200,000 places to go).", "Beat the game average in 1 of 2 gameweeks."]


def test_elite_entries_keep_at_most_elite_n_when_last_ranks_tie(monkeypatch):
    n = gwreview.ELITE_N
    rows = [{"entry": i, "last_rank": min(i, n), "rank_sort": i} for i in range(1, n + 2)]  # n and n+1 tie at n
    monkeypatch.setattr(gwreview, "_standings_page",
                        lambda page, max_age_h: {"standings": {"results": rows if page == 1 else []}})
    ids, how = gwreview.elite_entries(12, None, now_is_gw=True)
    assert ids == list(range(1, n + 1))
    assert how == f"top {n} before GW12 ({n} found)"


# ----------------------------------------------------------------------------- tuning store
@pytest.fixture()
def tuned(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "DIR", tmp_path)
    monkeypatch.setattr(tuning, "PARAMS", tmp_path / "params.json")
    monkeypatch.setattr(tuning, "LEDGER", tmp_path / "ledger.md")
    return tmp_path


def test_set_value_revert_and_bounds(tuned):
    assert tuning.player_params().k_goal == 8.0 and tuning.fingerprint() == ""
    ch = tuning.set_value("player.k_goal", 5.7143, "test")
    assert tuning.player_params().k_goal == 5.7143 and ch["old"] == 8.0
    fp = tuning.fingerprint()
    assert fp and fp != tuning.fingerprint(("solver",))
    tuning.revert(ch["id"])
    assert tuning.player_params().k_goal == 8.0 and len(tuning.history()) == 2
    with pytest.raises(ValueError):
        tuning.set_value("player.k_goal", 99.0, "too far")
    tuning.set_value("solver.decay", 0.8, "test")
    assert tuning.solver_values() == {"decay": 0.8}
    tuning.append_ledger(["## GW1", "- Result: test"])
    assert "Self-improvement ledger" in (tuned / "ledger.md").read_text()


def test_neighbours_stay_in_bounds():
    k = tuning.KNOBS["player.xg_blend"]
    assert k.neighbours(1.0) == [0.9]
    assert tuning.KNOBS["player.k_goal"].neighbours(8.0) == [5.7143, 11.2]


# ----------------------------------------------------------------------------- availability
def test_unflagged_play_matches_the_uplift():
    p = np.array([0.3, 0.7, 0.95, 0.95])
    clear = np.array([True, True, True, False])
    h = np.array([1, 1, 1, 1])
    mins = pd.DataFrame({"p_play": p, "p60": p * 0.8, "emin": p * 70})
    up = xp._unflagged_uplift(mins, clear, h)
    assert np.allclose(up["p_play"], xp.unflagged_play(p, clear, h))
    assert up["p_play"].iloc[0] == pytest.approx(0.3) and up["p_play"].iloc[3] == pytest.approx(0.95)
    assert xp.unflagged_play(p, clear, h, 0.0) == pytest.approx(p)


def test_doubtful_scale():
    assert xp.doubtful_chance(0.75, 1.0) == 0.75
    assert xp.doubtful_chance(0.75, 0.5) == pytest.approx(0.875)
    el = pd.DataFrame({"status": ["d"], "news": ["Knock - 75% chance of playing"], "chance_of_playing_next_round": [75.0]},
                      index=[1])
    now = pd.Timestamp("2026-11-20", tz="UTC")
    dates = {12: pd.Timestamp("2026-11-28", tz="UTC"), 13: pd.Timestamp("2026-12-02", tz="UTC")}
    assert xp.availability(el, dates, now).at[1, 12] == pytest.approx(0.75)
    assert xp.availability(el, dates, now, 0.6).at[1, 12] == pytest.approx(0.85)


# ----------------------------------------------------------------------------- the adoption rule
def _frame(values, season="2026-27"):
    return pd.DataFrame({"mse_rel": values, "top30": np.full(len(values), 5.0)},
                        index=pd.MultiIndex.from_tuples([(season, g) for g in range(1, len(values) + 1)]))


def test_compare_direction_and_adoption():
    rng = np.random.default_rng(1)
    base = _frame(8 + rng.normal(0, 0.5, 40))
    better = base.copy()
    better["mse_rel"] -= 0.05 + rng.normal(0, 0.01, 40)        # lower error = better
    c = ev.compare(base, better, "mse_rel")
    assert c.diff > 0 and c.z > 3 and c.n == 40
    t = improve.Trial("player.k_goal", 8.0, 5.7, c, ev.compare(base, better, "top30"))
    assert t.passes
    noisy = base.copy()
    noisy["mse_rel"] -= rng.normal(0.01, 0.5, 40)
    assert not improve.Trial("player.k_goal", 8.0, 5.7, ev.compare(base, noisy, "mse_rel")).passes
    tiny = base.copy()
    tiny["mse_rel"] -= 0.001 + rng.normal(0, 0.0002, 40)        # consistent, but too small to matter
    small = ev.compare(base, tiny, "mse_rel")
    assert small.z > 3 and not improve.Trial("player.k_bps", 8.0, 5.3, small).passes
    worse_guard = better.copy()
    worse_guard["top30"] -= 0.3 + rng.normal(0, 0.05, 40)
    assert not improve.Trial("player.k_goal", 8.0, 5.7, c, ev.compare(base, worse_guard, "top30")).passes


def test_every_factor_has_ideas_and_every_experiment_is_valid():
    for f in gwreview.FACTORS + ("operations",):
        assert improve.STRUCTURAL[f]
    for e in improve.EXPERIMENTS.values():
        assert all(k in tuning.KNOBS for k in e.knobs)
        assert e.primary in ev.METRICS and (e.guard is None or e.guard in ev.METRICS)
    assert json.dumps(gwreview.LEVERS) and set(gwreview.LEVERS) == set(improve.STRUCTURAL)


def test_every_experiment_runs_whatever_the_factor(tuned, tmp_path, monkeypatch):
    review = tmp_path / "gw11_review.md"
    review.write_text("")
    monkeypatch.setattr(gwreview, "parse", lambda text: {"gw": 11, "final": True})
    monkeypatch.setattr(gwreview, "load_reviews", lambda: [])
    monkeypatch.setattr(gwreview, "certainty", lambda win, f: {"settled": False})
    monkeypatch.setattr(gwreview, "certainty_text", lambda f, cert: "")
    monkeypatch.setattr(improve.evidence, "pool", lambda reviews: [])
    rng = np.random.default_rng(2)
    base = pd.DataFrame({m: 5 + rng.normal(0, 0.1, 20) for m in ev.METRICS if m not in ("logloss", "points")},
                        index=pd.MultiIndex.from_tuples([("2026-27", g) for g in range(1, 21)]))
    calls = []

    def points_model(b, player=None, team=None):      # every candidate clearly worse: nothing is kept
        calls.append(player or team)
        noise = rng.normal(0, 0.01, len(base))
        return base if not (player or team) else base.assign(
            mse_rel=base["mse_rel"] + 0.1 + noise, top30=base["top30"] - 0.1 + noise, top10=base["top10"] - 0.1 + noise)
    monkeypatch.setattr(improve.ev, "base", lambda **k: None)
    monkeypatch.setattr(improve.ev, "points_model", points_model)
    monkeypatch.setattr(improve.ev, "availability_rows", lambda: pd.DataFrame({"gw": []}))
    seen, real_render = {}, improve.render

    def render(*a, **k):
        seen["outcomes"] = a[6]
        return real_render(*a, **k)
    monkeypatch.setattr(improve, "render", render)
    assert improve.run(str(review), dry_run=True, factor="chips", heavy=False, log=lambda *_: None) == 0
    outs = {o.experiment: o for o in seen["outcomes"]}
    assert list(outs) == ["points_model", "top_end", "availability", "planning"]   # chips used to run none
    assert len(calls) == 1 + len(outs["points_model"].trials)                       # one projection per candidate
    te = outs["top_end"].trials
    assert te and {t.knob for t in te} <= set(improve.EXPERIMENTS["top_end"].knobs)
    assert all(t.primary.metric == "top10" and t.guard.metric == "mse_rel" for t in te)
    assert outs["availability"].skipped and outs["planning"].skipped == "heavy replay skipped (--light)"
    assert not any(o.adopted for o in outs.values())


@pytest.mark.parametrize("plan_within_h,hours_left,runs", [
    (48.0, 51.0, True), (48.0, 49.0, False), (24.0, 27.0, True), (24.0, 25.0, False)])
def test_improve_waits_for_the_planning_window(tmp_path, monkeypatch, plan_within_h, hours_left, runs):
    review = tmp_path / "gw11_review.md"
    review.write_text("")
    deadline = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=hours_left)
    monkeypatch.setattr(improve, "latest_final_review", lambda: review)
    monkeypatch.setattr(improve, "_state", lambda: {})
    monkeypatch.setattr(gwreview, "parse", lambda text: {"gw": 11, "final": True})
    monkeypatch.setattr(improve.fpl_api, "bootstrap", lambda max_age_h: {"events": [
        {"id": 12, "is_next": True, "deadline_time": deadline.isoformat()}]})
    monkeypatch.setattr(improve.Settings, "from_env", classmethod(lambda cls: cls(plan_within_h=plan_within_h)))
    assert (improve.due(log=lambda *_: None) == review) is runs


# ----------------------------------------------------------------------------- what a scheduled run reports
def _trial(z, knob="player.k_goal"):
    return improve.Trial(knob, 8.0, 5.7, ev.Comparison("mse_rel", 0.001 * z, 0.001, 40, {}))


@pytest.mark.parametrize("outcomes,says,not_says", [
    ([improve.Outcome("points_model", [_trial(1.2)])], "the limit is in the code", "Not run"),
    ([improve.Outcome("points_model", [_trial(1.2)]), improve.Outcome("availability", skipped="needs 3; has 1")],
     "Not run: availability: needs 3; has 1", "the limit is in the code"),
    ([improve.Outcome("availability", skipped="needs 3; has 1")], "No test could run", "limit is in the code"),
])
def test_unchanged_report_says_why_and_names_the_skill(outcomes, says, not_says):
    lines = improve.result_lines("chips", outcomes, None, "reports/gw11_improvement.md")
    text = " ".join(lines)
    assert says in text and not_says not in text
    assert "/fpl-improve" in text and "reports/gw11_improvement.md" in text
    if any(o.trials for o in outcomes):
        assert "closest: player.k_goal 8 → 5.7 at +1.2 s.e." in text


def test_changed_report_needs_nothing():
    t = _trial(4.0)
    text = " ".join(improve.result_lines("squad", [improve.Outcome("points_model", [t], adopted=t)], t, "x.md"))
    assert text.startswith("Changed player.k_goal 8 → 5.7") and "/fpl-improve" not in text


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(improve, "STATE", tmp_path / "improve-state.json")
    monkeypatch.setattr(improve, "REPORTS", tmp_path)
    return tmp_path


def test_missed_gameweeks(state):
    assert improve.missed(12) == []                                 # never run: earlier reviews are not "missed"
    improve._save_state(done_gw=9, failed={"gw": 11, "error": "ValueError"})
    assert improve.missed(12) == [(10, "no scheduled run came while it was due"),
                                  (11, "its runs ended with ValueError")]
    assert improve.missed(10) == [] and improve.missed(8) == []
    lines = improve.result_lines("squad", [], None, "x.md", improve.missed(12))
    assert "GW10 was never worked on: no scheduled run came while it was due." in lines


def test_scheduled_failure_is_reported_once(state, monkeypatch):
    review = state / "gw11_review.md"
    review.write_text("")
    monkeypatch.setattr(gwreview, "parse", lambda text: {"gw": 11, "final": True})
    monkeypatch.setattr(improve, "due", lambda log: review)
    sent, delivered = [], [False]
    monkeypatch.setattr(improve, "_report", lambda log, title, lines=(), problems=(), kind=None: (
        sent.append((title, kind, lines)) or delivered[0]))
    err = [ValueError("bad data")]

    def boom(*a, **k):
        raise err[0]
    monkeypatch.setattr(improve, "_run", boom)
    quiet = dict(if_due=True, log=lambda *_: None)

    assert improve.run(**quiet) == 1 and len(sent) == 1            # Discord unreachable: not marked reported
    assert sent[0][:2] == ("GW11 self-improvement failed", "error") and "ValueError: bad data" in sent[0][2][0]
    delivered[0] = True
    assert improve.run(**quiet) == 1 and len(sent) == 2            # tried again, delivered this time
    assert improve.run(**quiet) == 1 and len(sent) == 2            # same error, same gameweek: once
    err[0] = KeyError("x")
    assert improve.run(**quiet) == 1 and len(sent) == 3            # a different error is news
    assert improve._state()["failed"] == {"gw": 11, "error": "KeyError"}


def test_a_manual_run_is_not_caught(state, monkeypatch):
    review = state / "gw11_review.md"
    review.write_text("")

    def boom(*a, **k):
        raise ValueError("bad data")
    monkeypatch.setattr(improve, "_run", boom)
    with pytest.raises(ValueError):
        improve.run(str(review), log=lambda *_: None)


def test_a_review_with_no_limiting_factor_is_done_when_scheduled(state, monkeypatch):
    review = state / "gw10_review.md"
    review.write_text("")
    monkeypatch.setattr(gwreview, "parse", lambda text: {"gw": 10, "final": True, "engine_managed": False})
    monkeypatch.setattr(gwreview, "load_reviews", lambda: [])
    quiet = dict(log=lambda *_: None)
    assert improve.run(str(review), **quiet) == 2                  # by hand: say there is nothing to do
    assert "done_gw" not in improve._state()
    assert improve.run(str(review), if_due=True, **quiet) == 0      # scheduled: done, not retried every run
    assert improve._state()["done_gw"] == 10


# ----------------------------------------------------------------------------- how sure the limiting factor is
def test_part_noise_is_signed_like_the_deficits():
    elite = [{"best_xi": b, "selection": s, "captain_extra": 2, "chip_extra": 0, "hits": 0}
             for b, s in ((40, 5), (50, 3), (60, 1))]
    C = gwreview.part_noise(elite)
    assert C["squad"]["squad"] == pytest.approx(100) and C["selection"]["selection"] == pytest.approx(4)
    assert C["squad"]["selection"] == pytest.approx(20)     # more best eleven = less squad deficit, here less bench
    assert C["captaincy"]["captaincy"] == 0
    assert gwreview.part_noise(elite[:2]) is None


def _weeks(squad, selection, var=4.0):
    cov = {f: {g: (var if f == g else 0.0) for g in gwreview.FACTORS} for f in gwreview.FACTORS}
    return [{"gw": i + 1, "deficit_cov": cov,
             "deficits": {"squad": a, "selection": b, "captaincy": 0, "chips": 0, "hits": 0}}
            for i, (a, b) in enumerate(zip(squad, selection))]


def test_one_week_never_settles_the_factor():
    win = _weeks([10], [0])
    f, _ = gwreview.choose_factor(win)
    c = gwreview.certainty(win, f)
    assert f == "squad" and not c["settled"] and c["needed"] is None
    assert c["se"]["squad"] == pytest.approx(2) and c["lead"] == pytest.approx(10)
    assert c["lead_se"] == pytest.approx(np.sqrt(8))
    assert "from one gameweek" in gwreview.certainty_text(f, c)
    assert "lead" not in gwreview.certainty([{**win[0], "deficit_cov": None}], f)   # no noise estimate at all


def test_a_steady_lead_settles_and_a_noisy_one_does_not():
    steady = _weeks([10, 11, 9, 10, 12, 8], [1, 0, 2, 1, 0, 1])
    c = gwreview.certainty(steady, "squad")
    assert c["settled"] and c["needed"] == pytest.approx(2.571, abs=1e-3) and c["runner_up"] == "selection"
    assert gwreview.certainty_text("squad", c).startswith("**Settled:**")
    noisy = _weeks([25, -5, 20, -10, 15, -5], [0] * 6)
    c = gwreview.certainty(noisy, gwreview.choose_factor(noisy)[0])
    assert not c["settled"] and c["se"]["squad"] > 5            # the spread of your own weeks, not the top 50's
    assert gwreview.certainty(noisy, "operations")["settled"]


# ----------------------------------------------------------------------------- the model's errors, pooled
def test_collect_sums_errors_by_club():
    likely = pd.DataFrame({"team": ["A", "A", "B"], "pos": [1, 2, 2], "xp": [3.0, 4.0, 5.0], "pts": [5, 4, 2],
                           "cs": [1.0, 1.0, 0.5], "a_cs": [4.0, 4.0, 0.0]}, index=[1, 2, 3])
    allp = likely.assign(p_play=[0.9, 0.9, 0.3], played=[1.0, 1.0, 0.0])
    teams = pd.DataFrame({"projected": [1.0, 2.0], "xg": [1.5, 1.0]}, index=["A", "B"])
    e = evidence.collect(likely, allp, likely.nlargest(2, "xp"), teams, ["cs"], [0, 0.5, 1.0001],
                         {1: "GKP", 2: "DEF"})
    assert e["comp:cs"] == {"A": [2, 6.0], "B": [1, -0.5]}
    assert e["pos:GKP"] == {"A": [1, 2.0]} and e["pos:DEF"] == {"A": [1, 0.0], "B": [1, -3.0]}
    assert e["top10"] == {"A": [1, 0.0], "B": [1, -3.0]}
    assert e["team_xg"] == {"A": [1, 0.5], "B": [1, -1.0]}
    assert e["play:0%-50%"]["B"] == [1, pytest.approx(-0.3)] and e["play:50%-100%"]["A"] == [2, pytest.approx(0.2)]


def _review(gw, errors):
    return {"gw": gw, "model": {"errors": errors}}


def test_pool_clusters_by_club_and_gameweek():
    reviews = [_review(1, {"comp:cs": {"A": [2, 1.0], "B": [2, 0.6], "C": [2, 0.8]}}),
               _review(2, {"comp:cs": {"A": [2, 0.4], "B": [2, 0.2], "C": [2, 0.6]}}),
               {"gw": 3, "model": None}, _review(4, {"unknown": {"A": [5, 1.0], "B": [5, 1.0]}})]
    [f] = evidence.pool(reviews)
    assert f.key == "comp:cs" and f.mean == pytest.approx(0.3) and f.n == 12 and f.gws == [1, 2]
    assert f.se == pytest.approx(np.sqrt(6 / 5 * 0.40) / 12)       # CR1 over the six club-weeks
    assert f.z > 5 and f.verdict == "watch"                         # two gameweeks: a shared shock is not ruled out
    assert "too low by 0.30" in f.size_text()
    reviews.append(_review(5, {"comp:cs": {"A": [2, 0.7], "B": [2, 0.5], "C": [2, 0.6]}}))
    assert evidence.pool(reviews)[0].verdict == "act"


def test_pool_uses_the_spread_between_gameweeks_when_larger():
    reviews = [_review(g, {"pos:MID": {"A": [1, m], "B": [1, m]}}) for g, m in ((1, 0.0), (2, 0.3), (3, 0.6))]
    [f] = evidence.pool(reviews)
    assert f.se == pytest.approx(0.3 / np.sqrt(3))
    assert f.verdict is None                                      # 1.7 s.e.: within noise


def test_leads_rank_act_before_watch_and_drop_small_or_noisy():
    def finding(key, mean, se):
        return evidence.Finding(key, evidence.check(key), mean, se, 500, [10, 11, 12])
    fs = [finding("comp:bonus", 0.10, 0.04), finding("pos:GKP", -0.6, 0.1), finding("comp:cs", 0.02, 0.001),
          finding("comp:goals", 0.2, 0.2)]
    ranked = sorted(fs, key=lambda f: (f.verdict != "act", f.verdict != "watch", -abs(f.z)))
    assert [f.key for f in evidence.leads(ranked)] == ["pos:GKP", "comp:bonus"]
    assert [r[4] for r in evidence.table(ranked)] == ["act", "watch"]
    assert "too high by 0.60" in fs[1].size_text()
    assert "2 to act on" not in evidence.summary(ranked) and "1 to act on" in evidence.summary(ranked)


# ----------------------------------------------------------------------------- the brief is aimed by evidence
def test_brief_leads_with_evidence_and_says_when_nothing_is_well_aimed(tuned, tmp_path):
    review = tmp_path / "gw12_review.md"
    review.write_text(gwreview.NOTES_HEADING + "\n")
    outcomes = [improve.Outcome("points_model", [_trial(1.2)])]
    gk = evidence.Finding("pos:GKP", evidence.check("pos:GKP"), 0.6, 0.1, 80, [10, 11, 12])
    unsettled = {"n": 2, "se": {}, "settled": False}

    text = improve.render(12, review, {"final": True}, "selection", {"selection": 9.8}, [{"gw": 11}, {"gw": 12}],
                          outcomes, None, None, True, cert=unsettled, found=[gk])
    assert "Strongest evidence: GKP points per likely starter, projection too low by 0.60" in text
    assert text.index("### Evidence") < text.index("### Ideas for the limiting factor")
    assert "**selection** (not settled)" in text

    text = improve.render(12, review, {"final": True}, "selection", {}, [], outcomes, None, None, True,
                          cert=unsettled, found=[])
    assert "No code change is well aimed yet" in text
    lines = improve.result_lines("selection", outcomes, None, "x.md", cert={"settled": True}, found=[])
    assert any("is settled, so its ideas aim the change" in x for x in lines) and "/fpl-improve" in " ".join(lines)
