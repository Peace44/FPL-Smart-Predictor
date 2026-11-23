"""Self-improvement: read a finished gameweek review, find its limiting factor, and test changes to
the settings behind that factor on past seasons. A change is kept only when it is clearly better.

    fpl improve                         the latest final review
    fpl improve reports/gw12_review.md  a given review
    fpl improve --dry-run               test, report, change nothing
    fpl improve --if-due                for the scheduler: only for a new final review, only while no
                                        deadline is close
    fpl improve --revert 3              undo change #3 (tuning/params.json history)

The review picks the target (which factor costs the most points against the top 50); the test
decides whether to act. One week of points is mostly luck, so the review only chooses where to
look: every candidate is judged on thousands of player-matches from earlier seasons and this one,
gameweek by gameweek, against the settings in effect. At most one knob moves per run, by one step
and within hard bounds (tuning.KNOBS), and only when it beats the current value by three standard
errors without making the guard metric or the current season clearly worse. Over the weeks that
is coordinate descent on whatever the reviews keep finding limiting.

When no setting passes, or none covers the factor, the limit is likely in the code: the run writes a
brief for a code change (`reports/gwNN_improvement.md`) that the `fpl-improve` skill works from. The
brief is aimed by evidence first: the model's errors pooled over every review, with standard errors
(`evidence.py`). The factor's own ideas follow, marked settled or not (`gwreview.certainty`): a few
weeks of team results rarely tell the parts apart, so an unsettled factor does not aim a change. The Discord
report says which of the reasons it was (no setting covers the factor, none passed, a test could
not run) and names the skill. A scheduled run that fails is reported too, once per gameweek and
kind of error, and a gameweek that no run worked on is named in the next run's report.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import evidence, fpl_api, gwreview, tuning
from .autopilot import Settings, _report
from .config import REPORTS

Z_ADOPT = 3.0          # many candidates are tried each run; 3 s.e. keeps false adoptions rare
Z_GUARD = -1.0         # the guard metric and the current season may not get clearly worse
# Smallest improvement worth a change, per gameweek. Paired comparisons can show a difference far too
# small to matter as "significant"; these are about a 0.0015 gain in correlation for the points model
# (MSE is mostly the game's own noise), a twentieth of a point for the top projections, a point a
# gameweek for the replayed optimiser.
MIN_EFFECT = {"mse_rel": 0.008, "top30": 0.03, "top10": 0.05, "logloss": 0.002, "points": 1.0}
MIN_AVAIL_GWS = 3
STATE = REPORTS / "improve-state.json"
HEAVY_BUDGET_MIN = 45
# A run must end before the autopilot starts planning, so the plan and its later checks see the same
# settings and the lock never holds up a run near the deadline. The margin covers HEAVY_BUDGET_MIN.
MARGIN_H = 2.0


@dataclass
class Experiment:
    name: str
    knobs: list[str]
    primary: str
    guard: str | None
    kind: str                # "xp" | "availability" | "planning"
    what: str


PLAYER = [k for k in tuning.KNOBS if k.startswith("player.")]
TEAM = [k for k in tuning.KNOBS if k.startswith("team.")]
EXPERIMENTS = {
    "points_model": Experiment("points_model", PLAYER + TEAM, "mse_rel", "top30", "xp",
                               "accuracy of every likely starter's projection (mean squared error), guarded by the "
                               "actual points of each gameweek's 30 best projections"),
    "top_end": Experiment("top_end", ["player.k_goal", "player.k_ast", "player.xg_blend", "player.xa_blend",
                                      "player.hl_days", "player.k_bps", "player.k_bres"] + TEAM, "top10", "mse_rel", "xp",
                          "actual points of each gameweek's ten best projections (the captaincy candidates), "
                          "guarded by overall accuracy"),
    "availability": Experiment("availability", [k for k in tuning.KNOBS if k.startswith("availability.")],
                               "logloss", None, "availability",
                               "how well P(plays) predicted who played, from the projections saved before each "
                               "live deadline"),
    "planning": Experiment("planning", [k for k in tuning.KNOBS if k.startswith("solver.")], "points", None, "planning",
                           "points scored by the optimiser replayed over last season and this one"),
}
PLAN = {
    "squad": ["points_model", "planning"],
    "selection": ["points_model", "availability"],
    "captaincy": ["top_end"],
    "hits": ["planning"],
    "chips": [],
    "operations": [],
}
STRUCTURAL = {
    "squad": [
        "Penalty and set-piece duty as an input: goal shares come from each player's own xG history, so a new "
        "penalty taker is undervalued for weeks (player_model.rates; FPL's bootstrap has penalties_order).",
        "FPL's own projection (ep_next) or ownership as a weak extra signal where the review shows it ranking "
        "players better (xp.project_live, blended and tested with `fpl evaluate`).",
        "Odds beyond the next one or two gameweeks are ratings only (odds.py); a longer odds window helps plans.",
    ],
    "selection": [
        "Late team news: the autopilot's last look is up to five hours before the deadline and only sees FPL flags. "
        "A check of predicted line-ups / press-conference news before the final run (autopilot.decide, xp.availability).",
        "Rotation risk around European and cup fixtures: the minutes model knows days since the last match but not "
        "the next midweek match (minutes.FEATURES).",
        "Start/bench choice by expected points only: when two players are close, prefer the one with fewer ways to "
        "blank (P(60 minutes), fixture) (solver objective).",
    ],
    "captaincy": [
        "Captaincy ignores ownership: when chasing rank, an armband on the player most of the top managers "
        "captain protects rank, a differential gains it. Weigh effective ownership in the captain choice "
        "(solver, captain objective; ownership is in the projection's players table).",
        "Top-end calibration: the README notes projections run low for the very best players. A walk-forward "
        "multiplier by price band (player_model.fit_calibration) tested with `fpl evaluate`.",
        "Captain by ceiling as well as mean: P(10+ points) needs a points distribution per player, not only the mean.",
    ],
    "chips": [
        "Chip timing rests on fixed reserves (solver.SolverOptions.chip_reserve) that no replay tests: add chips to "
        "backtest.simulate so `fpl improve` can tune them.",
        "Double and blank gameweeks: check the chip search covers them as soon as FPL moves fixtures.",
    ],
    "hits": [
        "Take a hit only when the confidence check agrees (optimise.robustness: share of noisy re-solves making the "
        "same move), not on the expected gain alone.",
    ],
    "operations": [
        "Reproduce the incident in tests/test_autopilot.py (FakeFPL), fix it in execute.py / autopilot.py, and "
        "make the review's operations check catch the same class of problem earlier.",
    ],
}


@dataclass
class Trial:
    knob: str
    old: float
    new: float
    primary: ev.Comparison | None = None
    guard: ev.Comparison | None = None
    note: str = ""

    @property
    def current_z(self) -> float:
        cur = (self.primary.by_season.get(ev.CURRENT_SEASON) or {}) if self.primary else {}
        z = cur.get("z")
        return float(z) if z is not None and not np.isnan(z) else 0.0

    @property
    def passes(self) -> bool:
        if self.primary is None or not np.isfinite(self.primary.se) or self.primary.n < 5:
            return False
        return (self.primary.z >= Z_ADOPT and self.primary.diff >= MIN_EFFECT.get(self.primary.metric, 0.0)
                and (self.guard is None or self.guard.z >= Z_GUARD) and self.current_z >= Z_GUARD)


@dataclass
class Outcome:
    experiment: str
    trials: list[Trial] = field(default_factory=list)
    adopted: Trial | None = None
    skipped: str | None = None
    baseline: pd.DataFrame | None = None
    seconds: float = 0.0


# ----------------------------------------------------------------------------- experiments
def _candidates(knobs: list[str]) -> list[Trial]:
    out = []
    for k in knobs:
        v = tuning.current(k)
        out += [Trial(k, v, n) for n in tuning.KNOBS[k].neighbours(v)]
    return out


def run_xp(exp: Experiment, b: ev.Base, log) -> Outcome:
    o = Outcome(exp.name)
    t0 = time.time()
    log(f"{exp.name}: baseline ...")
    o.baseline = base = ev.points_model(b)
    for i, t in enumerate(_candidates(exp.knobs), 1):
        section, name = t.knob.split(".", 1)
        log(f"{exp.name}: {i} {t.knob} {t.old:g} -> {t.new:g}")
        df = ev.points_model(b, **{section: {name: t.new}})
        t.primary = ev.compare(base, df, exp.primary)
        t.guard = ev.compare(base, df, exp.guard) if exp.guard else None
        o.trials.append(t)
    o.seconds = time.time() - t0
    return o


def run_availability(exp: Experiment, log) -> Outcome:
    o = Outcome(exp.name)
    rows = ev.availability_rows()
    n_gw = rows["gw"].nunique() if len(rows) else 0
    if n_gw < MIN_AVAIL_GWS:
        o.skipped = (f"needs projections saved before {MIN_AVAIL_GWS} live deadlines with the model's raw P(plays); "
                     f"has {n_gw}. They are saved with every plan from now on.")
        return o
    base = ev.availability_metric(rows)
    for t in _candidates(exp.knobs):
        name = t.knob.split(".", 1)[1]
        d = base - ev.availability_metric(rows, {name: t.new})          # positive = better
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        by = {}
        for s_gw, x in d.groupby(rows["gw"]):
            by[f"GW{s_gw}"] = {"diff": float(x.mean()), "n": len(x)}
        t.primary = ev.Comparison("logloss", float(d.mean()), se, len(d), {ev.CURRENT_SEASON: {
            "diff": float(d.mean()), "se": se, "z": float(d.mean() / se) if se > 0 else 0.0, "n": len(d)}} | by)
        o.trials.append(t)
    return o


def run_planning(exp: Experiment, log, budget_min: float) -> Outcome:
    o = Outcome(exp.name)
    t0 = time.time()
    cands = _candidates(exp.knobs)
    overrides = [{}] + [{t.knob.split(".", 1)[1]: t.new} for t in cands]
    log(f"planning: replaying {len(overrides)} settings (budget {budget_min:.0f} min) ...")
    res = ev.planning(overrides, log=log)
    base = res[0]
    for t, df in zip(cands, res[1:]):
        t.primary = ev.compare(base, df, "points")
        o.trials.append(t)
    o.baseline = base
    o.seconds = time.time() - t0
    return o


# ----------------------------------------------------------------------------- the run
def latest_final_review() -> Path | None:
    for f in sorted(REPORTS.glob("gw*_review.md"), reverse=True):
        d = gwreview.parse(f.read_text())
        if d and d.get("final"):
            return f
    return None


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save_state(**kw) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({**_state(), **kw}))


def _mark_done(gw: int) -> None:
    """Record `gw` as worked on. Never backwards: a run on an older review by hand does not make the
    scheduler redo newer ones."""
    st = _state()
    st.pop("failed", None)
    st.update(done_gw=max(gw, st.get("done_gw", 0)), when=str(pd.Timestamp.now(tz="UTC").floor("s")))
    REPORTS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st))


def missed(gw: int) -> list[tuple[int, str]]:
    """Gameweeks after the last one worked on and before `gw` that no run worked on, each with why.
    None before the first run: earlier reviews may be ones written by hand to fill in the history."""
    st = _state()
    done = st.get("done_gw")
    if not done:
        return []
    fail = st.get("failed") or {}
    return [(g, f"its runs ended with {fail['error']}" if fail.get("gw") == g else
             "no scheduled run came while it was due") for g in range(done + 1, gw)]


def failed(path: Path | None, e: Exception, log=print) -> bool:
    """Report a scheduled run that stopped with an error. The same error for the same gameweek is
    reported once, and only marked reported once Discord has it, so a run with no network tries again."""
    gw = None
    try:
        gw = (gwreview.parse(path.read_text()) or {}).get("gw") if path else None
    except Exception:  # noqa: BLE001  (the review itself may be what failed)
        pass
    error = type(e).__name__
    if _state().get("failed") == {"gw": gw, "error": error}:
        log(f"improve: same error as the last run ({error}); already reported")
        return False
    what = f"GW{gw} self-improvement" if gw else "Self-improvement"
    ok = _report(log, f"{what} failed", [f"`fpl improve --if-due` stopped with {error}: {e}"],
                 ["The traceback is in reports/auto.log. The next scheduled run tries again; "
                  "this error is not reported again for this gameweek."], kind="error")
    if ok:
        _save_state(failed={"gw": gw, "error": error})
    return ok


def _closest(outcomes: list[Outcome]) -> str:
    ts = [t for o in outcomes for t in o.trials if t.primary and np.isfinite(t.primary.z)]
    t = max(ts, key=lambda t: t.primary.z, default=None)
    return (f" (closest: {t.knob} {t.old:g} → {t.new:g} at {t.primary.z:+.1f} s.e., "
            f"{t.primary.diff:+.4f} {t.primary.metric})") if t else ""


def unchanged(f: str, outcomes: list[Outcome]) -> str:
    """Why no setting changed. Only when every test for the factor ran is the limit known to be in the code."""
    if not outcomes:
        return f"No setting covers {f}, so only a code change can move it."
    ran = [o for o in outcomes if not o.skipped]
    skipped = "; ".join(f"{o.experiment}: {o.skipped.rstrip('.')}" for o in outcomes if o.skipped)
    if not ran:
        return f"No test could run ({skipped}), so it is not known whether a setting would help."
    if skipped:
        return (f"No setting passed the tests that ran{_closest(outcomes)}. Not run: {skipped}. "
                "The limit may be in the code, or in a setting the missing test covers.")
    return f"No setting passed the test{_closest(outcomes)}: the limit is in the code."


def aim(f: str, cert: dict | None, found: list[evidence.Finding]) -> str:
    """What aims a code change this week: the strongest model error, else a settled factor, else nothing yet."""
    acts = [x for x in found if x.verdict == "act"]
    if acts:
        a = acts[0]
        return f"Strongest evidence: {a.check.label}, {a.size_text()} (z {a.z:+.1f}); it moves {a.check.factor}."
    if cert and cert.get("settled"):
        return f"No model error is strong enough yet; the limiting factor ({f}) is settled, so its ideas aim the change."
    return (f"No code change is well aimed yet: the limiting factor ({f}) is not settled and no model error is "
            "strong enough. The next reviews add evidence.")


def result_lines(f: str, outcomes: list[Outcome], adopted: Trial | None, brief: str,
                 skipped: list[tuple[int, str]] = (), cert: dict | None = None,
                 found: list[evidence.Finding] = ()) -> list[str]:
    """The Discord report of a run: what changed, or why nothing did and what you can do about it."""
    if adopted:
        L = [f"Changed {adopted.knob} {adopted.old:g} → {adopted.new:g} ({adopted.primary.metric} better by "
             f"{adopted.primary.z:.1f} s.e.). Nothing for you to do."]
    else:
        L = [unchanged(f, outcomes)] + ([aim(f, cert, list(found))] if cert is not None else []) + [
             f"A brief for a code change is in {brief}. To act on it, run /fpl-improve in Claude Code "
             "(optional: the autopilot carries on with the current settings either way)."]
    L += [f"GW{g} was never worked on: {why}." for g, why in skipped]
    return L


def due(log=print) -> Path | None:
    """For the scheduler: a final review not yet worked on, and the planning window for the next
    deadline not yet open (more than plan_within_h + MARGIN_H hours away)."""
    f = latest_final_review()
    if f is None:
        return None
    d = gwreview.parse(f.read_text())
    if d["gw"] <= _state().get("done_gw", 0):
        return None
    bs = fpl_api.bootstrap(1.0)
    nxt = next((e for e in bs["events"] if e["is_next"]), None)
    if nxt is not None:
        left = (pd.Timestamp(nxt["deadline_time"]) - pd.Timestamp.now(tz="UTC")).total_seconds() / 3600
        cutoff = Settings.from_env().plan_within_h + MARGIN_H
        if left < cutoff:
            log(f"improve: GW{d['gw']} review waits; the GW{nxt['id']} deadline is {left:.0f} hours away "
                f"(runs only more than {cutoff:.0f} hours before it)")
            return None
    return f


def run(review: str | None = None, dry_run: bool = False, factor: str | None = None, heavy: bool = True,
        if_due: bool = False, budget_min: float = HEAVY_BUDGET_MIN, log=print) -> int:
    if not if_due:
        path = Path(review) if review else latest_final_review()
        if path is None:
            log("no final review found: run `fpl review` once the gameweek's data is checked")
            return 2
        return _run(path, dry_run, factor, heavy, budget_min, log)
    path = Path(review) if review else None
    try:
        path = path or due(log)
        return 0 if path is None else _run(path, dry_run, factor, heavy, budget_min, log, scheduled=True)
    except Exception as e:  # noqa: BLE001  (unattended: say so on Discord; auto.sh only writes the log)
        log("improve: the run failed\n" + traceback.format_exc().rstrip())
        failed(path, e, log)
        return 1


def _run(path: Path, dry_run: bool, factor: str | None, heavy: bool, budget_min: float, log,
         scheduled: bool = False) -> int:
    text = path.read_text()
    data = gwreview.parse(text)
    if data is None:
        log(f"{path} has no review data (was it written by `fpl review`?)")
        return 2
    gw = data["gw"]
    reviews = [r for r in gwreview.load_reviews() if r["gw"] != gw] + [data]
    win = gwreview.window(sorted(reviews, key=lambda r: r["gw"]), gw)
    f, avg = gwreview.choose_factor(win, (data.get("operations") or {}).get("points_lost", 0.0))
    f = factor or f or data.get("limiting_factor")
    if f is None:
        why = "the review has no limiting factor (no recent week the engine managed could be compared with the top managers)"
        if scheduled and not dry_run:            # nothing to work on this week: not a failure to repeat every run
            _mark_done(gw)
            log(f"improve: GW{gw}: {why}; nothing to work on")
            return 0
        log(why)
        return 2
    cert = gwreview.certainty(win, f)
    found = evidence.pool(sorted(reviews, key=lambda r: r["gw"]))
    log(f"GW{gw}: limiting factor {f}" + (f" (average {avg[f]:+.1f} behind the top 50 over {len(win)} GW)" if f in avg else "")
        + ("" if cert["settled"] else ", not settled"))

    order = list(PLAN.get(f, []))
    if f == "selection" and (data.get("projection") or {}).get("minutes_loss", 0) > 2:
        order = ["availability", "points_model"]
    outcomes: list[Outcome] = []
    adopted: Trial | None = None
    b = None
    t_start = time.time()
    for name in order:
        exp = EXPERIMENTS[name]
        if exp.kind == "planning" and not heavy:
            outcomes.append(Outcome(name, skipped="heavy replay skipped (--light)"))
            continue
        if exp.kind == "planning" and (time.time() - t_start) / 60 > budget_min:
            outcomes.append(Outcome(name, skipped="time budget used up"))
            continue
        if exp.kind == "xp":
            b = b or ev.base(refresh=True, log=log)
            o = run_xp(exp, b, log)
        elif exp.kind == "availability":
            o = run_availability(exp, log)
        else:
            o = run_planning(exp, log, budget_min)
        passing = sorted([t for t in o.trials if t.passes], key=lambda t: -t.primary.z)
        if passing:
            o.adopted = adopted = passing[0]
        outcomes.append(o)
        if adopted:
            break

    change = None
    if adopted and not dry_run:
        change = tuning.set_value(adopted.knob, adopted.new,
                                  f"GW{gw} review, limiting factor {f}: {adopted.primary.metric} better by "
                                  f"{adopted.primary.z:.1f} s.e.")
    report = render(gw, path, data, f, avg, win, outcomes, adopted, change, dry_run, cert=cert, found=found)
    out = REPORTS / f"gw{gw:02d}_improvement.md"
    if not dry_run:
        skipped = missed(gw)
        for g, why in skipped:
            log(f"improve: GW{g} was never worked on: {why}")
        out.write_text(report)
        tuning.append_ledger(_ledger(gw, f, avg, outcomes, adopted, change, skipped, cert=cert, found=found))
        _mark_done(gw)
        _report(log, f"GW{gw} self-improvement: {f}",
                result_lines(f, outcomes, adopted, f"reports/{out.name}", skipped, cert=cert, found=found),
                kind="ok" if adopted else "info")
        log(f"written {out}")
    else:
        print(report)
    return 0


# ----------------------------------------------------------------------------- reporting
def _trial_rows(o: Outcome) -> list[list]:
    rows = []
    for t in sorted(o.trials, key=lambda t: -(t.primary.z if t.primary and np.isfinite(t.primary.z) else -99)):
        p, g = t.primary, t.guard
        rows.append([t.knob, f"{t.old:g} → {t.new:g}",
                     f"{p.diff:+.4f}" if p else "-", f"{p.z:+.1f}" if p and np.isfinite(p.z) else "-",
                     f"{t.current_z:+.1f}", f"{g.z:+.1f}" if g and np.isfinite(g.z) else "-",
                     "**kept**" if o.adopted is t else ("passes" if t.passes else "")])
    return rows


def render(gw, path, data, f, avg, win, outcomes, adopted, change, dry_run, cert: dict | None = None,
           found: list[evidence.Finding] = ()) -> str:
    cert = cert or {}
    L = [f"# Gameweek {gw}: self-improvement run" + (" (dry run)" if dry_run else ""), "",
         f"- Review: `{path.name}` ({'final' if data.get('final') else 'provisional'})",
         f"- Limiting factor: **{f}**{'' if cert.get('settled') else ' (not settled)'} ({gwreview.FACTOR_TEXT.get(f, f)})"
         + (f", {avg[f]:+.1f} points a week behind the top 50 over "
            f"{', '.join('GW' + str(r['gw']) for r in win)}" if f in avg else ""),
         f"- Rule: a setting changes only if it beats the current value by {Z_ADOPT:g} standard errors and by at "
         f"least the smallest gain worth having on the experiment's main measure, without the guard measure or the "
         f"current season getting worse by more than {-Z_GUARD:g} standard error. At most one setting moves per "
         f"run, by one step."]
    if avg:
        L += ["", "Average gap to the top 50 by part over the window: "
              + ", ".join(f"{k} {v:+.1f}" for k, v in sorted(avg.items(), key=lambda kv: -kv[1])) + "."]
    for o in outcomes:
        exp = EXPERIMENTS[o.experiment]
        L += ["", f"## {o.experiment}", "", f"Measures {exp.what}."]
        if o.skipped:
            L += ["", f"Skipped: {o.skipped}"]
            continue
        n = o.trials[0].primary.n if o.trials and o.trials[0].primary else 0
        unit = "gameweeks" if exp.kind != "availability" else "player-matches"
        L += ["", f"{len(o.trials)} candidates over {n} {unit}" + (f" ({o.seconds / 60:.1f} minutes)" if o.seconds else "")
              + f". Smallest gain worth a change: {MIN_EFFECT.get(exp.primary, 0):g} in {exp.primary} per "
              + ("gameweek." if exp.kind != "availability" else "player-match."), ""]
        if o.baseline is not None and exp.kind == "xp":
            s = ev.summary(o.baseline)
            L += ["Current settings, season averages per gameweek:", "",
                  gwreview.md_table(["Season", "MSE (likely starters)", "Bias", "Correlation", "Top 30 pts", "Top 10 pts"],
                                  [[i, f"{r.mse_rel:.3f}", f"{r.bias_rel:+.3f}", f"{r.corr_rel:.3f}", f"{r.top30:.2f}",
                                    f"{r.top10:.2f}"] for i, r in s.iterrows()]), ""]
        L.append(gwreview.md_table(["Knob", "Tried", f"Δ {exp.primary} (better > 0)", "z", "z this season",
                                  f"z {exp.guard}" if exp.guard else "guard", ""], _trial_rows(o)))
    L += ["", "## Result", ""]
    if adopted:
        L.append(f"Changed **{adopted.knob}** from {adopted.old:g} to {adopted.new:g}"
                 + (f" (change #{change['id']} in `tuning/params.json`; `fpl improve --revert {change['id']}` undoes it)."
                    if change else " (dry run: nothing written)."))
        if change:
            L.append("The next plan uses it. The next reviews show whether the factor it targets improves.")
    else:
        L.append(unchanged(f, outcomes))
        found = list(found)
        L += ["", "## Needs a code change", "",
              "For the `fpl-improve` skill (run /fpl-improve in Claude Code) or you to implement on a branch, test with "
              "`fpl evaluate` and the test suite, and record in `tuning/ledger.md`.", "",
              f"**{aim(f, cert, found)}**", "",
              "How to choose: start from the first row marked *act* below. If there is none, use the factor's ideas "
              "only when the factor is settled. If neither, no change is well aimed this week; rows marked *watch* "
              "say where to look once more reviews are in.", "",
              "### Evidence: where the projections were wrong", "", evidence.summary(found)]
        if evidence.leads(found):
            L += ["", gwreview.md_table(evidence.HEADERS, evidence.table(found))]
        L += ["", f"### Ideas for the limiting factor, {f}", ""]
        if cert:
            L += [gwreview.certainty_text(f, cert), ""]
        L += [f"{i}. {x}" for i, x in enumerate(STRUCTURAL.get(f, []), 1)]
        notes = gwreview.existing_notes(path.read_text()).strip()
        if notes:
            L += ["", "Your notes in the review:", "", "> " + notes.replace("\n", "\n> ")]
        tried = _past_structural()
        if tried:
            L += ["", "Already tried in earlier runs (see `tuning/ledger.md`): " + "; ".join(tried) + "."]
    return "\n".join(L) + "\n"


def _ledger(gw, f, avg, outcomes, adopted, change, skipped=(), cert: dict | None = None,
            found: list[evidence.Finding] = ()) -> list[str]:
    when = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    L = [f"## GW{gw} ({when}): limiting factor {f}" + (f", {avg[f]:+.1f} a week behind the top 50" if f in avg else "")
         + ("" if cert is None else ", settled" if cert.get("settled") else ", not settled")]
    if cert is not None and not adopted:
        L.append(f"- Aim: {aim(f, cert, list(found))}")
    for o in outcomes:
        if o.skipped:
            L.append(f"- {o.experiment}: skipped, {o.skipped}")
            continue
        best = max((t for t in o.trials if t.primary and np.isfinite(t.primary.z)), key=lambda t: t.primary.z, default=None)
        L.append(f"- {o.experiment}: {len(o.trials)} candidates; " + (
            f"kept {o.adopted.knob} {o.adopted.old:g} → {o.adopted.new:g} (z {o.adopted.primary.z:+.1f})" if o.adopted else
            f"none passed (best {best.knob} {best.old:g} → {best.new:g}, z {best.primary.z:+.1f})" if best else "none ran"))
    L.append(f"- Result: change #{change['id']} {change['key']} {change['old']:g} → {change['new']:g}" if change else
             f"- Result: no setting changed; code-change brief in reports/gw{gw:02d}_improvement.md")
    L += [f"- Never worked on: GW{g}, {why}" for g, why in skipped]
    return L


def _past_structural() -> list[str]:
    """Code changes recorded in the ledger by the skill (lines starting "- Code change:")."""
    try:
        text = tuning.LEDGER.read_text()
    except OSError:
        return []
    return [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("- Code change:")]
