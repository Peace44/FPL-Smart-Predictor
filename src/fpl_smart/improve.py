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

When no setting passes, the limit is in the code, not the settings: the run writes a brief for a
code change (`reports/gwNN_improvement.md`) that the `fpl-improve` skill works from.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import fpl_api, gwreview, notify, tuning
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


def due(log=print) -> Path | None:
    """For the scheduler: a final review not yet worked on, and no deadline in the next ~2 days."""
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
        if left < 50:
            log(f"improve: GW{d['gw']} review waits; the GW{nxt['id']} deadline is {left:.0f} hours away")
            return None
    return f


def run(review: str | None = None, dry_run: bool = False, factor: str | None = None, heavy: bool = True,
        if_due: bool = False, budget_min: float = HEAVY_BUDGET_MIN, log=print) -> int:
    path = Path(review) if review else (due(log) if if_due else latest_final_review())
    if path is None:
        if not if_due:
            log("no final review found: run `fpl review` once the gameweek's data is checked")
        return 0 if if_due else 2
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
        log("the review has no limiting factor (no comparison with the top managers was possible)")
        return 2
    log(f"GW{gw}: limiting factor {f}" + (f" (average {avg[f]:+.1f} behind the top 50 over {len(win)} GW)" if f in avg else ""))

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
    report = render(gw, path, data, f, avg, win, outcomes, adopted, change, dry_run)
    out = REPORTS / f"gw{gw:02d}_improvement.md"
    if not dry_run:
        out.write_text(report)
        tuning.append_ledger(_ledger(gw, f, avg, outcomes, adopted, change))
        STATE.write_text(json.dumps({**_state(), "done_gw": gw, "when": str(pd.Timestamp.now(tz="UTC").floor("s"))}))
        notify.publish({out.name: report})
        line = (f"Changed {adopted.knob} {adopted.old:g} → {adopted.new:g} ({adopted.primary.metric} better by "
                f"{adopted.primary.z:.1f} s.e.)." if adopted else
                "No setting passed the test: the limit is in the code. A brief for a code change is in "
                f"reports/{out.name}.")
        notify.discord(f"GW{gw} self-improvement: {f}", [line], kind="ok" if adopted else "info")
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


def render(gw, path, data, f, avg, win, outcomes, adopted, change, dry_run) -> str:
    L = [f"# Gameweek {gw}: self-improvement run" + (" (dry run)" if dry_run else ""), "",
         f"- Review: `{path.name}` ({'final' if data.get('final') else 'provisional'})",
         f"- Limiting factor: **{f}** ({gwreview.FACTOR_TEXT.get(f, f)})"
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
                  gwreview._table(["Season", "MSE (likely starters)", "Bias", "Correlation", "Top 30 pts", "Top 10 pts"],
                                  [[i, f"{r.mse_rel:.3f}", f"{r.bias_rel:+.3f}", f"{r.corr_rel:.3f}", f"{r.top30:.2f}",
                                    f"{r.top10:.2f}"] for i, r in s.iterrows()]), ""]
        L.append(gwreview._table(["Knob", "Tried", f"Δ {exp.primary} (better > 0)", "z", "z this season",
                                  f"z {exp.guard}" if exp.guard else "guard", ""], _trial_rows(o)))
    L += ["", "## Result", ""]
    if adopted:
        L.append(f"Changed **{adopted.knob}** from {adopted.old:g} to {adopted.new:g}"
                 + (f" (change #{change['id']} in `tuning/params.json`; `fpl improve --revert {change['id']}` undoes it)."
                    if change else " (dry run: nothing written)."))
        if change:
            L.append("The next plan uses it. The next reviews show whether the factor it targets improves.")
    else:
        best = [t for o in outcomes for t in o.trials if t.primary and np.isfinite(t.primary.z)]
        top = max(best, key=lambda t: t.primary.z, default=None)
        if not outcomes:
            L.append("No setting covers this factor.")
        else:
            L.append("No setting passed the test" + (
                f" (closest: {top.knob} {top.old:g} → {top.new:g} at {top.primary.z:+.1f} s.e., "
                f"{top.primary.diff:+.4f} {top.primary.metric})" if top else "") + ".")
        L += ["", "## Needs a code change", "",
              f"The settings around the current values cannot move **{f}**. Ideas, most promising first, for the "
              "`fpl-improve` skill (or you) to implement on a branch, test with `fpl evaluate` and the test suite, and "
              "record in `tuning/ledger.md`:", ""]
        L += [f"{i}. {x}" for i, x in enumerate(STRUCTURAL.get(f, []), 1)]
        notes = gwreview.existing_notes(path.read_text()).strip()
        if notes:
            L += ["", "Your notes in the review:", "", "> " + notes.replace("\n", "\n> ")]
        tried = _past_structural()
        if tried:
            L += ["", "Already tried in earlier runs (see `tuning/ledger.md`): " + "; ".join(tried) + "."]
    return "\n".join(L) + "\n"


def _ledger(gw, f, avg, outcomes, adopted, change) -> list[str]:
    when = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    L = [f"## GW{gw} ({when}): limiting factor {f}" + (f", {avg[f]:+.1f} a week behind the top 50" if f in avg else "")]
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
    return L


def _past_structural() -> list[str]:
    """Code changes recorded in the ledger by the skill (lines starting "- Code change:")."""
    try:
        text = tuning.LEDGER.read_text()
    except OSError:
        return []
    return [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("- Code change:")]
