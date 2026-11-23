"""Unattended mode: decide whether a run is due, plan, act on the account, and report.

Meant to be triggered often (every couple of hours); almost every trigger costs one small
download and exits. Timing, for a deadline D:

* D-48h: a plan is written, nothing is touched.
* D-30h (FPL_APPLY_WITHIN_HOURS): the first run that finds itself inside this window carries
  the plan out. It does not wait for the last minute because nobody knows whether the computer
  will be on then. It does wait while the previous gameweek still has matches to play, unless the
  deadline is under six hours away.
* After that the squad is settled. Later runs are short (no chip search): they re-check the
  line-up and captain, and happen when a squad player's injury flag changes, every 12 hours, and
  once more inside the last five hours.

All of that assumes it is triggered every couple of hours. FPL_CHECK_EVERY_HOURS tells it how
far apart the triggers really are, so it never waits for a check that will not come:
* under half the acting window (11 hours for a 24-hour window): at least two triggers fall
  inside the window. The first acts; the last one that still leaves an hour before the deadline
  is the final check.
* once a day: exactly one trigger falls inside the window. It acts then and never looks again.

Limits on what it may do on its own are in `Settings`.
"""
from __future__ import annotations

import datetime
import json
import os
import time
from dataclasses import dataclass

import pandas as pd

from . import account as acc
from . import execute, fpl_api, notify
from .config import CHIP_NAMES, REPORTS, team_id


def _env_num(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default


def _env_list(key: str) -> list[str]:
    return [x.strip() for x in os.environ.get(key, "").split(",") if x.strip()]


@dataclass
class Settings:
    enabled: bool = False          # FPL_AUTOPILOT=on : may change the account
    plan_within_h: float = 48.0
    apply_within_h: float = 30.0   # FPL_APPLY_WITHIN_HOURS
    max_hits: int = 1              # FPL_AUTO_MAX_HITS : most -4s it may take in one gameweek, all runs together
    final_h: float = 5.0           # one more look inside the last hours
    refresh_h: float = 12.0
    min_gap_h: float = 3.0         # never two runs closer than this unless a flag changed
    in_play_override_h: float = 6.0
    check_every_h: float = 2.0     # FPL_CHECK_EVERY_HOURS : how often the scheduler triggers a run

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(enabled=os.environ.get("FPL_AUTOPILOT", "").strip().lower() in ("on", "1", "true", "yes"),
                   apply_within_h=_env_num("FPL_APPLY_WITHIN_HOURS", 30.0),
                   max_hits=int(_env_num("FPL_AUTO_MAX_HITS", 1)),
                   check_every_h=_env_num("FPL_CHECK_EVERY_HOURS", 2.0))


# ----------------------------------------------------------------------------- run record
def _record_path(gw: int):
    return REPORTS / f"gw{gw:02d}" / "autopilot.json"


def last_run(gw: int) -> dict | None:
    try:
        return json.loads(_record_path(gw).read_text())
    except (OSError, ValueError):
        return None


def flags(bs: dict, squad: list[int]) -> dict[str, str]:
    """Availability of each squad player, as FPL flags it. A change here is news worth re-planning for."""
    els = {e["id"]: e for e in bs["elements"]}
    return {str(p): f"{els[p]['status']}|{els[p].get('chance_of_playing_next_round')}" for p in squad if p in els}


def decide(left_h: float, last: dict | None, flags_now: dict, s: Settings, can_apply: bool,
           gw_in_play: bool, now: pd.Timestamp) -> tuple[str, str]:
    """("skip" | "plan" | "apply", why)."""
    if left_h <= 0:
        return "skip", "the deadline has passed"
    if left_h > s.plan_within_h:
        return "skip", f"deadline is {left_h / 24:.1f} days away"
    too_late = left_h * 60 < execute.MIN_MINUTES
    # Waiting for the previous gameweek to finish is only safe while another check will come
    # before the deadline; with a once-a-day schedule this one may be the last.
    waiting = gw_in_play and left_h > max(s.in_play_override_h, s.check_every_h + 2.0)
    acting = can_apply and left_h <= s.apply_within_h and not too_late and not waiting
    go = "apply" if acting else "plan"
    if last is None:
        return go, "first run for this gameweek"
    if acting and not last.get("applied"):
        return go, "the plan has not been carried out yet"
    age = (now - pd.Timestamp(last["time"])).total_seconds() / 3600.0
    if flags_now and last.get("flags") and flags_now != last["flags"]:
        return go, "a squad player's availability flag changed"
    if age < s.min_gap_h:
        return "skip", f"last run was {age:.1f} hours ago"
    # The last look before the deadline: inside the final hours, or, when checks are further
    # apart than that, as soon as no later check can be counted on. A check that would come with
    # under an hour left is not counted on: it may land inside the minutes where nothing is sent.
    last_stretch = max(s.final_h, s.check_every_h + 1.0)
    if left_h <= last_stretch and age > last_stretch - left_h:      # no run yet inside that stretch
        return go, "final check before the deadline"
    if age >= s.refresh_h:
        return go, f"last run was {age:.0f} hours ago"
    return "skip", f"nothing new since the last run {age:.1f} hours ago"


# ----------------------------------------------------------------------------- the run
def _local(ts: pd.Timestamp) -> str:
    return ts.tz_convert(datetime.datetime.now().astimezone().tzinfo).strftime("%a %d %b, %H:%M")


def _report(log, title: str, lines=(), problems=(), kind: str | None = None, **kw) -> bool:
    """Send a report to Discord and note in the log whether it got there."""
    ok = notify.discord(title, lines, problems, kind, **kw)
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        log(f"Discord report \"{title}\": " + ("sent" if ok else "NOT sent (Discord unreachable or webhook refused)"))
    return ok


def _state() -> dict:
    try:
        return json.loads((REPORTS / "autopilot-state.json").read_text())
    except (OSError, ValueError):
        return {}


def _save_state(**kw) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "autopilot-state.json").write_text(json.dumps({**_state(), **kw}))


def idle_check(account, gw: int, deadline: pd.Timestamp, left_h: float, now: pd.Timestamp, s: Settings,
               every_h: float = 6.0, log=print) -> str | None:
    """A run with nothing to do yet. At most every few hours it uses the sign-in (so it does not
    lapse from disuse, and so a dead one is found days before a deadline, not hours) and reports
    that the schedule is alive. Returns the sign-in problem, if there is one."""
    if time.time() - _state().get("checked", 0) < every_h * 3600:
        return None
    problem, signin, reached = None, "No FPL sign-in is set up: it can plan but not act.", True
    if account is not None:
        try:
            account.access_token(force=True)
            signin = "FPL sign-in: working."
        except acc.AuthError as e:
            problem, signin = str(e), None
        except acc.ApiError:           # network trouble says nothing about the sign-in: check again next run
            signin, reached = "FPL sign-in: could not be checked just now.", False
    if reached:
        _save_state(checked=time.time(), problem=problem)
    lines = [f"GW{gw} deadline: {_local(deadline)} ({left_h / 24:.1f} days away). Nothing to do yet.",
             f"The plan is written inside the last {s.plan_within_h:.0f} hours and "
             + (f"carried out inside the last {s.apply_within_h:.0f}." if s.enabled and account is not None
                else "nothing is changed on your team (autopilot off or no sign-in).")]
    if signin:
        lines.append(signin)
    trouble = [f"Could not sign in to FPL, so nothing can be changed on your team: {problem}"] if problem else []
    _report(log, "Check: nothing to do yet" if not problem else "The FPL sign-in has expired", lines, trouble,
            "error" if problem else "info")
    return problem


def review_step(bs: dict, log=print) -> bool:
    """Once per checked gameweek: write reports/gwNN_review.md (rewriting a provisional one) and send
    the one Discord message for that gameweek: its result and what the review found. Marked done once
    written, so a failure to write is retried on the next run."""
    from . import gwreview
    done = [e for e in bs["events"] if e.get("finished") and e.get("data_checked")]
    if not done:
        return False
    gw = max(e["id"] for e in done)
    if _state().get("reviewed_gw", 0) >= gw:
        return False
    path, rv, _ = gwreview.write(gw, log=log)
    _save_state(reviewed_gw=gw)
    log(f"GW{gw} review written to {path}")
    S, ran = rv.data["summary"], last_run(gw)
    kind = "ok" if S.get("average") is not None and S["net"] >= S["average"] else "info"
    _report(log, f"GW{gw} review: {S['net']} points",
            gwreview.summary_lines(rv, autopilot_set=bool(ran and ran.get("settled"))), kind=kind)
    return True


def overspent(spent: int, gw: int, s: Settings) -> str | None:
    """More points spent on this gameweek's transfers than the autopilot may spend on its own.
    It is read from the account on every run, so it is reported until it is put right and not only
    by the run that caused it. In GW11 the run after the one that lost the Wildcard found the squad
    already in place, set the line-up and reported "team updated" over a -40."""
    allowed = 4 * max(0, s.max_hits)
    if spent <= allowed:
        return None
    return (f"FPL shows {spent} points spent on GW{gw} transfers, more than the {allowed} the autopilot may spend "
            "on its own. If a Wildcard or Free Hit was meant to cover them, play it on the FPL site before the "
            "deadline and the points are given back. If you took these hits yourself, nothing needs doing.")


def status_text(gw: int, deadline: pd.Timestamp, now: pd.Timestamp, mode: str, lines: list[str],
                problems: list[str]) -> str:
    L = [f"# FPL autopilot: gameweek {gw}", "",
         f"- **Last run:** {_local(now)} (your computer's time)",
         f"- **Deadline:** {_local(deadline)}",
         f"- **Mode:** {mode}", "", "## What it did", ""]
    L += [f"- {x}" for x in lines] or ["- Nothing."]
    if problems:
        L += ["", "## Needs your attention", ""] + [f"- {x}" for x in problems]
    return "\n".join(L) + "\n"


def run(force: bool = False, dry_run: bool = False, quick: bool = False, apply_now: bool = False,
        log=print) -> int:
    """One trigger. `force`: run whatever the timing says. `apply_now`: act even outside the window
    (the `fpl apply` command). `dry_run`: show the requests instead of sending them."""
    try:
        return _run(force, dry_run, quick, apply_now, log)
    except Exception as e:      # a run that dies quietly is how a deadline gets missed: say so
        import traceback
        traceback.print_exc()
        msg = f"The run failed before it could finish ({type(e).__name__}: {e}). It will try again on the next check."
        if not dry_run:
            REPORTS.mkdir(parents=True, exist_ok=True)
            text = "# FPL autopilot\n\n## Needs your attention\n\n- " + msg + "\n"
            (REPORTS / "status.md").write_text(text)
            _report(log, "A run failed", [], [msg], "error")
        return 1


def _run(force: bool, dry_run: bool, quick: bool, apply_now: bool, log) -> int:
    from . import planner, report
    s = Settings.from_env()
    now = pd.Timestamp.now(tz="UTC")
    bs = fpl_api.bootstrap(0.0)
    ev = next((e for e in bs["events"] if e["is_next"]), None)
    if ev is None:
        log("season is over; nothing to do")
        return 0
    gw, deadline = ev["id"], pd.Timestamp(ev["deadline_time"])
    left = (deadline - now).total_seconds() / 3600.0
    cur = next((e for e in bs["events"] if e["is_current"]), None)
    in_play = bool(cur) and not cur.get("finished")
    last = last_run(gw)
    account = acc.Account.from_env()
    if not dry_run:
        try:
            review_step(bs, log)
        except Exception as e:  # noqa: BLE001  (a report must never stop the run that manages the team)
            log(f"could not write the gameweek review: {e!r}")
    if not force and not apply_now and left > s.plan_within_h:
        log(f"GW{gw} deadline is {left / 24:.1f} days away; nothing to do")
        problem = idle_check(account, gw, deadline, left, now, s, log=log)
        if problem:
            log(f"sign-in problem: {problem}")
        return 0

    # ---- the account, if a sign-in is set up
    private, problems = None, []
    if account is not None:
        try:
            private = account.my_team()
        except acc.AuthError as e:
            problems.append(f"Could not sign in to FPL, so nothing can be changed on your team: {e}")
        except (acc.ApiError, KeyError, ValueError) as e:
            problems.append(f"Could not read your team from FPL just now ({e}); will try again on the next run.")
    may_write = (s.enabled or apply_now) and private is not None
    squad = [p["element"] for p in private["picks"]] if private else (last or {}).get("squad", [])
    flags_now = flags(bs, squad)

    action, why = decide(left, last, flags_now, s, may_write, in_play, now)
    if apply_now:
        action, why = ("apply" if may_write else "plan"), "asked for"
    elif force and action == "skip":
        action, why = ("apply" if may_write and left <= s.apply_within_h else "plan"), "forced"
    if dry_run and may_write:
        action = "apply"
    log(f"GW{gw} deadline in {left:.1f} hours: {action} ({why})")
    if action == "skip":
        return 0

    # ---- plan. Once the squad is settled for the week, later runs keep the chip weeks already
    # chosen and skip the search: they are there to fix the line-up, not to rethink the season.
    settled = bool(last and last.get("settled") and last.get("gw") == gw)
    hits_taken = acc.points_spent(private or {}) // 4
    quick = quick or settled or left < 0.75        # a full search would not finish before the deadline
    # No confidence check (sims): its only output is the report's "How sure is this" section, which
    # nobody reads before an unattended run acts, and it took 86 s of the GW11 run. `fpl plan` keeps it.
    kw = dict(quick=quick, sims=0,
              max_hits=max(0, s.max_hits - hits_taken), private=private,
              banned=_env_list("FPL_BAN"), locked=_env_list("FPL_LOCK"), log=log)
    if settled:
        kw.update(wildcard=(last.get("chips") or {}).get("wildcard"), freehit=(last.get("chips") or {}).get("freehit"))
    try:
        rec = planner.make(**kw)
    except ValueError:                 # the earlier chip weeks no longer fit (a chip was played by hand, say)
        if not settled:
            raise
        kw.update(wildcard=None, freehit=None)
        rec = planner.make(**kw)
    text = report.render(rec)
    planner.save(rec, text)
    w = rec.plan.weeks[0]
    name = rec.projection.players["name"].to_dict()

    # ---- act
    lines: list[str] = []
    outcome = None
    if action == "apply":
        outcome = execute.apply(rec, account, bs, dry_run=dry_run, max_new_hits=max(0, s.max_hits - hits_taken))
        execute.save(outcome)
        lines += [("Would have: " if dry_run else "") + a for a in outcome.actions]
        problems += outcome.warnings
        if outcome.error:
            problems.append(f"Stopped: {outcome.error}")
        elif outcome.verified:
            lines.append("Checked afterwards: FPL shows the planned squad, line-up and captain.")
    else:
        moves = len([p for p in w.squad if p not in rec.state.squad])
        lines.append(f"Wrote the plan only: {CHIP_NAMES.get(w.chip, 'no chip')}, {moves} transfer(s), "
                     f"captain {name.get(w.captain)}. Nothing was changed on your team.")
        if account is None:
            problems.append("No FPL sign-in is set up, so this cannot act for you (README, \"Letting it act for you\").")
        elif private is not None and not s.enabled:
            problems.append("Acting on its own is switched off (FPL_AUTOPILOT=on in .env turns it on).")
        elif private is not None and left > s.apply_within_h:
            lines.append(f"It will act once the deadline is within {s.apply_within_h:.0f} hours.")
        elif private is not None and in_play:
            lines.append(f"It will act once GW{cur['id']} has finished.")
    # A run that stopped says why itself; any other run checks the account's points for transfers.
    spent = outcome.points_cost if outcome is not None and outcome.points_cost is not None \
        else acc.points_spent(private or {})
    over = overspent(spent, gw, s) if may_write and not (outcome and outcome.error) else None
    if over:
        problems.insert(0, over)

    done_now = bool(outcome and outcome.ok and not dry_run)
    applied = done_now or bool(last and last.get("applied") and action != "apply")
    mode = ("dry run" if dry_run else "acting for you" if may_write else "plans only")
    status = status_text(gw, deadline, now, mode, lines, problems)
    if not dry_run:
        _record_path(gw).parent.mkdir(parents=True, exist_ok=True)
        _record_path(gw).write_text(json.dumps({
            "gw": gw, "time": str(now), "action": action, "applied": applied,
            "settled": done_now or bool(last and last.get("settled")), "flags": flags(bs, w.squad),
            "squad": w.squad, "chips": rec.plan.chips, "error": outcome.error if outcome else None}, indent=1))
        (REPORTS / "status.md").write_text(status)
    print(status)

    # ---- tell the owner
    if not dry_run:
        changed = bool(outcome and outcome.changed)
        what = ("needs your attention" if problems else "team updated" if changed
                else "team checked, nothing to change" if outcome else "plan ready")
        pos = rec.projection.players["pos"].to_dict()
        mark = lambda p: str(name.get(p)) + (" (C)" if p == w.captain else " (V)" if p == w.vice else "")  # noqa: E731
        lineup = " | ".join(f"{label}: " + ", ".join(mark(p) for p in w.lineup if pos.get(p) == k)
                            for k, label in ((1, "GK"), (2, "DEF"), (3, "MID"), (4, "FWD")))
        moves = len([p for p in w.squad if p not in rec.state.squad])
        _report(
            log, f"GW{gw}: {what}", lines, problems,
            "error" if (outcome and outcome.error) or over else "warn" if problems else "ok" if outcome else "info",
            fields={"Plan": f"{CHIP_NAMES.get(w.chip, 'No chip')}; {moves} transfer(s) against the squad it started "
                            f"from; projected {w.xp:.1f} points this gameweek.",
                    "Line-up": lineup + " | Bench: " + ", ".join(str(name.get(p)) for p in w.bench),
                    "Deadline": _local(deadline) + f" ({left:.0f} hours after this run started)"},
            footer=f"Mode: {mode}")
    return 1 if (outcome and outcome.error) else 0


def login_summary(pasted: str | None = None, log=print) -> int:
    """Check the sign-in and show what FPL says about the team. Reads only.
    `pasted`: a new refresh token, or the whole `oidc.user:...` value copied from the browser."""
    if pasted is not None:
        token = acc.parse_refresh_token(pasted)
        if len(token) < 40 or any(c.isspace() for c in token):
            print("That does not look like a sign-in token. Copy the whole value of the browser's "
                  "`oidc.user:...` entry (it starts with { and contains \"refresh_token\").")
            return 2
        account = acc.Account(team_id(), token)
    else:
        account = acc.Account.from_env()
    if account is None:
        print("No FPL sign-in is set up. Put FPL_REFRESH_TOKEN=... in .env (README, \"Letting it act for you\").")
        return 2
    try:
        mt = account.my_team()
    except (acc.AuthError, acc.ApiError) as e:
        print(f"Sign-in failed: {e}")
        return 1
    bs = fpl_api.bootstrap(1.0)
    els = {e["id"]: e for e in bs["elements"]}
    tr = mt.get("transfers") or {}
    chips = ", ".join(f"{c.get('name')} ({c.get('status_for_entry')})" for c in mt.get("chips") or []) or "not listed"
    cap = next((els[p["element"]]["web_name"] for p in mt["picks"] if p.get("is_captain")), "?")
    print(f"Signed in. Team {team_id()}: {len(mt['picks'])} players, bank £{(tr.get('bank') or 0) / 10:.1f}m, "
          f"free transfers left {acc.free_transfers_left(mt)}, points spent on transfers so far "
          f"{acc.points_spent(mt)}, captain {cap}.")
    print(f"Chips: {chips}")
    print(f"Autopilot: {'on' if Settings.from_env().enabled else 'off'}")
    return 0
