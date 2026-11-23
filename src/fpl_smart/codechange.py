"""The unattended /fpl-improve run: when a scheduled `fpl improve` changed no setting and wrote a brief
for a code change, Claude Code works on that brief headless, following the skill
(.claude/skills/fpl-improve/SKILL.md, section "Unattended runs"), and the result goes to Discord.

    fpl improve --code              work on the latest brief now
    fpl improve --code --if-due     for the scheduler: a brief no run has worked on, with at least
                                    MIN_HOURS left before the autopilot's planning window opens

Nothing reaches `main` without you. The session works in its own git worktree, on a branch of its own
(improve/gwNN-auto); a change that passes its tests waits there for `git merge`. The autopilot runs from
the main working tree, so an experiment must never touch it: the worktree shares `.venv` and
`data/raw` (downloads only), and has copies of `data/processed`, `reports` and the live tuning files,
so trial code cannot change the caches, reviews or settings the autopilot reads. Its `.env` holds only
FPL_TEAM_ID: no sign-in, odds key or webhook, and those are taken out of the session's environment.
The session may only run what is in ALLOWED (tests, `fpl evaluate/review/backtest`, git on its branch);
anything else is refused, never asked.

Afterwards this module checks rather than trusts: a change reported as kept must have commits and pass
the test suite on its branch. It writes the session's account to reports/gwNN_code_change.md, appends
its ledger line to tuning/ledger.md (so a rejected idea is not tried again), and reports on Discord.
A run is started once per brief, success or not: a failed one is reported and waits for you
(`fpl improve --code` runs it again). FPL_SELF_IMPROVE_CODE=off switches the scheduled run off.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import improve, tuning
from .autopilot import _report
from .config import REPORTS, ROOT

MAX_HOURS = 4.0      # longest session: a backtest before and after a planner change fits
MIN_HOURS = 1.0      # less time than this before the planning window opens: wait for the next free spell
WORKTREES = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "fpl-smart" / "worktrees"
SECRETS = {"DISCORD_WEBHOOK_URL", "ODDS_API_KEY", "FPL_REFRESH_TOKEN"}
OUTCOMES = ("kept", "rejected", "no_change", "failed")

SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": list(OUTCOMES),
                    "description": "kept: committed on the branch and every validation passed; rejected: tried, "
                                   "failed validation, reverted; no_change: nothing well aimed, nothing tried; "
                                   "failed: could not finish"},
        "summary": {"type": "string", "description": "One or two plain sentences for the Discord report: what was "
                                                     "chosen and why, and the result with its numbers"},
        "ledger_line": {"type": "string", "description": "The '- Code change: ...' line for tuning/ledger.md, or "
                                                         "empty when no idea was tried"},
        "report": {"type": "string", "description": "Markdown account: the evidence that aimed the change, what "
                                                    "changed, every validation number, the verdict"},
    },
    "required": ["outcome", "summary", "ledger_line", "report"],
}

# Commands the session may run, relative to its worktree. Anything else is refused (--permission-mode
# dontAsk), so this list, not the session's judgement, keeps it off the FPL account and off `main`.
ALLOWED = [
    "Read", "Glob", "Grep", "Edit", "Write",
    "Bash(.venv/bin/python -m pytest)", "Bash(.venv/bin/python -m pytest *)",
    "Bash(.venv/bin/fpl evaluate)", "Bash(.venv/bin/fpl evaluate *)", "Bash(.venv/bin/fpl review *)",
    "Bash(.venv/bin/fpl backtest *)", "Bash(.venv/bin/fpl player *)", "Bash(.venv/bin/fpl top *)",
    "Bash(.venv/bin/python scripts/eval_minutes.py*)", "Bash(.venv/bin/python scripts/tune_optimiser.py*)",
    "Bash(git status*)", "Bash(git diff*)", "Bash(git log*)", "Bash(git show*)", "Bash(git add *)",
    "Bash(git commit *)", "Bash(git restore *)", "Bash(git revert *)", "Bash(git reset *)",
    "Bash(ls*)", "Bash(wc *)", "Bash(head *)", "Bash(tail *)",
]
DENIED = [
    "Bash(git push*)", "Bash(git merge*)", "Bash(git switch*)", "Bash(git checkout*)", "Bash(git branch*)",
    "Bash(git worktree*)", "Bash(git -C*)", "Bash(.venv/bin/fpl apply*)", "Bash(.venv/bin/fpl auto*)",
    "Bash(.venv/bin/fpl plan*)", "Bash(.venv/bin/fpl login*)", "Bash(.venv/bin/fpl improve*)",
    "WebFetch", "WebSearch",
]

PROMPT = """/fpl-improve reports/gw{gw:02d}_review.md

Unattended run for GW{gw}, started by the scheduler: follow the skill's "Unattended runs" section. \
You are in a git worktree on branch `{branch}`. Time budget: {hours:.1f} hours, until {until} UTC; \
the session is stopped then."""


def enabled() -> bool:
    return os.environ.get("FPL_SELF_IMPROVE_CODE", "on").strip().lower() != "off"


@dataclass
class Result:
    gw: int
    branch: str
    outcome: str
    summary: str
    report: str = ""
    ledger_line: str = ""
    commits: int = 0
    files: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    session: str | None = None
    cost: float | None = None
    worktree: Path | None = None       # kept for a look when the run failed


# ----------------------------------------------------------------------------- when
def due(log=print) -> tuple[int, float] | None:
    """For the scheduler: the brief of the last scheduled `fpl improve` run, if no session has worked on
    it, and the hours it may take (up to MAX_HOURS, ending before the planning window opens)."""
    st = improve._state()
    gw = st.get("brief_gw")
    if not gw or gw <= st.get("code_gw", 0):
        return None
    free, nxt = improve.hours_free()
    hours = min(MAX_HOURS, free)
    if hours < MIN_HOURS:
        log(f"improve --code: the GW{gw} brief waits; {max(free, 0):.1f} hours before the GW{nxt} planning "
            f"window opens (needs {MIN_HOURS:g})")
        return None
    return gw, hours


def run(review: str | None = None, if_due: bool = False, log=print) -> int:
    if if_due:
        if not enabled():
            return 0
        d = due(log)
        if d is None:
            return 0
        gw, hours = d
        improve._save_state(code_gw=gw)      # before the session: a costly run is never repeated by itself
    else:
        gw, hours = _gw_of(review), MAX_HOURS
        if gw is None:
            log("no code-change brief found: `fpl improve` writes one when no setting passes")
            return 2
    if not (REPORTS / f"gw{gw:02d}_improvement.md").exists():
        log(f"no brief for GW{gw} (reports/gw{gw:02d}_improvement.md)")
        return 2
    try:
        res = work(gw, hours, log)
    except Exception as e:  # noqa: BLE001  (unattended: say so on Discord; auto.sh only writes the log)
        log("improve --code: the run failed\n" + traceback.format_exc().rstrip())
        _report(log, f"GW{gw} code change failed", [f"`fpl improve --code` stopped with {type(e).__name__}: {e}"],
                ["The traceback is in reports/auto.log. It is not tried again by itself; "
                 "`fpl improve --code` runs it again."], kind="error")
        return 1
    record(res, log)
    return 1 if res.outcome == "failed" else 0


def _gw_of(review: str | None) -> int | None:
    if review:
        m = re.search(r"gw(\d+)_", Path(review).name)
        return int(m.group(1)) if m else None
    briefs = sorted(REPORTS.glob("gw[0-9][0-9]_improvement.md"))
    return int(briefs[-1].name[2:4]) if briefs else None


# ----------------------------------------------------------------------------- the session
def claude_bin() -> str | None:
    """Claude Code's CLI. The Windows scheduler starts WSL without ~/.local/bin on PATH, hence the fallback."""
    for c in (os.environ.get("FPL_CLAUDE"), "claude", str(Path.home() / ".local" / "bin" / "claude")):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def _git(*args, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _free_branch(root: Path, base: Path, name: str) -> tuple[str, Path]:
    """`name`, or name-2, name-3... when an earlier run (or you) already used it."""
    _git("worktree", "prune", cwd=root)
    for i in range(1, 100):
        b = name if i == 1 else f"{name}-{i}"
        wt = base / b.replace("/", "-")
        exists = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{b}"], cwd=root,
                                capture_output=True).returncode == 0
        if not exists and not wt.exists():
            return b, wt
    raise RuntimeError(f"no free branch name for {name}")


def _env_keys(root: Path) -> set[str]:
    try:
        lines = (root / ".env").read_text().splitlines()
    except OSError:
        return set()
    return {ln.split("=", 1)[0].strip() for ln in lines if "=" in ln and not ln.strip().startswith("#")}


def prepare(root: Path, wt: Path) -> dict[str, str]:
    """Fill a fresh worktree with what the code reads but git does not hold. Returns the session's environment."""
    if (root / ".venv").exists():
        (wt / ".venv").symlink_to(root / ".venv")
    (wt / "data").mkdir(exist_ok=True)
    if (root / "data" / "raw").exists():
        (wt / "data" / "raw").symlink_to(root / "data" / "raw")
    if (root / "data" / "processed").exists():
        shutil.copytree(root / "data" / "processed", wt / "data" / "processed")
    if (root / "reports").exists():
        shutil.copytree(root / "reports", wt / "reports", ignore=shutil.ignore_patterns(".auto.lock", "auto.log"))
    # The settings in effect, which the scheduled runs may not have committed. skip-worktree keeps the
    # copies out of the branch's commits, so merging the branch cannot conflict with main's own.
    copied = []
    for f in ("params.json", "ledger.md"):
        if (root / "tuning" / f).exists():
            shutil.copy2(root / "tuning" / f, wt / "tuning" / f)
            copied.append(f"tuning/{f}")
    if copied:
        _git("update-index", "--skip-worktree", *copied, cwd=wt)
    team = os.environ.get("FPL_TEAM_ID")
    if team:
        (wt / ".env").write_text(f"FPL_TEAM_ID={team}\n")
    drop = SECRETS | _env_keys(root)
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["PYTHONPATH"] = str(wt / "src")      # ahead of the editable install, which points at the main tree
    if team:
        env["FPL_TEAM_ID"] = team
    return env


def session(claude: str, prompt: str, wt: Path, env: dict, hours: float) -> tuple[dict, str | None]:
    """Run Claude Code headless in the worktree. Returns its JSON result and why it failed to give one."""
    cmd = [claude, "-p", "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
           "--permission-mode", "dontAsk", "--allowedTools", *ALLOWED, "--disallowedTools", *DENIED]
    if os.environ.get("FPL_IMPROVE_MODEL"):
        cmd += ["--model", os.environ["FPL_IMPROVE_MODEL"]]
    p = subprocess.Popen(cmd, cwd=wt, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(prompt, timeout=hours * 3600)
    except subprocess.TimeoutExpired:
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, None)):     # the session and its commands
            try:
                os.killpg(p.pid, sig)
                p.communicate(timeout=wait)
                break
            except (subprocess.TimeoutExpired, ProcessLookupError):
                continue
        return {}, f"it ran out of its {hours:.1f}-hour budget and was stopped"
    try:
        data = json.loads(out)
    except ValueError:
        tail = (err or out or "").strip().splitlines()[-3:]
        return {}, f"Claude Code exited with code {p.returncode}: " + " ".join(tail)
    if data.get("is_error") or not data.get("structured_output"):
        return data, "the session ended without a result: " + str(data.get("result") or data.get("subtype"))[:300]
    return data, None


def work(gw: int, hours: float, log=print, root: Path = ROOT, base: Path = WORKTREES, claude: str | None = None) -> Result:
    claude = claude or claude_bin()
    if not claude:
        raise RuntimeError("Claude Code (`claude`) was not found; set FPL_CLAUDE to its full path")
    head = _git("rev-parse", "HEAD", cwd=root)
    branch, wt = _free_branch(root, base, f"improve/gw{gw:02d}-auto")
    wt.parent.mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "-q", "-b", branch, str(wt), head, cwd=root)
    env = prepare(root, wt)
    until = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=hours)).strftime("%a %d %b %H:%M")
    log(f"improve --code: GW{gw} brief: Claude Code works on branch {branch} in {wt} (up to {hours:.1f} hours)")
    data, why = session(claude, PROMPT.format(gw=gw, branch=branch, hours=hours, until=until), wt, env, hours)

    so = data.get("structured_output") or {}
    res = Result(gw, branch, so.get("outcome") if so.get("outcome") in OUTCOMES else "failed",
                 str(so.get("summary") or why or "no summary"), str(so.get("report") or ""),
                 str(so.get("ledger_line") or "").strip(), session=data.get("session_id"),
                 cost=data.get("total_cost_usd"))
    if why:
        res.outcome, res.summary = "failed", why
    res.commits = int(_git("rev-list", "--count", f"{head}..{branch}", cwd=root))
    res.files = _git("diff", "--name-only", f"{head}..{branch}", cwd=root).split()
    if res.outcome == "kept":
        if not res.commits:
            res.outcome = "failed"
            res.problems.append("It reported a change kept but committed nothing on its branch.")
        else:
            t = subprocess.run([str(wt / ".venv" / "bin" / "python"), "-m", "pytest", "-q"], cwd=wt, env=env,
                               capture_output=True, text=True)
            if t.returncode:
                res.outcome = "failed"
                res.problems.append("The test suite fails on its branch: "
                                    + " ".join(t.stdout.strip().splitlines()[-1:]))
    if any(f.startswith("tuning/") for f in res.files):
        res.problems.append("Its branch changes tuning/: settings move only through `fpl improve`; drop those "
                            "files from the merge.")
    if "src/fpl_smart/execute.py" in res.files:
        res.problems.append("Its branch changes execute.py, the checks before anything is sent to your FPL "
                            "account: read that diff before merging.")
    denied = data.get("permission_denials") or []
    if denied:
        names = sorted({str(d.get("tool_name", "?")) for d in denied if isinstance(d, dict)})
        res.notes.append(f"{len(denied)} command(s) it tried were not permitted ({', '.join(names)}).")

    if res.outcome == "failed":
        res.worktree = wt                      # left for a look; `git worktree remove --force` clears it
    else:
        _git("worktree", "remove", "--force", str(wt), cwd=root)
        if not res.commits:
            _git("branch", "-D", branch, cwd=root)
    return res


# ----------------------------------------------------------------------------- reporting
def record(res: Result, log=print) -> None:
    """The account in reports/, the ledger line, and the Discord report."""
    out = REPORTS / f"gw{res.gw:02d}_code_change.md"
    head = [f"# Gameweek {res.gw}: unattended code change", "",
            f"- Outcome: **{res.outcome}**", f"- Branch: `{res.branch}` ({res.commits} commit(s)"
            + (f": {', '.join(res.files)})" if res.files else ")"),
            f"- Summary: {res.summary}"]
    head += [f"- Problem: {p}" for p in res.problems] + [f"- Note: {n}" for n in res.notes]
    if res.session:
        head.append(f"- Claude Code session: `{res.session}`"
                    + (f" (API-equivalent cost ${res.cost:.2f})" if res.cost is not None else ""))
    if res.worktree:
        head.append(f"- Worktree kept for a look: `{res.worktree}`")
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head) + "\n\n" + (res.report.strip() or "_The session gave no account._") + "\n")

    if res.ledger_line:
        line = res.ledger_line.splitlines()[0].strip().lstrip("-* ").strip()
        if not line.startswith("Code change:"):
            line = "Code change: " + line
        when = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        tuning.append_ledger([f"- {line} [unattended run {when}, outcome {res.outcome}]"])

    account = f"The full account is in reports/{out.name}."
    if res.outcome == "kept":
        title, kind = f"GW{res.gw} code change ready to merge", "ok"
        lines = [res.summary, f"On branch `{res.branch}`: {res.commits} commit(s), {len(res.files)} file(s); "
                 "the test suite passes there.",
                 f"To use it: `git merge {res.branch}` in the project folder (the autopilot runs it from then on). "
                 f"To drop it: `git branch -D {res.branch}`.", account]
    elif res.outcome == "rejected":
        title, kind = f"GW{res.gw} code change tried and rejected", "info"
        lines = [res.summary, "Recorded in tuning/ledger.md so it is not tried again. Nothing for you to do.", account]
    elif res.outcome == "no_change":
        title, kind = f"GW{res.gw} code change: nothing well aimed this week", "info"
        lines = [res.summary, "Nothing for you to do.", account]
    else:
        title, kind = f"GW{res.gw} code change failed", "error"
        lines = [res.summary, account] + ([f"Its worktree is kept for a look: {res.worktree}"] if res.worktree else [])
    _report(log, title, lines + res.notes, res.problems, kind=kind)
    log(f"improve --code: GW{res.gw}: {res.outcome}; written {out}")
