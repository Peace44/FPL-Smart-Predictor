"""Settings the self-improvement loop may change, and where the live code reads them.

The tuned values live in `tuning/params.json` (committed, so every change is in git history);
`tuning/ledger.md` says why each one changed. Anything not in the file keeps the default written
in the code (PlayerParams, TeamParams, SolverOptions, xp.UNFLAGGED_MISS_CUT).

`KNOBS` is the whole list of what `fpl improve` is allowed to touch, with hard bounds and the
size of one step. A knob outside this list can only be changed by editing the code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace

import pandas as pd

from .config import ROOT

DIR = ROOT / "tuning"
PARAMS = DIR / "params.json"
LEDGER = DIR / "ledger.md"
SECTIONS = ("player", "team", "solver", "availability")


@dataclass(frozen=True)
class Knob:
    section: str
    name: str
    lo: float
    hi: float
    step: float
    mult: bool = False          # multiplicative step (value * step, value / step) instead of +/- step
    integer: bool = False

    @property
    def key(self) -> str:
        return f"{self.section}.{self.name}"

    def neighbours(self, value: float) -> list[float]:
        up, down = (value * self.step, value / self.step) if self.mult else (value + self.step, value - self.step)
        out = []
        for v in (down, up):
            v = min(self.hi, max(self.lo, v))
            v = round(v) if self.integer else round(v, 4)
            if abs(v - value) > 1e-9 and v not in out:
                out.append(v)
        return out


KNOBS = {k.key: k for k in [
    # points model (player_model.PlayerParams)
    Knob("player", "hl_days", 100, 1000, 1.3, mult=True),
    Knob("player", "k_goal", 2, 30, 1.4, mult=True),
    Knob("player", "k_ast", 2, 30, 1.4, mult=True),
    Knob("player", "xg_blend", 0.0, 1.0, 0.1),
    Knob("player", "xa_blend", 0.0, 1.0, 0.1),
    Knob("player", "k_dc", 1, 20, 1.5, mult=True),
    Knob("player", "k_bps", 2, 40, 1.5, mult=True),
    Knob("player", "k_bres", 3, 60, 1.5, mult=True),
    Knob("player", "k_sv", 3, 60, 1.5, mult=True),
    Knob("player", "k_yc", 3, 60, 1.5, mult=True),
    # team strength (team_model.TeamParams)
    Knob("team", "halflife_days", 40, 400, 1.3, mult=True),
    Knob("team", "xg_weight", 0.3, 1.0, 0.05),
    Knob("team", "market_weight", 0.0, 0.9, 0.1),
    Knob("team", "ridge", 0.2, 10.0, 1.5, mult=True),
    Knob("team", "odds_blend", 0.3, 1.0, 0.1),
    # availability (xp.py): how much a missing injury flag lowers a regular's risk of missing the
    # next match, and how far FPL's "x% chance of playing" is taken at face value (1 = exactly)
    Knob("availability", "unflagged_miss_cut", 0.0, 0.9, 0.1),
    Knob("availability", "doubtful_scale", 0.3, 1.3, 0.1),
    # optimiser (solver.SolverOptions)
    Knob("solver", "decay", 0.5, 0.95, 0.05),
    Knob("solver", "ft_value", 0.0, 3.0, 0.5),
    Knob("solver", "hit_cost", 4.0, 8.0, 1.0),
]}

AVAILABILITY_DEFAULTS = {"unflagged_miss_cut": 0.45, "doubtful_scale": 1.0}


# ----------------------------------------------------------------------------- reading
def load() -> dict:
    """{"player": {...}, "team": {...}, "solver": {...}, "availability": {...}} of tuned values only."""
    try:
        raw = json.loads(PARAMS.read_text())
    except (OSError, ValueError):
        raw = {}
    return {s: dict(raw.get(s) or {}) for s in SECTIONS}


def _apply(obj, values: dict):
    names = {f.name for f in fields(obj)}
    return replace(obj, **{k: v for k, v in values.items() if k in names})


def player_params(overrides: dict | None = None):
    from .player_model import PlayerParams
    return _apply(PlayerParams(), {**load()["player"], **(overrides or {})})


def team_params(overrides: dict | None = None):
    from .team_model import TeamParams
    return _apply(TeamParams(), {**load()["team"], **(overrides or {})})


def solver_values() -> dict:
    """Tuned SolverOptions fields, to pass as keyword arguments."""
    from .solver import SolverOptions
    names = {f.name for f in fields(SolverOptions)}
    return {k: v for k, v in load()["solver"].items() if k in names}


def availability(overrides: dict | None = None) -> dict:
    return {**AVAILABILITY_DEFAULTS, **load()["availability"], **(overrides or {})}


def current(key: str) -> float:
    """The value in effect for a knob ("section.name"), tuned or default."""
    section, name = key.split(".", 1)
    if section == "player":
        return float(getattr(player_params(), name))
    if section == "team":
        return float(getattr(team_params(), name))
    if section == "availability":
        return float(availability()[name])
    if section == "solver":
        from .solver import SolverOptions
        return float(solver_values().get(name, getattr(SolverOptions(), name)))
    raise KeyError(key)


def fingerprint(sections=("player", "team")) -> str:
    """Short hash of the tuned values that change projections ("" when nothing is tuned), so caches
    built with other settings are not reused."""
    t = load()
    sub = {s: t[s] for s in sections if t[s]}
    if not sub:
        return ""
    return hashlib.sha1(json.dumps(sub, sort_keys=True).encode()).hexdigest()[:8]


# ----------------------------------------------------------------------------- writing
def set_value(key: str, value: float, reason: str, when: str | None = None) -> dict:
    """Change one knob (within its bounds) and keep the old value in the file's history."""
    knob = KNOBS[key]
    if not knob.lo - 1e-9 <= value <= knob.hi + 1e-9:
        raise ValueError(f"{key}={value} is outside its bounds [{knob.lo}, {knob.hi}]")
    old = current(key)
    try:
        raw = json.loads(PARAMS.read_text())
    except (OSError, ValueError):
        raw = {}
    raw.setdefault(knob.section, {})[knob.name] = value
    hist = raw.setdefault("history", [])
    entry = {"id": (max((h.get("id", 0) for h in hist), default=0) + 1), "key": key, "old": old, "new": value,
             "when": when or str(pd.Timestamp.now(tz="UTC").floor("s")), "reason": reason}
    hist.append(entry)
    DIR.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(raw, indent=1, sort_keys=False) + "\n")
    return entry


def history() -> list[dict]:
    try:
        return list(json.loads(PARAMS.read_text()).get("history") or [])
    except (OSError, ValueError):
        return []


def revert(change_id: int) -> dict:
    """Put a knob back to the value it had before change `change_id`."""
    h = next((x for x in history() if x.get("id") == change_id), None)
    if h is None:
        raise ValueError(f"no change with id {change_id} in {PARAMS}")
    return set_value(h["key"], h["old"], f"revert of change {change_id}")


def append_ledger(lines: list[str]) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text("# Self-improvement ledger\n\nOne entry per run of `fpl improve`: the limiting factor the "
                          "review found, what was tried, and what changed. `tuning/params.json` holds the values in "
                          "effect; `fpl improve --revert ID` undoes a change.\n")
    with LEDGER.open("a") as f:
        f.write("\n" + "\n".join(lines) + "\n")
