"""A new FPL season reuses the gameweek numbers. Everything kept per gameweek (reports/gwNN/ with the
plans, the autopilot's record and the projections saved before each deadline, the reviews, the
self-improvement briefs) and the records of which gameweek was last reported, reviewed or worked
on would otherwise be read as this season's: GW1 would find last season's GW1, and last season's
GW38 review would stay the newest. On the first run in a new season they move to
reports/<last season>/ and everything starts afresh. Tuned settings (tuning/) carry over.
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from . import fpl_api
from .config import CURRENT_SEASON, REPORTS

MARKER = REPORTS / "season.json"
SEASONAL = ("gw[0-9][0-9]", "gw[0-9][0-9]_review.md", "gw[0-9][0-9]_improvement.md",
            "autopilot-state.json", "improve-state.json")


def of(bs: dict) -> str:
    """The season FPL is on, e.g. "2026-27", from the year of the first deadline."""
    y = pd.Timestamp(bs["events"][0]["deadline_time"]).year
    return f"{y}-{(y + 1) % 100:02d}"


def recorded() -> str | None:
    try:
        return json.loads(MARKER.read_text()).get("season")
    except (OSError, ValueError):
        return None


def config_problem(season: str) -> str | None:
    if season == CURRENT_SEASON:
        return None
    return (f"src/fpl_smart/config.py still has CURRENT_SEASON = \"{CURRENT_SEASON}\" while FPL is on {season}: "
            "the data download and the points model treat this season's matches as the old season's until the "
            "season constants are moved on.")


def rollover(bs: dict | None = None, log=print) -> str | None:
    """Move the last season's files out of the way if FPL has started a new season. Returns that
    season, or None. The first run only records the season: the files already there are this season's."""
    from .autopilot import _report
    season = of(bs or fpl_api.bootstrap(1.0))
    old = recorded()
    if old == season:
        problem = config_problem(season)
        if problem:
            log(problem)
        return None
    REPORTS.mkdir(parents=True, exist_ok=True)
    if old is None:
        MARKER.write_text(json.dumps({"season": season}))
        return None
    dest = REPORTS / old
    dest.mkdir(exist_ok=True)
    moved = []
    for pattern in SEASONAL:
        for p in sorted(REPORTS.glob(pattern)):
            if (dest / p.name).exists():
                log(f"season: {p.name} is already in {dest}; left in place")
                continue
            shutil.move(str(p), str(dest / p.name))
            moved.append(p.name)
    MARKER.write_text(json.dumps({"season": season}))
    log(f"new season {season}: moved {len(moved)} of last season's files to {dest}")
    problem = config_problem(season)
    _report(log, f"New season {season}",
            [f"Last season's gameweek files ({len(moved)}) moved to reports/{old}/.",
             "The autopilot's records, the reviews and self-improvement start afresh; tuned settings carry over."],
            [problem] if problem else [], kind="info")
    return old
