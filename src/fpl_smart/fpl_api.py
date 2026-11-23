"""Thin cached client for the public Fantasy Premier League API."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .config import RAW

BASE = "https://fantasy.premierleague.com/api"
CACHE = RAW / "fpl_api"
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0"


def _get(path: str, cache_name: str, max_age_h: float) -> dict | list:
    """GET `BASE/path`, caching the JSON on disk for `max_age_h` hours."""
    f = CACHE / cache_name
    if f.exists() and (time.time() - f.stat().st_mtime) < max_age_h * 3600:
        return json.loads(f.read_text())
    last = None
    for attempt in range(4):
        try:
            r = _session.get(f"{BASE}/{path}", timeout=30)
            if r.status_code == 200:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(r.text)
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # network hiccup: retry
            last = repr(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FPL API failed for {path}: {last}")


def bootstrap(max_age_h: float = 1.0) -> dict:
    return _get("bootstrap-static/", "bootstrap.json", max_age_h)


def fixtures(max_age_h: float = 1.0) -> list:
    return _get("fixtures/", "fixtures.json", max_age_h)


def element_summary(element_id: int, max_age_h: float = 12.0) -> dict:
    return _get(f"element-summary/{element_id}/", f"elements/{element_id}.json", max_age_h)


def all_element_summaries(ids: list[int], max_age_h: float = 12.0, workers: int = 6) -> dict[int, dict]:
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(lambda i: element_summary(i, max_age_h), ids))
    return dict(zip(ids, out))


def entry(team_id: int, max_age_h: float = 0.25) -> dict:
    return _get(f"entry/{team_id}/", f"entry_{team_id}.json", max_age_h)


def entry_history(team_id: int, max_age_h: float = 0.25) -> dict:
    return _get(f"entry/{team_id}/history/", f"entry_{team_id}_history.json", max_age_h)


def entry_transfers(team_id: int, max_age_h: float = 0.25) -> list:
    return _get(f"entry/{team_id}/transfers/", f"entry_{team_id}_transfers.json", max_age_h)


def entry_picks(team_id: int, gw: int, max_age_h: float = 0.25) -> dict:
    return _get(f"entry/{team_id}/event/{gw}/picks/", f"entry_{team_id}_picks_{gw}.json", max_age_h)


def event_live(gw: int, max_age_h: float = 1.0) -> dict:
    """Every player's points and minutes in one gameweek."""
    return _get(f"event/{gw}/live/", f"live_{gw}.json", max_age_h)


def current_and_next_gw(bs: dict | None = None) -> tuple[int | None, int | None]:
    """(last started gameweek, next gameweek to plan for)."""
    bs = bs or bootstrap()
    cur = next((e["id"] for e in bs["events"] if e["is_current"]), None)
    nxt = next((e["id"] for e in bs["events"] if e["is_next"]), None)
    return cur, nxt
