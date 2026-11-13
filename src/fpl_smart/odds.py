"""Upcoming Premier League odds from the-odds-api.com, converted to expected goals.

The free tier allows 500 credits a month and this call costs 2 (h2h + totals, one region),
so responses are cached on disk and re-used for `max_age_h` hours.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from . import fpl_api
from .config import RAW, odds_api_key
from .market import devig, implied_lambdas

URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
CACHE = RAW / "odds_api" / "epl_odds.json"

# the-odds-api full club names -> FPL names (only where a simple normalisation is not enough)
ALIASES = {
    "manchester united": "man utd", "manchester city": "man city", "tottenham hotspur": "spurs",
    "nottingham forest": "nott'm forest", "wolverhampton wanderers": "wolves",
    "brighton and hove albion": "brighton", "newcastle united": "newcastle",
    "west ham united": "west ham", "leeds united": "leeds", "afc bournemouth": "bournemouth",
    "sheffield united": "sheffield utd", "leicester city": "leicester",
}


def _norm(name: str) -> str:
    n = name.lower().strip()
    n = ALIASES.get(n, n)
    for suffix in (" city", " town", " fc", " afc"):
        if n.endswith(suffix) and n != "man city":
            n = n[: -len(suffix)]
    return n


def fetch(max_age_h: float = 6.0) -> tuple[list, dict]:
    """Raw odds events plus quota info. Falls back to a stale cache if the API is unavailable."""
    if CACHE.exists() and (time.time() - CACHE.stat().st_mtime) < max_age_h * 3600:
        return json.loads(CACHE.read_text()), {"cached": True}
    key = odds_api_key()
    if not key:
        return (json.loads(CACHE.read_text()) if CACHE.exists() else []), {"error": "no ODDS_API_KEY"}
    try:
        r = requests.get(URL, params={"apiKey": key, "regions": "uk", "markets": "h2h,totals",
                                      "oddsFormat": "decimal"}, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        return (json.loads(CACHE.read_text()) if CACHE.exists() else []), {"error": repr(e)[:200]}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(r.text)
    return r.json(), {"cached": False, "remaining": r.headers.get("x-requests-remaining"),
                      "used": r.headers.get("x-requests-used")}


def upcoming_lambdas(max_age_h: float = 6.0) -> tuple[pd.DataFrame, dict]:
    """One row per listed match: home_code, away_code, kickoff, lam_h, lam_a, n_books."""
    events, info = fetch(max_age_h)
    codes = {_norm(t["name"]): t["code"] for t in fpl_api.bootstrap()["teams"]}
    rows, unmapped = [], set()
    for ev in events:
        h, a = codes.get(_norm(ev["home_team"])), codes.get(_norm(ev["away_team"]))
        if h is None or a is None:
            unmapped.update({ev["home_team"], ev["away_team"]} - {n for n in (ev["home_team"], ev["away_team"]) if _norm(n) in codes})
            continue
        ph, pa, po = [], [], []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                o = {x["name"]: x for x in mk["outcomes"]}
                if mk["key"] == "h2h" and {ev["home_team"], ev["away_team"], "Draw"} <= set(o):
                    p = devig(o[ev["home_team"]]["price"], o["Draw"]["price"], o[ev["away_team"]]["price"])
                    ph.append(p[0]); pa.append(p[2])
                elif mk["key"] == "totals" and {"Over", "Under"} <= set(o) and o["Over"].get("point") == 2.5:
                    po.append(devig(o["Over"]["price"], o["Under"]["price"])[0])
        if len(ph) < 2:
            continue
        lh, la = implied_lambdas(float(np.median(ph)), float(np.median(pa)), float(np.median(po)) if po else None)
        rows.append({"home_code": h, "away_code": a, "kickoff": pd.Timestamp(ev["commence_time"]),
                     "lam_h": lh, "lam_a": la, "n_books": len(ph), "has_totals": bool(po)})
    info["unmapped"] = sorted(unmapped)
    return pd.DataFrame(rows), info
