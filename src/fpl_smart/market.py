"""Bookmaker odds -> expected goals.

Historical closing odds come from football-data.co.uk (free CSVs); upcoming odds come
from the-odds-api.com (see odds.py). Both are converted to a pair of Poisson means that
reproduce the de-vigged home/away win probabilities and the over/under 2.5 line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import poisson

from . import fpl_api
from .config import CURRENT_SEASON, PAST_SEASONS, RAW

FD_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
FD_NAMES = {"Man United": "Man Utd", "Tottenham": "Spurs", "Sheffield United": "Sheffield Utd"}
_K = np.arange(0, 13)


def _outcome_probs(lh: float, la: float) -> tuple[float, float, float]:
    ph, pa = poisson.pmf(_K, lh), poisson.pmf(_K, la)
    m = np.outer(ph, pa)
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def implied_lambdas(p_home: float, p_away: float, p_over25: float | None) -> tuple[float, float]:
    """Poisson means (home, away) matching normalised 1X2 probabilities and P(over 2.5)."""
    if p_over25 is None or not np.isfinite(p_over25):
        total = 2.75
    else:
        p_over25 = float(np.clip(p_over25, 0.05, 0.95))
        total = brentq(lambda t: 1 - poisson.cdf(2, t) - p_over25, 0.3, 8.0)
    target = p_home - p_away

    def f(share):  # share of total goals belonging to the home side
        h, _, a = _outcome_probs(total * share, total * (1 - share))
        return (h - a) - target

    share = brentq(f, 0.02, 0.98)
    return total * share, total * (1 - share)


def devig(*odds: float) -> list[float]:
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return [i / s for i in inv]


def _fd_file(season: str):
    return RAW / "football_data" / f"E0_{season[2:4]}{season[5:7]}.csv"


def download_football_data(seasons: list[str] | None = None, force_current: bool = True) -> None:
    for s in seasons or PAST_SEASONS + [CURRENT_SEASON]:
        f = _fd_file(s)
        if f.exists() and f.stat().st_size > 0 and not (force_current and s == CURRENT_SEASON):
            continue
        try:
            r = requests.get(FD_URL.format(f.stem.split("_")[1]), timeout=60,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(r.content)
        except requests.RequestException:
            if not f.exists():
                raise


def _norm(name: str) -> str:
    name = FD_NAMES.get(name, name).lower()
    for suffix in (" city", " town"):
        if name.endswith(suffix) and name != "man city":
            name = name[: -len(suffix)]
    return name


def _team_codes(season: str) -> dict[str, int]:
    """Normalised team name -> stable team code for one season."""
    if season == CURRENT_SEASON:
        return {_norm(t["name"]): t["code"] for t in fpl_api.bootstrap()["teams"]}
    t = pd.read_csv(RAW / "vaastav" / season / "teams.csv")
    return {_norm(n): c for n, c in zip(t["name"], t["code"])}


def load_market(seasons: list[str] | None = None) -> pd.DataFrame:
    """One row per team per match with market-implied goals for (mkt_lam) and against."""
    rows = []
    for s in seasons or PAST_SEASONS + [CURRENT_SEASON]:
        f = _fd_file(s)
        if not f.exists():
            continue
        d = pd.read_csv(f, encoding="latin-1").dropna(subset=["HomeTeam", "AwayTeam"])
        codes = _team_codes(s)
        for _, r in d.iterrows():
            h, a = codes.get(_norm(r["HomeTeam"])), codes.get(_norm(r["AwayTeam"]))
            if h is None or a is None:
                raise KeyError(f"unmapped team in {s}: {r['HomeTeam']} / {r['AwayTeam']}")
            o = None
            for pre in ("AvgC", "Avg", "B365C", "B365"):
                cols = [f"{pre}H", f"{pre}D", f"{pre}A"]
                if all(c in d.columns and pd.notna(r[c]) for c in cols):
                    o = [r[c] for c in cols]
                    ou = [r.get(f"{pre}>2.5"), r.get(f"{pre}<2.5")]
                    break
            if o is None:
                continue
            ph, _, pa = devig(*o)
            pov = devig(*ou)[0] if all(pd.notna(x) for x in ou) else None
            lh, la = implied_lambdas(ph, pa, pov)
            rows.append((s, h, a, True, lh, la))
            rows.append((s, a, h, False, la, lh))
    return pd.DataFrame(rows, columns=["season", "team_code", "opp_code", "home", "mkt_lam", "mkt_lam_a"])


def attach_market(tm: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    market = load_market() if market is None else market
    return tm.merge(market, on=["season", "team_code", "opp_code", "home"], how="left")
