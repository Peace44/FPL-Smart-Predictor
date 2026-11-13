"""Build one tidy player-match table across past seasons (vaastav archive) and the
current season (official API), plus a team-match table derived from it.

Players are keyed by `code` and teams by `team_code`, both stable across seasons.
"""
from __future__ import annotations

import pandas as pd
import requests

from . import fpl_api
from .config import CURRENT_SEASON, PAST_SEASONS, PROCESSED, RAW

VAASTAV = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
VAASTAV_FILES = {"merged_gw.csv": "gws/merged_gw.csv", "fixtures.csv": "fixtures.csv",
                 "teams.csv": "teams.csv", "players_raw.csv": "players_raw.csv"}

RENAME = {
    "round": "gw", "goals_scored": "goals", "expected_goals": "xg", "expected_assists": "xa",
    "expected_goals_conceded": "xgc", "clean_sheets": "cs", "goals_conceded": "gc",
    "yellow_cards": "yc", "red_cards": "rc", "own_goals": "og", "penalties_missed": "pm",
    "penalties_saved": "ps", "total_points": "pts", "was_home": "home",
    "clearances_blocks_interceptions": "cbi", "recoveries": "rec", "tackles": "tck",
    "defensive_contribution": "defcon", "xP": "fpl_xp", "opponent_team": "opp",
}
KEEP = ["season", "gw", "fixture", "kickoff", "code", "element", "name", "pos", "team", "team_code",
        "opp", "opp_code", "home", "minutes", "starts", "goals", "assists", "xg", "xa", "xgc", "cs",
        "gc", "saves", "bonus", "bps", "yc", "rc", "og", "pm", "ps", "pts", "value", "selected",
        "cbi", "rec", "tck", "defcon", "fpl_xp", "has_xg", "has_defcon"]


def download_vaastav(force: bool = False) -> None:
    for s in PAST_SEASONS:
        d = RAW / "vaastav" / s
        d.mkdir(parents=True, exist_ok=True)
        for local, remote in VAASTAV_FILES.items():
            f = d / local
            if f.exists() and f.stat().st_size > 0 and not force:
                continue
            r = requests.get(f"{VAASTAV}/{s}/{remote}", timeout=120)
            r.raise_for_status()
            f.write_bytes(r.content)


def _finish(df: pd.DataFrame, fx: pd.DataFrame, teams: pd.DataFrame, season: str) -> pd.DataFrame:
    """Attach team ids/codes from the fixture list and normalise columns."""
    df = df.rename(columns=RENAME)
    df["season"] = season
    df["kickoff"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df["home"] = df["home"].astype(bool)
    fx = fx.set_index("id")
    th, ta = df["fixture"].map(fx["team_h"]), df["fixture"].map(fx["team_a"])
    df["team"] = th.where(df["home"], ta).astype(int)
    df["opp"] = ta.where(df["home"], th).astype(int)
    code = teams.set_index("id")["code"]
    df["team_code"] = df["team"].map(code)
    df["opp_code"] = df["opp"].map(code)
    for c in ["xg", "xa", "xgc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["cbi", "rec", "tck", "defcon", "fpl_xp", "starts"]:
        if c not in df:
            df[c] = float("nan")
    return df


def load_past_season(season: str) -> pd.DataFrame:
    d = RAW / "vaastav" / season
    df = pd.read_csv(d / "merged_gw.csv", low_memory=False)
    players = pd.read_csv(d / "players_raw.csv").set_index("id")
    df = df.drop_duplicates(["element", "fixture"])
    df["code"] = df["element"].map(players["code"])
    df["pos"] = df["element"].map(players["element_type"])
    df["name"] = df["element"].map(players["web_name"])
    df = df[df["pos"].between(1, 4)].copy()  # drop 2024-25 assistant managers
    df = _finish(df, pd.read_csv(d / "fixtures.csv"), pd.read_csv(d / "teams.csv"), season)
    # FPL only published xG / starts from the 2022 World Cup break onwards.
    gw_xg = df.groupby("gw")["xg"].transform("sum")
    df["has_xg"] = gw_xg > 0
    df["has_defcon"] = df["defcon"].notna()
    # FPL's own projection is missing (all zero) in some archived gameweeks.
    gw_xp = df.groupby("gw")["fpl_xp"].transform(lambda s: s.abs().sum())
    df.loc[gw_xp == 0, "fpl_xp"] = float("nan")
    return df[KEEP]


def load_current_season(max_age_h: float = 12.0) -> pd.DataFrame:
    bs = fpl_api.bootstrap()
    els = pd.DataFrame(bs["elements"]).set_index("id")
    els = els[els["element_type"].between(1, 4)]
    summaries = fpl_api.all_element_summaries(list(els.index), max_age_h)
    rows = [h for s in summaries.values() for h in s["history"]]
    if not rows:
        return pd.DataFrame(columns=KEEP)
    df = pd.DataFrame(rows)
    df["code"] = df["element"].map(els["code"])
    df["pos"] = df["element"].map(els["element_type"])
    df["name"] = df["element"].map(els["web_name"])
    fx = pd.DataFrame(fpl_api.fixtures())
    df = _finish(df, fx, pd.DataFrame(bs["teams"]), CURRENT_SEASON)
    # Only keep matches that have finished (the API lists live ones too).
    done = fx.set_index("id")["finished"]
    df = df[df["fixture"].map(done).fillna(False)].copy()
    df["has_xg"] = True
    df["has_defcon"] = True
    df["fpl_xp"] = float("nan")
    return df[KEEP]


def build_matches(max_age_h: float = 12.0, save: bool = True) -> pd.DataFrame:
    """Player-match table for all seasons, sorted by time."""
    download_vaastav()
    parts = [load_past_season(s) for s in PAST_SEASONS] + [load_current_season(max_age_h)]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    num = ["minutes", "starts", "goals", "assists", "cs", "gc", "saves", "bonus", "bps", "yc", "rc",
           "og", "pm", "ps", "pts", "value", "selected", "cbi", "rec", "tck", "defcon"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    # before FPL published starts, treat 60+ minutes as a start
    df["starts"] = df["starts"].where(df["has_xg"], (df["minutes"] >= 60).astype(float))
    df = df.sort_values(["kickoff", "fixture", "code"]).reset_index(drop=True)
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED / "matches.parquet")
    return df


def load_matches() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "matches.parquet")


def team_matches(m: pd.DataFrame) -> pd.DataFrame:
    """One row per team per fixture: goals and xG for/against."""
    g = (m.groupby(["season", "gw", "fixture", "kickoff", "team_code", "opp_code", "home"], as_index=False)
           .agg(xg=("xg", "sum"), goals=("goals", "sum"), og=("og", "sum"), has_xg=("has_xg", "first")))
    opp = g[["season", "fixture", "team_code", "xg", "goals", "og"]].rename(
        columns={"team_code": "opp_code", "xg": "xga", "goals": "opp_goals", "og": "opp_og"})
    g = g.merge(opp, on=["season", "fixture", "opp_code"], how="left")
    g["gf"] = g["goals"] + g["opp_og"]  # own goals by the opponent count for us
    g["ga"] = g["opp_goals"] + g["og"]
    return g.sort_values(["kickoff", "fixture"]).reset_index(drop=True)
