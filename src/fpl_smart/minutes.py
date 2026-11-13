"""Minutes model: P(plays), P(plays 60+) and expected minutes for a player's h-th next
team match, learned with gradient boosting from his recent pattern of starts/minutes.

The training data has no injury news, so the model learns the *average* consequence of
e.g. "missed the last three matches". Live availability flags are applied on top in xp.py.
"""
from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import PROCESSED

MAX_H = 10
FEATURES = [
    "m1", "m2", "m3", "m4", "s1", "s2", "s3", "s4",
    "st_e2", "st_e5", "st_e12", "pl_e2", "pl_e5", "pl_e12", "p60_e2", "p60_e5", "p60_e12",
    "min_e2", "min_e5", "min_e12", "min_when_played", "zero_run", "start_run",
    "season_rows", "season_start_share", "season_min_share", "n_rows",
    "gap_days", "season_change", "team_change", "pos", "value", "own_rel", "h",
]
TARGETS = {"play": "binary", "p60": "binary", "mins": "regression"}
# Out-of-sample check (3 held-out seasons) showed the minutes regression runs 2-4 minutes low for
# likely starters and slightly high for fringe players; this monotone map removes that.
EMIN_PRED = [0.0, 0.9, 11.3, 29.6, 50.1, 65.4, 72.6, 77.5, 82.3, 86.7, 90.0]
EMIN_ACT = [0.0, 0.7, 9.7, 28.7, 52.5, 69.9, 75.5, 79.1, 82.5, 86.5, 90.0]


def _ewm(g: pd.core.groupby.SeriesGroupBy, hl: float) -> pd.Series:
    return g.transform(lambda s: s.ewm(halflife=hl, adjust=True).mean())


def add_state(m: pd.DataFrame) -> pd.DataFrame:
    """Per player-row features describing his minutes pattern up to and including that row."""
    m = m.sort_values(["code", "kickoff", "fixture"]).reset_index(drop=True)
    m["i"] = m.groupby("code").cumcount()
    m["played"] = (m["minutes"] > 0).astype(float)
    m["p60"] = (m["minutes"] >= 60).astype(float)
    m["start"] = m["starts"].fillna(0).clip(0, 1).astype(float)
    g = m.groupby("code")
    for k in range(1, 5):
        m[f"m{k}"] = g["minutes"].shift(k - 1)
        m[f"s{k}"] = g["start"].shift(k - 1)
    for hl in (2, 5, 12):
        m[f"st_e{hl}"] = _ewm(g["start"], hl)
        m[f"pl_e{hl}"] = _ewm(g["played"], hl)
        m[f"p60_e{hl}"] = _ewm(g["p60"], hl)
        m[f"min_e{hl}"] = _ewm(g["minutes"], hl)
    num = _ewm(m["minutes"].groupby(m["code"]), 8)
    den = _ewm(g["played"], 8)
    m["min_when_played"] = (num / den.where(den > 0)).astype(float)

    def run_length(flag: pd.Series) -> pd.Series:
        # consecutive True count ending at each row, within player
        brk = (~flag).groupby(m["code"]).cumsum()
        return flag.groupby([m["code"], brk]).cumsum()

    m["zero_run"] = run_length(m["played"] == 0)
    m["start_run"] = run_length(m["start"] == 1)
    gs = m.groupby(["code", "season"])
    m["season_rows"] = gs.cumcount() + 1
    m["season_start_share"] = gs["start"].cumsum() / m["season_rows"]
    m["season_min_share"] = gs["minutes"].cumsum() / (90.0 * m["season_rows"])
    m["n_rows"] = m["i"] + 1
    m["own_rel"] = m["selected"] / m.groupby(["season", "gw"])["selected"].transform("max")
    return m


STATE_COLS = [f for f in FEATURES if f not in
              ("gap_days", "season_change", "team_change", "pos", "value", "own_rel", "h")]


def training_frame(ms: pd.DataFrame, max_h: int = MAX_H) -> pd.DataFrame:
    """(state after row t) -> minutes in row t+h, for h = 1..max_h. `ms` must come from add_state."""
    parts = []
    g = ms.groupby("code")
    base = ms[["code", "i", "kickoff", "season", "team_code"] + STATE_COLS]
    for h in range(1, max_h + 1):
        tgt = ms[["code", "i", "kickoff", "season", "team_code", "minutes", "pos", "gw"]].copy()
        tgt["i"] = tgt["i"] - h
        tgt = tgt.rename(columns={"kickoff": "t_kick", "season": "t_season", "team_code": "t_team",
                                  "minutes": "t_minutes", "gw": "t_gw"})
        d = base.merge(tgt, on=["code", "i"], how="inner")
        # price / ownership as known at the decision time (the row right after the base row)
        nxt = ms[["code", "i", "value", "own_rel"]].copy()
        nxt["i"] = nxt["i"] - 1
        d = d.merge(nxt, on=["code", "i"], how="left")
        d["h"] = h
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    # first-ever rows: no history at all (new signings) -> all state NaN, horizon 1
    first = ms[ms["i"] == 0][["code", "kickoff", "season", "team_code", "minutes", "pos", "gw",
                              "value", "own_rel"]].rename(
        columns={"kickoff": "t_kick", "season": "t_season", "team_code": "t_team",
                 "minutes": "t_minutes", "gw": "t_gw"})
    first["h"] = 1
    first["kickoff"] = pd.Series(pd.NaT, index=first.index, dtype=ms["kickoff"].dtype)
    d = pd.concat([d, first], ignore_index=True)
    d["gap_days"] = (d["t_kick"] - d["kickoff"]).dt.total_seconds() / 86400.0
    d["season_change"] = (d["t_season"] != d["season"]).astype(float).where(d["season"].notna())
    d["team_change"] = (d["t_team"] != d["team_code"]).astype(float).where(d["team_code"].notna())
    d["play"] = (d["t_minutes"] > 0).astype(int)
    d["p60"] = (d["t_minutes"] >= 60).astype(int)
    d["mins"] = d["t_minutes"].astype(float)
    return d


class MinutesModel:
    def __init__(self, models: dict[str, lgb.Booster]):
        self.models = models

    @classmethod
    def train(cls, frame: pd.DataFrame, rounds: int = 350) -> "MinutesModel":
        models = {}
        X = frame[FEATURES]
        for name, objective in TARGETS.items():
            params = dict(objective=objective, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                          verbose=-1, seed=7, num_threads=0)
            models[name] = lgb.train(params, lgb.Dataset(X, frame[name]), num_boost_round=rounds)
        return cls(models)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for name, mdl in self.models.items():
            out[name] = mdl.predict(X[FEATURES])
        # enforce coherence: P(60+) <= P(play); minutes within what those probabilities allow
        out["play"] = out["play"].clip(0, 1)
        out["p60"] = np.minimum(out["p60"].clip(0, 1), out["play"])
        lo = 60 * out["p60"] + 1 * (out["play"] - out["p60"])
        hi = 90 * out["p60"] + 59 * (out["play"] - out["p60"])
        out["mins"] = pd.Series(np.interp(out["mins"], EMIN_PRED, EMIN_ACT), index=out.index).clip(lo, hi)
        return out.rename(columns={"play": "p_play", "p60": "p60", "mins": "emin"})

    def save(self, name: str) -> None:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED / f"minutes_{name}.pkl", "wb") as f:
            pickle.dump({k: v.model_to_string() for k, v in self.models.items()}, f)

    @classmethod
    def load(cls, name: str) -> "MinutesModel":
        with open(PROCESSED / f"minutes_{name}.pkl", "rb") as f:
            raw = pickle.load(f)
        return cls({k: lgb.Booster(model_str=v) for k, v in raw.items()})


def backtest_features(ms: pd.DataFrame) -> pd.DataFrame:
    """For every row, the features available at its gameweek deadline (state after the player's
    last row before that gameweek began). Returns FEATURES columns aligned to ms.index."""
    k = ms.groupby(["code", "season", "gw"]).cumcount() + 1   # 1 normally, 2 in a double gameweek
    base_i = ms["i"] - k
    base = ms[["code", "i", "kickoff", "season", "team_code"] + STATE_COLS].rename(
        columns={"i": "base_i", "kickoff": "b_kick", "season": "b_season", "team_code": "b_team"})
    d = ms[["code", "kickoff", "season", "team_code", "pos", "value", "own_rel"]].copy()
    d["base_i"] = base_i
    d["h"] = k
    d = d.reset_index().merge(base, on=["code", "base_i"], how="left").set_index("index")
    d["gap_days"] = (d["kickoff"] - d["b_kick"]).dt.total_seconds() / 86400.0
    d["season_change"] = (d["season"] != d["b_season"]).astype(float).where(d["b_season"].notna())
    d["team_change"] = (d["team_code"] != d["b_team"]).astype(float).where(d["b_team"].notna())
    return d[FEATURES].reindex(ms.index)
