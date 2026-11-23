"""Walk-forward tests of a set of settings, scored gameweek by gameweek so that two settings can be
compared pair-wise (same gameweeks, same players) with an honest standard error.

* points model: every player-match of the test seasons is projected using only what was known
  before its gameweek (team ratings refitted each gameweek, the minutes model trained before the
  season, calibration fitted on earlier data).
* availability: the projections saved before each live deadline, re-scored under other readings of
  FPL's injury flags (`reports/gwNN/projection-*.parquet`; only exists from GW11 2026-27 on).
* planning: the optimiser replayed over a season (`backtest.simulate`), points by gameweek.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest, history, market, tuning
from . import minutes as MN
from . import player_model as PM
from . import team_model as TM
from .config import CURRENT_SEASON, PROCESSED, REPORTS

SEASONS = ("2024-25", "2025-26", CURRENT_SEASON)
RATING_SEASONS_FROM = "2023-24"
METRICS = {           # name -> higher is better?
    "mse_rel": False, "bias_rel": None, "corr_rel": True, "top30": True, "top10": True,
    "logloss": False, "points": True,
}


# ----------------------------------------------------------------------------- points model
@dataclass
class Base:
    """Everything that does not depend on the settings under test."""
    tm: pd.DataFrame
    ms_state: pd.DataFrame          # minutes state + walk-forward minutes predictions
    seasons: tuple
    _prep: dict = field(default_factory=dict)


def base(refresh: bool = True, seasons=SEASONS, log=lambda m: None) -> Base:
    log("evaluation data: matches ...")
    m = history.build_matches(max_age_h=12.0) if refresh else history.load_matches()
    tm = market.attach_market(history.team_matches(m))
    st = MN.add_state(m)
    bf = MN.backtest_features(st)
    preds = []
    for s in seasons:
        rows = st.index[st["season"] == s]
        if not len(rows):
            continue
        log(f"evaluation data: minutes predictions for {s} ...")
        mm = backtest._minutes_model(st, s, st.loc[rows, "kickoff"].min())
        preds.append(mm.predict(bf.loc[rows]))
    P = pd.concat(preds)
    st = st.join(P[["p_play", "p60", "emin"]].rename(columns={"p60": "p60_hat"}))
    return Base(tm, st, tuple(s for s in seasons if (st["season"] == s).any()))


def _prepared(b: Base, tp: TM.TeamParams) -> pd.DataFrame:
    key = json.dumps(tp.__dict__, sort_keys=True)
    if key not in b._prep:
        if len(b._prep) > 4:
            b._prep.clear()
        twf = TM.walk_forward(b.tm, [s for s in sorted(b.tm["season"].unique()) if s >= RATING_SEASONS_FROM], tp)
        b._prep[key] = PM.prepare(b.ms_state, twf)
    return b._prep[key]


def project_season(b: Base, season: str, p: PM.PlayerParams, tp: TM.TeamParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows of the season, their expected points by component), walk-forward."""
    ms0 = _prepared(b, tp)
    start = ms0.loc[ms0["season"] == season, "kickoff"].min()
    train = (ms0["kickoff"] < start).to_numpy()
    if season == "2025-26":
        # the first season with defensive contributions: that part is calibrated in-season, as live
        cal_dc, _ = PM.fit_calibration(ms0, train | (ms0["season"] == season).to_numpy(), p)
        cal, _ = PM.fit_calibration(ms0, train, p)
        cal.dc, cal.priors["dc"] = cal_dc.dc, cal_dc.priors["dc"]
        ms = ms0.copy()
        ms["bres"] = (ms["bonus"] - PM.bonus_fitted(ms, cal.bonus_coef)) * (ms["minutes"] > 0) * ms["has_xg"]
    else:
        cal, ms = PM.fit_calibration(ms0, train, p)
    S = PM.state_backtest(ms, p.hl_days)
    te = ms[ms["season"] == season]
    R = PM.rates(S.loc[te.index], te["pos"], te["value"], cal, p)
    key = te["team_code"].astype(str) + "_" + te["fixture"].astype(str)
    E = PM.expected_points_team(R, key, te["pos"], te["lam"], te["lam_a"], te["p_play"], te["p60_hat"], te["emin"],
                                cal, defcon=season >= "2025-26")
    return te, E


def gw_metrics(te: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    """Per gameweek: accuracy among likely starters and the actual points of the top projections."""
    d = pd.DataFrame({"season": te["season"].to_numpy(), "gw": te["gw"].to_numpy(), "code": te["code"].to_numpy(),
                      "pts": te["pts"].to_numpy(float), "xp": E["xp"].to_numpy(float),
                      "rel": (te["p_play"] > 0.5).to_numpy()})
    out = []
    for (s, g), x in d.groupby(["season", "gw"]):
        r = x[x["rel"]]
        agg = x.groupby("code").agg(xp=("xp", "sum"), pts=("pts", "sum"))
        out.append({"season": s, "gw": int(g), "n": len(r),
                    "mse_rel": float(np.mean((r["pts"] - r["xp"]) ** 2)) if len(r) else np.nan,
                    "bias_rel": float(np.mean(r["xp"] - r["pts"])) if len(r) else np.nan,
                    "corr_rel": float(np.corrcoef(r["xp"], r["pts"])[0, 1]) if len(r) > 2 else np.nan,
                    "top30": float(agg.nlargest(30, "xp")["pts"].mean()),
                    "top10": float(agg.nlargest(10, "xp")["pts"].mean())})
    return pd.DataFrame(out).set_index(["season", "gw"])


def points_model(b: Base, player: dict | None = None, team: dict | None = None) -> pd.DataFrame:
    """Per-gameweek metrics over the base's seasons for the tuned settings plus `player`/`team` overrides."""
    p, tp = tuning.player_params(player), tuning.team_params(team)
    return pd.concat([gw_metrics(*project_season(b, s, p, tp)) for s in b.seasons])


# ----------------------------------------------------------------------------- comparison
@dataclass
class Comparison:
    metric: str
    diff: float          # candidate minus baseline, in the "better" direction (positive = better)
    se: float
    n: int
    by_season: dict

    @property
    def z(self) -> float:
        return self.diff / self.se if self.se > 0 else (np.inf if self.diff > 0 else 0.0)


def compare(base_df: pd.DataFrame, cand: pd.DataFrame, metric: str) -> Comparison:
    better = METRICS[metric]
    d = (cand[metric] - base_df[metric]).dropna()
    if better is False:
        d = -d
    by = {}
    for s, x in d.groupby(level=0):
        se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
        by[s] = {"diff": float(x.mean()), "se": se, "z": float(x.mean() / se) if se and se > 0 else np.nan, "n": len(x)}
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
    return Comparison(metric, float(d.mean()) if len(d) else 0.0, se, len(d), by)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Season averages of the per-gameweek metrics."""
    return df.groupby(level=0).mean(numeric_only=True)


# ----------------------------------------------------------------------------- availability
def availability_rows() -> pd.DataFrame:
    """Players with one fixture in a finished gameweek, as the last pre-deadline projection saw them,
    with whether they played. Only snapshots that kept the model's raw P(plays) can be re-scored."""
    from . import fpl_api, gwreview
    bs = fpl_api.bootstrap(6.0)
    events = {e["id"]: e for e in bs["events"]}
    parts = []
    for d in sorted(REPORTS.glob("gw[0-9][0-9]")):
        gw = int(re.sub(r"\D", "", d.name))
        ev = events.get(gw)
        if not ev or not ev.get("finished"):
            continue
        snap = gwreview.snapshot_for(gw, pd.Timestamp(ev["deadline_time"]))
        if snap is None or "p_play_base" not in snap:
            continue
        one = snap[snap.groupby("pid")["pid"].transform("size") == 1]
        live = {e["id"]: e["stats"]["minutes"] for e in fpl_api.event_live(gw, 24.0 * 30)["elements"]}
        parts.append(one.assign(played=one["pid"].map(live).fillna(0).gt(0).astype(float),
                                gen=pd.to_datetime(one["generated"], utc=True, errors="coerce"),
                                kick=pd.to_datetime(one["kickoff"], utc=True)))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def availability_metric(rows: pd.DataFrame, overrides: dict | None = None) -> pd.Series:
    """Log loss of "played" per row under the tuned availability settings plus `overrides`."""
    from . import xp
    av = tuning.availability(overrides)
    out = np.empty(len(rows))
    for gw, x in rows.groupby("gw"):
        el = pd.DataFrame({"status": x["status"].to_numpy(), "news": x["news"].fillna("").to_numpy(),
                           "chance_of_playing_next_round": pd.to_numeric(x["chance"], errors="coerce").to_numpy()},
                          index=x.index)
        now = x["gen"].iloc[0] if pd.notna(x["gen"].iloc[0]) else x["kick"].min() - pd.Timedelta(hours=6)
        a = xp.availability(el, {int(gw): x["kick"].min()}, now, av["doubtful_scale"])[int(gw)].to_numpy()
        p = a * xp.unflagged_play(x["p_play_base"].to_numpy(), x["clear"].to_numpy(bool), x["h"].to_numpy(),
                                  av["unflagged_miss_cut"])
        p = np.clip(p, 1e-4, 1 - 1e-4)
        y = x["played"].to_numpy()
        out[rows.index.get_indexer(x.index)] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return pd.Series(out, index=rows.index)


# ----------------------------------------------------------------------------- planning
def planning(overrides: list[dict], seasons=("2025-26", CURRENT_SEASON), workers: int = 10,
             log=lambda m: None) -> list[pd.DataFrame]:
    """Replay the optimiser for each set of SolverOptions overrides (on top of the tuned values);
    returns one per-gameweek frame (index season, gw; column points) per override."""
    from .optimise import worker_pool
    for s in seasons:
        log(f"planning replay: projections for {s} ...")
        backtest.project_season(s, 8)
    jobs = [(i, s, o) for i, o in enumerate(overrides) for s in seasons]
    log(f"planning replay: {len(jobs)} season replays on {min(workers, len(jobs))} processes ...")
    with worker_pool(min(workers, len(jobs))) as ex:
        res = list(ex.map(_planning_job, jobs))
    out = []
    for i in range(len(overrides)):
        frames = [r for (j, _), r in zip([(j, s) for j, s, _ in jobs], res) if j == i]
        out.append(pd.concat(frames))
    return out


def _planning_job(args) -> pd.DataFrame:
    i, season, o = args
    from .solver import SolverOptions
    data = backtest.project_season(season, 8)
    r = backtest.simulate(data, season, "plan", 8, SolverOptions(**{**tuning.solver_values(), **o}))
    w = r.weeks.reset_index()[["gw", "points"]].assign(season=season)
    return w.set_index(["season", "gw"])


# ----------------------------------------------------------------------------- saved runs
def save_run(df: pd.DataFrame, name: str) -> str:
    f = PROCESSED / f"eval_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.parquet"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f)
    return str(f)


def load_run(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"eval_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.parquet")
