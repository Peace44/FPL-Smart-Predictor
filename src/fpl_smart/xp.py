"""Expected points for every player in every upcoming gameweek.

`project_core` is the shared engine (used live and in backtests): given what was known at
a point in time it produces per-fixture expected points. `project_live` wraps it with the
current FPL state, injury flags and bookmaker odds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import fpl_api, history, market, odds as odds_mod
from . import minutes as MN
from . import player_model as PM
from . import team_model as TM
from .config import CURRENT_SEASON, PROCESSED


@dataclass
class Projection:
    players: pd.DataFrame            # one row per player (index: FPL element id)
    xp: pd.DataFrame                 # element id x gameweek -> expected points
    detail: pd.DataFrame             # one row per player per fixture with components
    fixtures: pd.DataFrame           # upcoming team-fixtures with expected goals
    ratings: TM.Ratings | None = None
    next_gw: int = 0
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- core engine
def project_core(ms: pd.DataFrame, players: pd.DataFrame, fixtures: pd.DataFrame, when: pd.Timestamp,
                 cal: PM.Calibration, mm: MN.MinutesModel, params: PM.PlayerParams | None = None,
                 scoring: dict | None = None, defcon: bool = True,
                 avail: pd.DataFrame | None = None, fit_blend: pd.Series | None = None,
                 unflagged: pd.Series | None = None, miss_cut: float | None = None) -> pd.DataFrame:
    """Per player-fixture expected points.

    ms: prepared history (minutes.add_state + player_model.prepare, bonus residuals filled).
    players: index = any id; columns code, pos, team_code, value, own_rel.
    fixtures: columns gw, kickoff, team_code, opp_code, home, lam, lam_a, h (1 = team's next match).
    avail: optional (player index x gw) availability multipliers in [0, 1].
    fit_blend: optional per-player weight in [0, 1]; where > 0 the minutes prediction is blended
        towards "as he played before his current absence" (used for flagged/returning players).
    unflagged: optional per-player bool; True = FPL shows no injury/suspension flag (see below).
    miss_cut: share of an unflagged regular's risk of missing the next match that the missing flag
        removes (default UNFLAGGED_MISS_CUT; tuned values come from tuning.availability()).
    The result keeps `p_play_base` (before the flag adjustments), `clear` and `avail`, so the
    gameweek review can check, and `fpl improve` re-fit, how the flags were used.
    """
    params = params or PM.PlayerParams()
    past = ms[ms["kickoff"] < when]
    last = past.groupby("code").tail(1).set_index("code")

    d = players.reset_index(names="pid").merge(fixtures, on="team_code", how="inner")
    if d.empty:
        return d

    def minute_features(state_rows: pd.DataFrame) -> pd.DataFrame:
        st = state_rows.reindex(d["code"].to_numpy())
        X = pd.DataFrame({c: st[c].to_numpy() for c in MN.STATE_COLS}, index=d.index)
        lk = state_rows["kickoff"].reindex(d["code"].to_numpy())
        lk.index = d.index
        X["gap_days"] = (d["kickoff"] - lk).dt.total_seconds() / 86400.0
        ls = last["season"].reindex(d["code"].to_numpy()).to_numpy()
        X["season_change"] = np.where(pd.isna(ls), np.nan, (ls != d["season"].to_numpy()).astype(float))
        lt = last["team_code"].reindex(d["code"].to_numpy()).to_numpy(dtype=float)
        X["team_change"] = np.where(np.isnan(lt), np.nan, (lt != d["team_code"].to_numpy()).astype(float))
        X["pos"], X["value"], X["own_rel"] = d["pos"].to_numpy(), d["value"].to_numpy(), d["own_rel"].to_numpy()
        X["h"] = d["h"].clip(upper=MN.MAX_H).to_numpy()
        return X

    mins = mm.predict(minute_features(last))
    if fit_blend is not None and (fit_blend > 0).any():
        # state at the last match he actually played: "how he is used when fit"
        last_played = past[past["minutes"] > 0].groupby("code").tail(1).set_index("code")
        Xf = minute_features(last_played)
        Xf["zero_run"] = 0.0
        fit = mm.predict(Xf)
        w = fit_blend.reindex(d["pid"]).fillna(0).to_numpy()[:, None]
        has = last_played.index.to_numpy()
        w = w * np.isin(d["code"].to_numpy(), has)[:, None]
        mins = pd.DataFrame(w * np.maximum(fit.to_numpy(), mins.to_numpy()) + (1 - w) * mins.to_numpy(),
                            index=mins.index, columns=mins.columns)
    base = mins["p_play"].to_numpy().copy()
    clear = np.zeros(len(d), dtype=bool)
    if unflagged is not None:
        clear = unflagged.reindex(d["pid"]).fillna(False).to_numpy(dtype=bool)
        mins = _unflagged_uplift(mins, clear, d["h"].to_numpy(),
                                 UNFLAGGED_MISS_CUT if miss_cut is None else miss_cut)
    a = np.ones(len(d))
    if avail is not None:
        a = np.array([avail.at[p, g] if (p in avail.index and g in avail.columns) else 1.0
                      for p, g in zip(d["pid"], d["gw"])])
        mins = mins.mul(a, axis=0)
    d = pd.concat([d, mins.assign(p_play_base=base, clear=clear, avail=a)], axis=1)

    S = PM.state_asof(ms, when, params.hl_days)
    S = S.reindex(d["code"].to_numpy()).fillna(0.0)
    S.index = d.index
    R = PM.rates(S, d["pos"], d["value"], cal, params)
    key = d["team_code"].astype(str) + "_" + d["fixture"].astype(str)
    E = PM.expected_points_team(R, key, d["pos"], d["lam"], d["lam_a"], d["p_play"], d["p60"], d["emin"], cal,
                                scoring=scoring, defcon=defcon)
    return pd.concat([d, R.drop(columns=["g_share", "a_share", "g_ev", "g_pr", "a_ev", "a_pr"]), E], axis=1)


UNFLAGGED_MISS_CUT = 0.45  # share of a regular starter's absences that FPL flags before the deadline


def unflagged_play(p: np.ndarray, clear: np.ndarray, h: np.ndarray, miss_cut: float = UNFLAGGED_MISS_CUT) -> np.ndarray:
    """P(plays) once a missing injury flag is taken into account (see _unflagged_uplift)."""
    p = np.asarray(p, dtype=float)
    nailed = np.clip((p - 0.5) / 0.4, 0, 1)
    fade = np.clip(1.0 - 0.25 * (np.asarray(h, dtype=float) - 1), 0, 1)
    cut = miss_cut * nailed * fade * np.asarray(clear, dtype=bool)
    return 1 - (1 - p) * (1 - cut)


def _unflagged_uplift(mins: pd.DataFrame, clear: np.ndarray, h: np.ndarray,
                      miss_cut: float = UNFLAGGED_MISS_CUT) -> pd.DataFrame:
    """The minutes model never saw injury news, so its P(plays) averages over flagged and unflagged
    players. Knowing a regular starter carries no flag removes a good part of his risk of missing
    the *next* match; the effect fades for later matches and for rotation players."""
    p = mins["p_play"].to_numpy()
    new_p = unflagged_play(p, clear, h, miss_cut)
    scale = np.divide(new_p, p, out=np.ones_like(p), where=p > 1e-9)
    out = mins.copy()
    out["p_play"] = new_p
    out["p60"] = np.minimum(mins["p60"].to_numpy() * scale, new_p)
    out["emin"] = np.minimum(mins["emin"].to_numpy() * scale, 90.0 * new_p)
    return out


def team_fixture_rows(fx: pd.DataFrame, codes: dict[int, int]) -> pd.DataFrame:
    """Fixture list (team_h/team_a ids) -> one row per team per fixture, with stable codes."""
    f = fx.copy()
    f["kickoff"] = pd.to_datetime(f["kickoff_time"], utc=True)
    h = pd.DataFrame({"gw": f["event"], "kickoff": f["kickoff"], "fixture": f["id"],
                      "team_code": f["team_h"].map(codes), "opp_code": f["team_a"].map(codes), "home": True})
    a = pd.DataFrame({"gw": f["event"], "kickoff": f["kickoff"], "fixture": f["id"],
                      "team_code": f["team_a"].map(codes), "opp_code": f["team_h"].map(codes), "home": False})
    out = pd.concat([h, a], ignore_index=True).sort_values(["team_code", "kickoff"])
    out["h"] = out.groupby("team_code").cumcount() + 1
    return out


def add_lambdas(tf: pd.DataFrame, ratings: TM.Ratings, odds: pd.DataFrame | None = None,
                blend: float = 0.7) -> pd.DataFrame:
    tf = tf.copy()
    tf["lam"] = ratings.predict(tf["team_code"], tf["opp_code"], tf["home"])
    tf["lam_a"] = ratings.predict(tf["opp_code"], tf["team_code"], ~tf["home"])
    tf["rat_lam"], tf["rat_lam_a"] = tf["lam"], tf["lam_a"]
    tf["has_odds"] = False
    if odds is not None and len(odds):
        o = {(r.home_code, r.away_code): r for r in odds.itertuples()}
        for i, r in tf.iterrows():
            key = (r.team_code, r.opp_code) if r.home else (r.opp_code, r.team_code)
            m = o.get(key)
            if m is None or abs((m.kickoff - r.kickoff).total_seconds()) > 4 * 86400:
                continue
            lf, la = (m.lam_h, m.lam_a) if r.home else (m.lam_a, m.lam_h)
            tf.at[i, "lam"] = blend * lf + (1 - blend) * r.rat_lam
            tf.at[i, "lam_a"] = blend * la + (1 - blend) * r.rat_lam_a
            tf.at[i, "has_odds"] = True
    return tf


# ----------------------------------------------------------------------------- availability
_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_DATE = re.compile(r"(?:expected back|until)\s+(\d{1,2})\s+([a-z]{3})", re.I)
_UNKNOWN_RAMP = [0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
_RETURN_RAMP = [0.70, 0.85, 0.95]


def parse_return_date(news: str, now: pd.Timestamp) -> pd.Timestamp | None:
    m = _DATE.search(news or "")
    if not m or m.group(2).lower() not in _MONTHS:
        return None
    day, month = int(m.group(1)), _MONTHS[m.group(2).lower()]
    year = now.year + (1 if month < now.month - 6 else 0)
    try:
        return pd.Timestamp(year=year, month=month, day=day, tz="UTC")
    except ValueError:
        return None


def doubtful_chance(c: float, scale: float = 1.0) -> float:
    """FPL's "c chance of playing" for a doubtful player, read with `scale` (1 = at face value,
    below 1 = he plays more often than FPL's number says)."""
    return float(min(1.0, max(0.0, 1.0 - (1.0 - c) * scale)))


def availability(el: pd.DataFrame, gw_dates: dict[int, pd.Timestamp], now: pd.Timestamp,
                 doubtful_scale: float = 1.0) -> pd.DataFrame:
    """Probability each player is available for each gameweek, from FPL status flags and news."""
    gws = sorted(gw_dates)
    out = pd.DataFrame(1.0, index=el.index, columns=gws)
    for pid, r in el.iterrows():
        status, news = r["status"], r.get("news") or ""
        chance = r.get("chance_of_playing_next_round")
        c = None if pd.isna(chance) else float(chance) / 100.0
        if status == "a" and (c is None or c >= 1.0):
            continue
        low = news.lower()
        if status == "u" or "joined" in low or "left the club" in low or "on loan" in low or "transferred" in low:
            out.loc[pid] = 0.0
            continue
        back = parse_return_date(news, now)
        if back is not None:
            k = 0
            for g in gws:
                if gw_dates[g] < back:
                    out.at[pid, g] = 0.0
                else:
                    out.at[pid, g] = 1.0 if status == "s" else (_RETURN_RAMP[k] if k < len(_RETURN_RAMP) else 1.0)
                    k += 1
            continue
        c = 0.0 if c is None and status in ("i", "s", "n") else (0.75 if c is None else c)
        if status not in ("s", "n") and 0.0 < c < 1.0:
            c = doubtful_chance(c, doubtful_scale)
        for k, g in enumerate(gws):
            if status in ("s", "n"):
                out.at[pid, g] = c if k == 0 else 1.0
            elif c == 0.0:
                out.at[pid, g] = _UNKNOWN_RAMP[min(k, len(_UNKNOWN_RAMP) - 1)]
            else:
                out.at[pid, g] = 1 - (1 - c) * 0.5 ** k
    return out


# ----------------------------------------------------------------------------- live wrapper
def _minutes_model(ms: pd.DataFrame, tag: str, retrain: bool = False) -> MN.MinutesModel:
    f = PROCESSED / f"minutes_{tag}.pkl"
    if f.exists() and not retrain:
        return MN.MinutesModel.load(tag)
    mm = MN.MinutesModel.train(MN.training_frame(ms))
    mm.save(tag)
    return mm


def project_live(horizon: int = 8, refresh: bool = True) -> Projection:
    from . import tuning
    params, team_params = tuning.player_params(), tuning.team_params()
    av = tuning.availability()
    now = pd.Timestamp.now(tz="UTC")
    bs = fpl_api.bootstrap(0.0 if refresh else 6.0)
    fx = pd.DataFrame(fpl_api.fixtures(0.0 if refresh else 6.0))
    _, nxt = fpl_api.current_and_next_gw(bs)
    if nxt is None:
        raise RuntimeError("Season is over: no upcoming gameweek.")
    last_gw = min(38, nxt + horizon - 1)

    m = history.build_matches(max_age_h=1.0 if refresh else 48.0)
    try:
        market.download_football_data([CURRENT_SEASON], force_current=refresh)
    except Exception:  # odds history is a nice-to-have; ratings still work without the latest file
        pass
    tm = market.attach_market(history.team_matches(m))
    seasons = [s for s in sorted(tm["season"].unique()) if s >= "2023-24"]
    twf = TM.walk_forward(tm, seasons, team_params)
    ms = PM.prepare(MN.add_state(m), twf)
    cal, ms = PM.fit_calibration(ms, (ms["kickoff"] < now).to_numpy(), params)

    done = m[m["season"] == CURRENT_SEASON]
    tag = f"live_{CURRENT_SEASON}_gw{int(done['gw'].max()) if len(done) else 0}"
    mm = _minutes_model(ms, tag)

    teams = pd.DataFrame(bs["teams"])
    codes = dict(zip(teams["id"], teams["code"]))
    cur = tm[tm["season"] == CURRENT_SEASON]
    ratings = TM.fit(tm, now, sorted(teams["code"]), cur["kickoff"].min() if len(cur) else now, team_params)

    up = fx[(~fx["finished"].astype(bool)) & fx["event"].notna() & fx["kickoff_time"].notna()].copy()
    tf = team_fixture_rows(up, codes)
    tf = tf[(tf["gw"] >= nxt) & (tf["gw"] <= last_gw)].copy()
    tf["season"] = CURRENT_SEASON
    odds_df, odds_info = odds_mod.upcoming_lambdas(3.0 if refresh else 48.0)
    tf = add_lambdas(tf, ratings, odds_df, team_params.odds_blend)

    el = pd.DataFrame(bs["elements"]).set_index("id")
    el = el[el["element_type"].between(1, 4)].copy()
    sel = el["selected_by_percent"].astype(float)
    players = pd.DataFrame({
        "code": el["code"], "pos": el["element_type"], "team_code": el["team"].map(codes),
        "value": el["now_cost"].astype(float), "own_rel": sel / max(sel.max(), 1e-9),
    })
    gw_dates = tf.groupby("gw")["kickoff"].min().to_dict()
    avail = availability(el, gw_dates, now, av["doubtful_scale"])

    # Flagged players, and players whose flag was cleared in the last fortnight, are judged on how
    # they were used before the absence (the flag itself carries the availability risk).
    added = pd.to_datetime(el["news_added"], utc=True, errors="coerce")
    recent = (now - added) < pd.Timedelta(days=14)
    fit_blend = pd.Series(0.0, index=el.index)
    fit_blend[el["status"].isin(["i", "d", "s", "n"])] = 1.0
    fit_blend[(el["status"] == "a") & (el["news"].fillna("") == "") & recent] = 0.8

    clear = (el["status"] == "a") & (el["chance_of_playing_next_round"].isna()
                                     | (el["chance_of_playing_next_round"] >= 100))
    detail = project_core(ms, players, tf, now, cal, mm, params, scoring=bs["game_config"]["scoring"],
                          avail=avail, fit_blend=fit_blend, unflagged=clear, miss_cut=av["unflagged_miss_cut"])
    xp = detail.pivot_table(index="pid", columns="gw", values="xp", aggfunc="sum").reindex(el.index).fillna(0.0)
    xp = xp.reindex(columns=range(nxt, last_gw + 1), fill_value=0.0)

    short = dict(zip(teams["code"], teams["short_name"]))
    out = pd.DataFrame({
        "name": el["web_name"], "team": el["team"].map(codes).map(short), "team_id": el["team"],
        "pos": el["element_type"], "price": el["now_cost"] / 10.0, "now_cost": el["now_cost"],
        "status": el["status"], "chance": el["chance_of_playing_next_round"], "news": el["news"],
        "own": sel, "form": el["form"].astype(float), "pts": el["total_points"],
        "ep_next": pd.to_numeric(el["ep_next"], errors="coerce"), "code": el["code"],
    })
    nx = detail[detail["gw"] == nxt].groupby("pid").agg(p60=("p60", "max"), emin=("emin", "sum"))
    out = out.join(nx)
    tf["team"], tf["opp"] = tf["team_code"].map(short), tf["opp_code"].map(short)
    info = {"odds": odds_info, "minutes_model": tag, "generated": str(now), "calibration": cal,
            "tuning": tuning.load()}
    return Projection(out, xp, detail, tf, ratings, nxt, info)
