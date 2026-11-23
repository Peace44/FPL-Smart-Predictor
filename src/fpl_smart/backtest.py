"""Replay the optimiser over a season, one gameweek at a time, using only what was known
before each deadline, and score its teams with the points that were actually scored.

What this does and does not show:
* Projections for gameweek g use matches before g only; the minutes model is trained on
  earlier seasons; calibration is refitted as the season goes.
* No injury news exists in the archive, so the replay cannot react to flags the way the live
  tool does. Fixture dates are the final ones (a postponement is "known" slightly early).
* Chips are left out: the replay measures transfer, line-up and captaincy decisions.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field, replace

import pandas as pd

from . import history, market
from . import minutes as MN
from . import player_model as PM
from . import team_model as TM
from . import tuning
from . import xp as xp_mod
from .config import CURRENT_SEASON, PROCESSED
from .my_team import MAX_FT, TeamState, selling_price
from .optimise import worker_pool
from .scoring import score_gw
from .solver import SolverOptions, solve

STRATEGIES = ("plan", "myopic", "hold")


@dataclass
class GWData:
    gw: int
    xp: pd.DataFrame                 # player code x gameweek
    players: pd.DataFrame            # index code; name, pos, team_id, now_cost
    points: dict                     # actual gameweek points by code
    minutes: dict


@dataclass
class BacktestResult:
    season: str
    strategy: str
    weeks: pd.DataFrame
    total: float
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- projections
def _minutes_model(ms: pd.DataFrame, season: str, start: pd.Timestamp) -> MN.MinutesModel:
    tag = f"pre_{season}"
    if (PROCESSED / f"minutes_{tag}.pkl").exists():
        return MN.MinutesModel.load(tag)
    fr = MN.training_frame(ms)
    mm = MN.MinutesModel.train(fr[fr["t_kick"] < start])
    mm.save(tag)
    return mm


def project_season(season: str, horizon: int = 8, first_gw: int = 1, last_gw: int = 38,
                   use_cache: bool = True, log=lambda m: None) -> dict[int, GWData]:
    """Walk-forward projections for every gameweek of `season` (cached on disk, per set of tuned
    model settings). For the season in progress the cache is rebuilt once newer gameweeks exist."""
    fp = tuning.fingerprint()
    cache = PROCESSED / (f"backtest_{season}_h{horizon}" + (f"_{fp}" if fp else "") + ".pkl")
    m = None
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            out = pickle.load(f)
        complete = all(g in out for g in range(first_gw, min(last_gw, max(out)) + 1))
        if complete and season == CURRENT_SEASON:
            m = history.build_matches(max_age_h=12.0)
            played = m.loc[m["season"] == season, "gw"]
            complete = len(played) == 0 or max(out) >= min(last_gw, int(played.max()))
        if complete:
            return {g: d for g, d in out.items() if first_gw <= g <= last_gw}

    params, tp = tuning.player_params(), tuning.team_params()
    if m is None:
        m = history.load_matches() if season != CURRENT_SEASON else history.build_matches(max_age_h=12.0)
    tm = market.attach_market(history.team_matches(m))
    seasons = [s for s in sorted(tm["season"].unique()) if "2023-24" <= s <= season]
    twf = TM.walk_forward(tm, seasons, tp)
    ms = PM.prepare(MN.add_state(m), twf)
    cur = ms[ms["season"] == season]
    if cur.empty:
        raise ValueError(f"no data for season {season}")
    start = cur["kickoff"].min()
    mm = _minutes_model(ms, season, start)
    teams = sorted(cur["team_code"].unique())
    tms = tm[tm["season"] == season]
    wf = twf[twf["season"] == season].set_index(["fixture", "team_code"])[["lam", "lam_a"]]
    defcon = bool(cur["has_defcon"].any())

    out: dict[int, GWData] = {}
    gws = sorted(int(g) for g in cur["gw"].unique())
    for g in gws:
        when = cur.loc[cur["gw"] == g, "kickoff"].min()
        # refit the calibration constants every gameweek (about a second)
        cal, msg = PM.fit_calibration(ms, (ms["kickoff"] < when).to_numpy(), params)
        # who is in the game, and at what price, at this deadline
        snap = cur[cur["gw"] <= g].groupby("code").tail(1).set_index("code")
        players = pd.DataFrame({"code": snap.index, "pos": snap["pos"], "team_code": snap["team_code"],
                                "value": snap["value"].astype(float),
                                "own_rel": snap["selected"] / max(snap["selected"].max(), 1)}, index=snap.index)
        fx = tms[(tms["gw"] >= g) & (tms["gw"] < g + horizon)][
            ["season", "gw", "fixture", "kickoff", "team_code", "opp_code", "home"]].copy()
        fx = fx.sort_values(["team_code", "kickoff"])
        fx["h"] = fx.groupby("team_code").cumcount() + 1
        r = TM.fit(tm, when, teams, start, tp)
        fx["lam"] = r.predict(fx["team_code"], fx["opp_code"], fx["home"])
        fx["lam_a"] = r.predict(fx["opp_code"], fx["team_code"], ~fx["home"])
        now = fx["gw"] == g          # odds exist for the imminent gameweek only
        key = list(zip(fx.loc[now, "fixture"], fx.loc[now, "team_code"]))
        fx.loc[now, ["lam", "lam_a"]] = wf.reindex(key).to_numpy()
        fx[["lam", "lam_a"]] = fx[["lam", "lam_a"]].fillna(
            pd.DataFrame({"lam": r.predict(fx["team_code"], fx["opp_code"], fx["home"]),
                          "lam_a": r.predict(fx["opp_code"], fx["team_code"], ~fx["home"])}, index=fx.index))
        det = xp_mod.project_core(msg, players, fx, when, cal, mm, params, defcon=defcon)
        xp = det.pivot_table(index="pid", columns="gw", values="xp", aggfunc="sum").reindex(players.index).fillna(0.0)
        xp = xp.reindex(columns=range(g, min(g + horizon, gws[-1] + 1)), fill_value=0.0)
        act = cur[cur["gw"] == g].groupby("code").agg(pts=("pts", "sum"), mins=("minutes", "sum"))
        pl = pd.DataFrame({"name": snap["name"], "pos": snap["pos"].astype(int), "team_id": snap["team_code"].astype(int),
                           "now_cost": snap["value"].astype(int)}, index=snap.index)
        out[g] = GWData(g, xp, pl, act["pts"].to_dict(), act["mins"].to_dict())
        log(f"{season} GW{g}: projected {len(players)} players")
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return {g: d for g, d in out.items() if first_gw <= g <= last_gw}


# ----------------------------------------------------------------------------- simulation
def options_for(strategy: str, base: SolverOptions | None = None) -> tuple[SolverOptions, int | None]:
    """Solver options and horizon override for a named strategy."""
    base = replace(base or SolverOptions(**tuning.solver_values()), use_chips=False, chips=[], time_limit=25,
                   gap=0.004, threads=1)
    if strategy == "plan":
        return base, None
    if strategy == "myopic":        # only ever looks at the next gameweek
        return base, 1
    if strategy == "hold":          # picks a squad once, then only sets the line-up
        return replace(base, no_transfers=True, max_hits_per_gw=0, ft_value=0.0), None
    raise ValueError(f"unknown strategy {strategy}")


def simulate(data: dict[int, GWData], season: str, strategy: str = "plan", horizon: int = 8,
             opt: SolverOptions | None = None, budget: int = 1000, initial_horizon: int = 8,
             log=lambda m: None) -> BacktestResult:
    """Every strategy starts from the same initial squad (default settings, `initial_horizon` weeks),
    so differences in the totals come from in-season decisions only."""
    initial_opt, _ = options_for("plan", None)
    opt, h_override = options_for(strategy, opt)
    gws = sorted(data)
    squad: list[int] = []
    buy: dict[int, int] = {}
    bank, ft = budget, 1
    rows = []
    for g in gws:
        d = data[g]
        cost = d.players["now_cost"].to_dict()
        gone = [p for p in squad if p not in d.players.index]
        if gone:  # left the game entirely: keep a zero projection so the squad stays valid
            raise RuntimeError(f"GW{g}: squad players missing from the player list: {gone}")
        sell = {p: selling_price(buy[p], cost[p]) for p in squad}
        state = TeamState(0, g, list(squad), dict(buy), sell, bank, ft)
        first = not squad
        if first:   # initial squad: unlimited changes, same horizon for every strategy
            o = replace(initial_opt, chips=[{"name": "wildcard", "gws": [g]}], wildcard_gw=g, time_limit=60)
            plan = solve(d.xp, d.players, state, initial_horizon, o)
        else:
            plan = solve(d.xp, d.players, state, h_override or horizon, opt)
        w = plan.weeks[0]
        for p in w.transfers_out:
            buy.pop(p, None)
        for p in w.transfers_in:
            buy[p] = cost[p]
        squad, bank = list(w.squad), w.bank
        pos = d.players["pos"].to_dict()
        sc = score_gw(w.lineup, w.bench, w.captain, w.vice, d.points, d.minutes, pos)
        pts = sc.points - 4 * w.hits
        n_tr = 0 if first else len(w.transfers_in)
        rows.append({"gw": g, "points": pts, "xp": w.xp, "transfers": n_tr, "hits": w.hits,
                     "captain": d.players.at[w.captain, "name"], "cap_pts": d.points.get(w.captain, 0),
                     "subs": len(sc.subs), "bank": bank / 10,
                     "value": (bank + sum(selling_price(buy[p], cost[p]) for p in squad)) / 10,
                     "in": ", ".join(d.players.at[p, "name"] for p in w.transfers_in) if not first else "(initial squad)",
                     "out": ", ".join(d.players.at[p, "name"] for p in w.transfers_out) if not first else ""})
        if not first:
            ft = min(MAX_FT, max(ft - n_tr, 0) + 1)
        log(f"{season} {strategy} GW{g}: {pts:.0f} pts (xp {w.xp:.1f}), {n_tr} transfer(s), running total "
            f"{sum(r['points'] for r in rows):.0f}")
    weeks = pd.DataFrame(rows).set_index("gw")
    return BacktestResult(season, strategy, weeks, float(weeks["points"].sum()))


def _sim_job(args) -> BacktestResult:
    data, season, strategy, horizon = args
    return simulate(data, season, strategy, horizon)


def run(season: str = "2025-26", strategies: list[str] | None = None, horizon: int = 8, first_gw: int = 1,
        last_gw: int = 38, workers: int | None = None, log=lambda m: None) -> list[BacktestResult]:
    strategies = list(strategies or STRATEGIES)
    data = project_season(season, horizon, first_gw, last_gw, log=log)
    log(f"simulating {', '.join(strategies)} over GW{min(data)}-{max(data)}")
    jobs = [(data, season, s, horizon) for s in strategies]
    if (workers or len(jobs)) <= 1 or len(jobs) == 1:
        return [_sim_job(j) for j in jobs]
    with worker_pool(min(workers or len(jobs), len(jobs))) as ex:
        return list(ex.map(_sim_job, jobs))


def render(results: list[BacktestResult]) -> str:
    if not results:
        return "no results"
    L = [f"Backtest {results[0].season}, GW{results[0].weeks.index.min()}-{results[0].weeks.index.max()} (no chips)", ""]
    desc = {"plan": "multi-week optimiser", "myopic": "next gameweek only", "hold": "initial squad, no transfers"}
    L.append(f"{'strategy':<32}{'points':>8}{'per GW':>8}{'projected':>11}{'transfers':>11}{'hits':>6}")
    for r in results:
        w = r.weeks
        L.append(f"{r.strategy + ' (' + desc.get(r.strategy, '') + ')':<32}{r.total:>8.0f}{w['points'].mean():>8.1f}"
                 f"{w['xp'].sum():>11.0f}{int(w['transfers'].sum()):>11d}{int(w['hits'].sum()):>6d}")
    L.append("")
    pts = pd.DataFrame({r.strategy: r.weeks["points"] for r in results})
    L.append("Points by gameweek:")
    L.append(pts.round(0).astype(int).T.to_string())
    return "\n".join(L)
