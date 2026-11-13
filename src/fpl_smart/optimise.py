"""Decision layer on top of the solver.

* `chip_search`: the solver takes the wildcard and free-hit weeks as fixed inputs (as free
  variables they make the model intractable), so this tries the candidate weeks in parallel
  and keeps the best combination. Bench boost and triple captain are timed inside each solve.
* `robustness`: re-solves with noisy projections to show how often each of this week's moves
  survives being wrong about the numbers.
"""
from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .my_team import TeamState
from .solver import Plan, SolverOptions, chip_windows, solve


# Workers must be started fresh, not forked: HiGHS keeps a global thread pool, and a forked copy of
# a process that has already solved a model inherits that pool without its threads and returns
# no solution.
SPAWN = multiprocessing.get_context("spawn")


@contextlib.contextmanager
def worker_pool(workers: int):
    """A pool of freshly started worker processes. Spawned workers re-import the launching script;
    when that is not a real file (code piped to `python -`), hide it so they still start."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    hide = bool(path) and not os.path.exists(path)
    if hide:
        del main.__file__
    try:
        with ProcessPoolExecutor(workers, mp_context=SPAWN) as ex:
            yield ex
    finally:
        if hide:
            main.__file__ = path


def default_workers() -> int:
    """Leave two cores free, and allow roughly 450 MB of free memory per solver process."""
    n = max(1, min(10, (os.cpu_count() or 2) - 2))
    try:
        free_gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
        n = max(1, min(n, int(free_gb / 0.45)))
    except (ValueError, OSError, AttributeError):
        pass
    return n


def _job(args) -> Plan | Exception:
    xp, players, state, horizon, opt, warm = args
    try:
        return solve(xp, players, state, horizon, opt, warm)
    except Exception as e:  # one failed scenario must not sink the whole search
        where = traceback.extract_tb(e.__traceback__)[-1]
        return RuntimeError(f"{type(e).__name__}: {e} ({where.name}, line {where.lineno})")


def solve_many(jobs: list[tuple], workers: int | None = None) -> list[Plan | Exception]:
    """jobs: (xp, players, state, horizon, opt, warm). Runs them in separate processes."""
    workers = min(workers or default_workers(), len(jobs))
    if workers <= 1:
        return [_job(j) for j in jobs]
    with worker_pool(workers) as ex:
        return list(ex.map(_job, jobs))


def horizon_gws(xp: pd.DataFrame, state: TeamState, horizon: int | None) -> list[int]:
    gws = [g for g in sorted(xp.columns) if g >= state.next_gw]
    return gws[:horizon] if horizon else gws


def available_chips(state: TeamState, opt: SolverOptions) -> list:
    return opt.chips if opt.chips is not None else (state.chips if opt.use_chips else [])


# ----------------------------------------------------------------------------- chip timing
@dataclass
class ChipSearch:
    scenarios: list[Plan]           # every (wildcard week, free hit week) tried, best first
    windows: dict                   # chip -> {"gws", "expires"} inside the horizon
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Plan:
        return self.scenarios[0]

    def choose(self, margin: float = 1.5) -> Plan:
        """The plan to act on. A wildcard or free hit cannot be taken back, so when a plan that
        keeps both in hand this gameweek scores within `margin` of the leader, prefer it."""
        top = self.scenarios[0]
        now = top.weeks[0].gw
        for p in self.scenarios:
            if top.objective - p.objective > margin:
                break
            if now not in (p.wildcard_gw, p.freehit_gw):
                return p
        return top

    def without_squad_chips(self) -> Plan | None:
        return next((p for p in self.scenarios if p.wildcard_gw is None and p.freehit_gw is None), None)


def chip_search(xp: pd.DataFrame, players: pd.DataFrame, state: TeamState, horizon: int | None = None,
                opt: SolverOptions | None = None, workers: int | None = None, search_time: int = 60,
                search_gap: float = 0.005, polish_time: int = 150, top: int = 3,
                log=lambda msg: None) -> ChipSearch:
    """Find the best gameweeks for the wildcard and the free hit.

    Stage 1 tries each chip on its own in every allowed week; stage 2 combines the best few
    weeks of each; stage 3 gives the leading combinations more solver time (warm-started)."""
    opt = opt or SolverOptions()
    gws = horizon_gws(xp, state, horizon)
    win = chip_windows(available_chips(state, opt), gws)
    wc_weeks = [opt.wildcard_gw] if opt.wildcard_gw else win["wildcard"]["gws"]
    fh_weeks = [opt.freehit_gw] if opt.freehit_gw else win["freehit"]["gws"]
    fast = replace(opt, time_limit=search_time, gap=search_gap, threads=1)
    done: dict[tuple, Plan] = {}
    notes: list[str] = []

    def run(combos: list[tuple], base: SolverOptions, stage: str, warm: bool = False) -> None:
        combos = [c for c in dict.fromkeys(combos) if warm or c not in done]
        if not combos:
            return
        log(f"{stage}: {len(combos)} solve(s), up to {base.time_limit}s each")
        jobs = []
        for wc, fh in combos:
            prev = done.get((wc, fh)) if warm else None
            o = replace(base, wildcard_gw=wc, freehit_gw=fh, pool=prev.pool if prev else base.pool)
            jobs.append((xp, players, state, horizon, o, prev.values if prev else None))
        for c, res in zip(combos, solve_many(jobs, workers)):
            if isinstance(res, Exception):
                notes.append(f"scenario wildcard={c[0]} free hit={c[1]} failed: {res}")
            elif c not in done or res.objective > done[c].objective:
                done[c] = res

    forced_wc, forced_fh = opt.wildcard_gw, opt.freehit_gw
    singles = [(forced_wc, forced_fh)]
    if not forced_wc:
        singles += [(w, forced_fh) for w in wc_weeks if w != forced_fh]
    if not forced_fh:
        singles += [(forced_wc, f) for f in fh_weeks if f != forced_wc]
    run(singles, fast, "chip search 1/3 (each chip on its own)")

    if not forced_wc and not forced_fh and wc_weeks and fh_weeks:
        rank = lambda keys: sorted((k for k in keys if k in done), key=lambda k: -done[k].objective)  # noqa: E731
        best_wc = [k[0] for k in rank([(w, None) for w in wc_weeks])][:top]
        best_fh = [k[1] for k in rank([(None, f) for f in fh_weeks])][:top]
        run([(w, f) for w in best_wc for f in best_fh if w != f], fast, "chip search 2/3 (combinations)")

    if not done:
        raise RuntimeError("every chip scenario failed: " + "; ".join(notes))
    if polish_time and polish_time > 0:
        leaders = sorted(done, key=lambda k: -done[k].objective)[:top]
        run(leaders, replace(opt, time_limit=polish_time, threads=1), "chip search 3/3 (refining the leaders)",
            warm=True)

    plans = sorted(done.values(), key=lambda p: -p.objective)
    return ChipSearch(plans, win, notes)


# ----------------------------------------------------------------------------- robustness
def perturb(xp: pd.DataFrame, players: pd.DataFrame, rng: np.random.Generator, player_sd: float = 0.13,
            week_sd: float = 0.10, team_sd: float = 0.07) -> pd.DataFrame:
    """A plausible alternative set of projections: each player is persistently better or worse
    than projected (form, role), each player-week wobbles, and each team-week shifts together
    (a fixture turning out easier or harder than the odds said)."""
    n, gws = len(xp), list(xp.columns)
    ln = lambda sd, size: rng.normal(-sd * sd / 2, sd, size)  # noqa: E731  (mean-one multipliers)
    f = ln(player_sd, (n, 1)) + ln(week_sd, (n, len(gws)))
    teams = players["team_id"].reindex(xp.index)
    codes, inv = np.unique(teams.fillna(-1).to_numpy(), return_inverse=True)
    f = f + ln(team_sd, (len(codes), len(gws)))[inv]
    return xp * np.exp(f)


@dataclass
class Robustness:
    n: int
    sold: pd.Series                 # share of re-solves selling each player this gameweek
    bought: pd.Series
    squad: pd.Series                # share of re-solves with each player in this gameweek's squad
    captain: pd.Series
    n_transfers: pd.Series          # distribution of the number of transfers this gameweek
    chips: pd.Series                # share of re-solves playing each chip this gameweek
    base_kept: float                # share of re-solves making exactly the recommended transfers


def robustness(xp: pd.DataFrame, players: pd.DataFrame, state: TeamState, plan: Plan,
               horizon: int | None = None, opt: SolverOptions | None = None, n: int = 20,
               workers: int | None = None, time_limit: int = 40, seed: int = 7,
               log=lambda msg: None) -> Robustness:
    """Re-solve `n` times with perturbed projections, keeping the plan's wildcard / free-hit weeks.

    Each re-solve starts from the recommended plan and keeps searching for something better under
    its own numbers until the time limit, with the normal (tight) optimality tolerance: a loose
    tolerance would let the solver accept the starting plan unexamined and report false agreement."""
    opt = replace(opt or SolverOptions(), wildcard_gw=plan.wildcard_gw, freehit_gw=plan.freehit_gw,
                  pool=plan.pool, time_limit=time_limit, threads=1)
    rng = np.random.default_rng(seed)
    log(f"confidence check: {n} re-solves with noisy projections, up to {time_limit}s each")
    jobs = [(perturb(xp, players, rng), players, state, horizon, opt, plan.values) for _ in range(n)]
    out = solve_many(jobs, workers)
    res = [r for r in out if isinstance(r, Plan)]
    if not res:
        raise RuntimeError(f"all robustness re-solves failed: {out[0]!r}")
    first = [r.weeks[0] for r in res]
    share = lambda items: (pd.Series([i for it in items for i in it], dtype="object")  # noqa: E731
                           .value_counts() / len(res)).sort_values(ascending=False)
    base = plan.weeks[0]
    moves = lambda w: (set(w.squad) if w.chip == "freehit" else (frozenset(w.transfers_in), frozenset(w.transfers_out)))  # noqa: E731
    same = sum(moves(w) == moves(base) for w in first) / len(res)
    return Robustness(
        n=len(res),
        sold=share([w.transfers_out for w in first]),
        bought=share([w.transfers_in for w in first]),
        squad=share([w.squad for w in first]),
        captain=share([[w.captain] for w in first]),
        n_transfers=share([[len(w.transfers_in)] for w in first]).sort_index(),
        chips=share([[w.chip or "none"] for w in first]),
        base_kept=same,
    )
