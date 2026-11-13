"""Multi-gameweek squad optimiser (mixed-integer linear programme, solved with HiGHS).

Decides, for every gameweek in the horizon at once: transfers (with banked free transfers and
-4 hits), starting XI, bench order, captain / vice, and when to play each chip. Planning several
weeks together is what stops it chasing one good fixture into a bad run.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass, field, replace

import pandas as pd
import pulp

from .my_team import MAX_FT, TeamState

SQUAD = {1: 2, 2: 5, 3: 5, 4: 3}
MIN_PLAY = {1: 1, 2: 3, 3: 2, 4: 1}
MAX_PLAY = {1: 1, 2: 5, 3: 5, 4: 3}
CHIPS = ("wildcard", "freehit", "bboost", "3xc")


@dataclass
class SolverOptions:
    # decay and ft_value were chosen on a replay of 124 gameweeks (scripts/tune_optimiser.py). The
    # differences between settings were within noise; these did best among the multi-week ones.
    decay: float = 0.75            # weight of gameweek t+k is decay**k: far-off projections are shakier
    hit_cost: float = 4.0
    bench_gk: float = 0.03         # chance a bench player's points end up counting
    bench_out: tuple = (0.25, 0.08, 0.02)
    vice_weight: float = 0.05
    ft_value: float = 0.5          # value of each free transfer still banked after the horizon
    transfer_nudge: float = 0.05   # tiny cost per transfer so moves must be worth making
    chips: list | None = None      # [{"name", "gws"}] to override what the team has; [] = none
    use_chips: bool = True
    wildcard_gw: int | None = None  # wildcard / free hit weeks are fixed inputs (searched from outside,
    freehit_gw: int | None = None   # see optimise): as free variables they make the model intractable
    # Value of keeping a chip that outlives the plan. Free hit, bench boost and triple captain: the
    # points the chip must add to be worth playing now. Wildcard: points per (discounted) remaining
    # week, because its benefit is spread over the weeks after it is played.
    chip_reserve: dict = field(default_factory=lambda: {"wildcard": 1.7, "freehit": 7.0, "bboost": 8.0, "3xc": 6.0})
    max_hits_per_gw: int = 2
    max_transfers_gw1: int | None = None
    no_transfers: bool = False
    banned: tuple = ()
    locked: tuple = ()
    per_pos_keep: dict = field(default_factory=lambda: {1: 12, 2: 36, 3: 44, 4: 22})
    pool: tuple | None = None      # fix the candidate players (keeps the model identical between re-solves)
    time_limit: int = 180
    gap: float = 0.002
    threads: int = 0
    verbose: bool = False


@dataclass
class GWPlan:
    gw: int
    chip: str | None
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    free_transfers: int
    squad: list[int]
    lineup: list[int]
    bench: list[int]               # GK first, then outfield in substitution order
    captain: int
    vice: int
    bank: int
    xp: float                      # expected points this week, net of hits


@dataclass
class Plan:
    weeks: list[GWPlan]
    objective: float
    status: str
    total_xp: float
    candidates: int
    seconds: float = 0.0
    gap: float | None = None       # relative gap to the best possible objective when the solver stopped
    wildcard_gw: int | None = None
    freehit_gw: int | None = None
    pool: tuple = ()               # candidate players the model chose from
    fh_gain: float = 0.0           # points the free hit adds over the regular squad in its week
    values: dict = field(default_factory=dict, repr=False)   # raw non-zero solution, to warm-start a re-solve

    @property
    def chips(self) -> dict[str, int]:
        return {w.chip: w.gw for w in self.weeks if w.chip}


class _HiGHS(pulp.HiGHS):
    """HiGHS with an optional starting solution ({variable name: value}; missing names are zero)."""
    start: dict | None = None

    def callSolver(self, lp):
        if self.start:
            import highspy
            sol = highspy.HighsSolution()
            sol.col_value = [float(self.start.get(v.name, 0.0)) for v in lp.variables()]
            lp.solverModel.setSolution(sol)
        lp.solverModel.run()


@contextlib.contextmanager
def _quiet(enabled: bool = True):
    """Silence the solver's banner, which is written straight to the process's stdout."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 1)
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def best_eleven(ids: list[int], xp: dict, pos: dict) -> float:
    """Expected points of the best legal eleven (captain doubled) from a 15-man squad."""
    by = {k: sorted((p for p in ids if pos[p] == k), key=lambda p: -xp[p]) for k in SQUAD}
    eleven = [p for k in SQUAD for p in by[k][:MIN_PLAY[k]]]
    rest = sorted((p for k in (2, 3, 4) for p in by[k][MIN_PLAY[k]:]), key=lambda p: -xp[p])
    eleven += rest[: 11 - len(eleven)]
    return sum(xp[p] for p in eleven) + max((xp[p] for p in eleven), default=0.0)


def chip_windows(avail: list, gws: list[int]) -> dict:
    """For each chip: the gameweeks inside the horizon where its current copy can be played,
    and whether that copy expires before the horizon ends (so holding it back is pointless)."""
    out = {}
    for c in CHIPS:
        copies = [a for a in avail if a["name"] == c and set(a["gws"]) & set(gws)]
        if not copies:
            out[c] = {"gws": [], "expires": False}
            continue
        first = min(copies, key=lambda a: min(a["gws"]))   # the copy whose window is open now
        out[c] = {"gws": sorted(set(first["gws"]) & set(gws)), "expires": max(first["gws"]) <= gws[-1]}
    return out


def candidate_pool(xp: pd.DataFrame, players: pd.DataFrame, state: TeamState, opt: SolverOptions,
                   gws: list[int]) -> list[int]:
    """Players worth considering: best by projected points per position (overall and per week),
    best-value cheap options, and whoever is already owned."""
    w = pd.Series({g: opt.decay ** i for i, g in enumerate(gws)})
    score = (xp[gws] * w).sum(axis=1)
    keep: set[int] = set(state.squad) | set(opt.locked)
    for pos, n in opt.per_pos_keep.items():
        ids = players.index[players["pos"] == pos]
        s = score.reindex(ids).dropna()
        keep |= set(s.nlargest(n).index)
        value = (s / players.loc[s.index, "now_cost"]).nlargest(max(4, n // 4))
        keep |= set(value.index)
        cheap = players.loc[ids].sort_values(["now_cost"]).head(60)
        keep |= set(score.reindex(cheap.index).nlargest(3).index)       # budget enablers
        for g in gws[:4]:                                                # one-week punts (free hit)
            keep |= set(xp.loc[ids, g].nlargest(max(3, n // 5)).index)
    keep -= set(opt.banned) - set(state.squad)
    return sorted(p for p in keep if p in xp.index and p in players.index)


def solve(xp: pd.DataFrame, players: pd.DataFrame, state: TeamState,
          horizon: int | None = None, opt: SolverOptions | None = None, warm: dict | None = None) -> Plan:
    """xp: element id x gameweek expected points. players: index element id; columns pos, team_id, now_cost.
    `warm`: `Plan.values` of an earlier solve of the same model shape, used as the starting solution."""
    opt = opt or SolverOptions()
    gws = [g for g in sorted(xp.columns) if g >= state.next_gw]
    if horizon:
        gws = gws[:horizon]
    T = len(gws)
    pool = sorted(set(opt.pool) | set(state.squad)) if opt.pool else candidate_pool(xp, players, state, opt, gws)
    pos = players["pos"].to_dict()
    team = players["team_id"].to_dict()
    cost = players["now_cost"].astype(int).to_dict()
    sellv = {p: (state.sell_price[p] if p in state.sell_price else cost[p]) for p in pool}
    X = {(p, t): float(xp.at[p, g]) for p in pool for t, g in enumerate(gws)}
    disc = [opt.decay ** t for t in range(T)]
    outfield = [p for p in pool if pos[p] != 1]
    keepers = [p for p in pool if pos[p] == 1]

    avail = opt.chips if opt.chips is not None else (state.chips if opt.use_chips else [])
    info = chip_windows(avail, gws)
    chip_gws = {c: [gws.index(g) for g in info[c]["gws"]] for c in ("bboost", "3xc")}
    wc_t = gws.index(opt.wildcard_gw) if opt.wildcard_gw in gws else None
    fh_t = gws.index(opt.freehit_gw) if opt.freehit_gw in gws else None
    if wc_t is not None and wc_t == fh_t:
        raise ValueError("wildcard and free hit cannot be played in the same gameweek")
    for c, g in (("wildcard", opt.wildcard_gw), ("freehit", opt.freehit_gw)):
        if g is not None and g not in info[c]["gws"]:
            raise ValueError(f"{c} cannot be played in gameweek {g} (available: {info[c]['gws'] or 'never'})")
    for c in ("bboost", "3xc"):  # one chip per gameweek
        chip_gws[c] = [t for t in chip_gws[c] if t not in (wc_t, fh_t)]
    fh_weeks = [fh_t] if fh_t is not None else []

    m = pulp.LpProblem("fpl", pulp.LpMaximize)
    B = lambda name, keys: pulp.LpVariable.dicts(name, keys, cat="Binary")  # noqa: E731
    PT = [(p, t) for p in pool for t in range(T)]
    x, lu, cap, vc = B("x", PT), B("l", PT), B("c", PT), B("v", PT)
    tin, tout = B("in", PT), B("out", PT)
    bench = B("b", [(p, t, k) for p in outfield for t in range(T) for k in range(3)])
    chip = {c: {t: pulp.LpVariable(f"chip_{c}_{t}", cat="Binary") for t in chip_gws[c]}
            for c in ("bboost", "3xc")}
    chip["wildcard"] = {wc_t: 1} if wc_t is not None else {}
    chip["freehit"] = {fh_t: 1} if fh_t is not None else {}
    used = lambda c, t: chip[c].get(t, 0)  # noqa: E731
    xf = B("xf", [(p, t) for p in pool for t in fh_weeks])       # free-hit squad
    eff = {(p, t): (xf[p, t] if t == fh_t else x[p, t]) for p, t in PT}   # squad actually fielded
    bbp = B("bb", [(p, t) for p in pool for t in chip_gws["bboost"]])
    tcp = B("tc", [(p, t) for p in pool for t in chip_gws["3xc"]])
    ft = {t: pulp.LpVariable(f"ft_{t}", 0, MAX_FT, cat="Integer") for t in range(T + 1)}
    hits = {t: pulp.LpVariable(f"hits_{t}", 0, opt.max_hits_per_gw, cat="Integer") for t in range(T)}
    bank = {t: pulp.LpVariable(f"bank_{t}", 0) for t in range(T)}

    owned0 = {p: 1 if p in state.squad else 0 for p in pool}
    missing = [p for p in state.squad if p not in pool]
    if missing:
        raise ValueError(f"squad players without projections: {missing}")

    obj = []
    for t in range(T):
        prev = (lambda p: owned0[p]) if t == 0 else (lambda p, t=t: x[p, t - 1])
        n_in = pulp.lpSum(tin[p, t] for p in pool)
        wc, fh = used("wildcard", t), used("freehit", t)
        for p in pool:
            m += x[p, t] == prev(p) + tin[p, t] - tout[p, t]
            m += tin[p, t] + tout[p, t] <= 1
        m += pulp.lpSum(x[p, t] for p in pool) == 15
        for ps, n in SQUAD.items():
            m += pulp.lpSum(x[p, t] for p in pool if pos[p] == ps) == n
        for tm in set(team[p] for p in pool):
            m += pulp.lpSum(x[p, t] for p in pool if team[p] == tm) <= 3
        prev_bank = state.bank if t == 0 else bank[t - 1]
        m += bank[t] == prev_bank + pulp.lpSum(sellv[p] * tout[p, t] for p in pool) \
            - pulp.lpSum(cost[p] * tin[p, t] for p in pool)

        # free transfers: used ones are free, the rest cost a hit; unused ones roll over (max five).
        # A wildcard or free hit week leaves the banked count untouched.
        if t == wc_t:
            m += hits[t] == 0
            m += ft[t + 1] == ft[t] if state.free_transfers > 0 or t > 0 else ft[t + 1] == 1
        elif t == fh_t:
            m += hits[t] == 0
            m += n_in == 0
            m += ft[t + 1] == ft[t] if state.free_transfers > 0 or t > 0 else ft[t + 1] == 1
            m += pulp.lpSum(xf[p, t] for p in pool) == 15
            for ps, n in SQUAD.items():
                m += pulp.lpSum(xf[p, t] for p in pool if pos[p] == ps) == n
            for tm in set(team[p] for p in pool):
                m += pulp.lpSum(xf[p, t] for p in pool if team[p] == tm) <= 3
            m += pulp.lpSum(cost[p] * xf[p, t] for p in pool) <= \
                pulp.lpSum(sellv[p] * x[p, t] for p in pool) + bank[t]
        else:
            m += hits[t] >= n_in - ft[t]
            m += ft[t + 1] <= ft[t] - n_in + hits[t] + 1
            m += ft[t + 1] >= 1
        m += pulp.lpSum(used(c, t) for c in ("bboost", "3xc")) <= 1

        # line-up, bench order, armband
        m += pulp.lpSum(lu[p, t] for p in pool) == 11
        for ps in SQUAD:
            n = pulp.lpSum(lu[p, t] for p in pool if pos[p] == ps)
            m += n >= MIN_PLAY[ps]
            m += n <= MAX_PLAY[ps]
        for p in pool:
            m += lu[p, t] <= eff[p, t]
            m += cap[p, t] + vc[p, t] <= lu[p, t]
        for p in outfield:
            m += lu[p, t] + pulp.lpSum(bench[p, t, k] for k in range(3)) == eff[p, t]
        for k in range(3):
            m += pulp.lpSum(bench[p, t, k] for p in outfield) == 1
        m += pulp.lpSum(cap[p, t] for p in pool) == 1
        m += pulp.lpSum(vc[p, t] for p in pool) == 1

        week = pulp.lpSum(X[p, t] * (lu[p, t] + cap[p, t] + opt.vice_weight * vc[p, t]) for p in pool)
        week += pulp.lpSum(opt.bench_out[k] * X[p, t] * bench[p, t, k] for p in outfield for k in range(3))
        week += pulp.lpSum(opt.bench_gk * X[p, t] * (eff[p, t] - lu[p, t]) for p in keepers)
        obj.append(disc[t] * (week - opt.hit_cost * hits[t]) - opt.transfer_nudge * n_in)
        # chips are valued undiscounted: they expire, so "later" should not look artificially worse
        if t in chip_gws["bboost"]:
            for p in pool:
                m += bbp[p, t] <= eff[p, t] - lu[p, t]
            # aggregated link (four bench places): far tighter than one "bbp <= chip" row per player,
            # which lets the relaxation spread a boost over dozens of fractional bench players
            m += pulp.lpSum(bbp[p, t] for p in pool) <= 4 * chip["bboost"][t]
            obj.append(pulp.lpSum(0.9 * X[p, t] * bbp[p, t] for p in pool))
        if t in chip_gws["3xc"]:
            for p in pool:
                m += tcp[p, t] <= cap[p, t]
            m += pulp.lpSum(tcp[p, t] for p in pool) == chip["3xc"][t]
            obj.append(pulp.lpSum(X[p, t] * tcp[p, t] for p in pool))

    m += ft[0] == state.free_transfers
    for c in ("bboost", "3xc"):
        if chip[c]:
            m += pulp.lpSum(chip[c].values()) <= 1
    # A chip that outlives the horizon keeps some value if we do not burn it now.
    for c in CHIPS:
        if chip[c] and not info[c]["expires"]:
            scale = sum(disc[wc_t:]) if c == "wildcard" else 1.0
            obj.append(-opt.chip_reserve.get(c, 0.0) * scale * pulp.lpSum(chip[c].values()))
    if opt.max_transfers_gw1 is not None:
        if wc_t != 0:
            m += pulp.lpSum(tin[p, 0] for p in pool) <= opt.max_transfers_gw1
    if opt.no_transfers:
        for p, t in PT:
            m += tin[p, t] == 0
    for p in opt.banned:
        if p in pool:
            for t in range(T):
                m += tin[p, t] == 0
                if t == fh_t:
                    m += xf[p, t] == 0
    for p in opt.locked:
        if p in pool:
            for t in range(T):
                m += x[p, t] == 1
    obj.append(opt.ft_value * (ft[T] - 1))
    m += pulp.lpSum(obj)

    t0 = time.time()
    solver = _HiGHS(msg=opt.verbose, timeLimit=opt.time_limit, gapRel=opt.gap,
                    threads=opt.threads or None)
    solver.start = warm
    with _quiet(not opt.verbose):
        m.solve(solver)
    secs = time.time() - t0
    status = pulp.LpStatus[m.status]
    try:
        gap = float(m.solverModel.getInfo().mip_gap)
        gap = gap if gap == gap and gap < 1e6 else None
    except Exception:  # the gap is only diagnostic
        gap = None
    val = lambda v: (v.value() or 0.0) if hasattr(v, "value") else float(v)  # noqa: E731
    if x[pool[0], 0].value() is None or round(sum(val(lu[p, 0]) for p in pool)) != 11:
        raise RuntimeError(f"solver returned no solution (status: {status}, {secs:.0f}s)")

    weeks, total = [], 0.0
    ft_now = state.free_transfers
    for t, g in enumerate(gws):
        on = lambda d, p: val(d[p, t]) > 0.5  # noqa: E731
        squad = [p for p in pool if on(eff, p)]
        lineup = sorted([p for p in squad if on(lu, p)], key=lambda p: (pos[p], -X[p, t]))
        bench_gk = [p for p in squad if pos[p] == 1 and not on(lu, p)]
        bench_out = [p for k in range(3) for p in outfield if val(bench[p, t, k]) > 0.5]
        c = next(p for p in pool if on(cap, p))
        v = next(p for p in pool if on(vc, p))
        played = next((cn for cn in CHIPS if val(used(cn, t)) > 0.5), None)
        h = int(round(val(hits[t])))
        pts = sum(X[p, t] for p in lineup) + X[c, t] * (2 if played == "3xc" else 1)
        if played == "bboost":
            pts += sum(X[p, t] for p in bench_gk + bench_out)
        pts -= opt.hit_cost * h
        total += pts
        ins = [p for p in pool if on(tin, p)]
        weeks.append(GWPlan(g, played, ins, [p for p in pool if on(tout, p)], h, ft_now, squad, lineup,
                            bench_gk + bench_out, c, v, int(round(val(bank[t]))), round(pts, 2)))
        if t in (wc_t, fh_t):       # the counter is the model's own; replaying the rules gives the true figure
            ft_now = ft_now if (ft_now > 0 or t > 0) else 1
        else:
            ft_now = min(MAX_FT, max(ft_now - len(ins), 0) + 1)
    values = {v.name: v.value() for v in m.variables() if v.value()}
    objective = float(pulp.value(m.objective))
    fh_gain = 0.0
    if fh_t is not None:
        # Like the other one-off chips, the free hit is valued undiscounted: what it adds is the
        # gap between its one-week squad and the squad it stands in for. Without this a free hit
        # looks better the sooner it is played, purely because near weeks weigh more.
        own = [p for p in pool if val(x[p, fh_t]) > 0.5]
        w = weeks[fh_t]
        fielded = sum(X[p, fh_t] for p in w.lineup) + X[w.captain, fh_t]
        fh_gain = fielded - best_eleven(own, {p: X[p, fh_t] for p in own}, pos)
        objective += (1 - disc[fh_t]) * fh_gain
    return Plan(weeks, objective, status, round(total, 2), len(pool), round(secs, 1), gap=gap,
                wildcard_gw=opt.wildcard_gw, freehit_gw=opt.freehit_gw, pool=tuple(pool),
                fh_gain=round(fh_gain, 2), values=values)


def hold_value(xp: pd.DataFrame, players: pd.DataFrame, state: TeamState, horizon: int | None = None,
               opt: SolverOptions | None = None) -> Plan:
    """Baseline: same optimiser, but no transfers and no chips (just best XI and captain each week)."""
    o = replace(opt or SolverOptions(), use_chips=False, chips=[], max_hits_per_gw=0, ft_value=0.0,
                no_transfers=True)
    return solve(xp, players, state, horizon, o)
