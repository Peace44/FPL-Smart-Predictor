"""One call that goes from live data to a recommendation for the next deadline."""
from __future__ import annotations

import json
import pickle
import unicodedata
from dataclasses import dataclass, field, replace

import pandas as pd

from . import fpl_api, my_team, optimise, xp as xp_mod
from .config import PROCESSED, REPORTS, team_id
from .my_team import TeamState
from .solver import Plan, SolverOptions, hold_value, solve

BASE_HORIZON, TAIL, MAX_HORIZON = 9, 3, 12
FINISH_TIME = 150      # seconds for the final solve of the chosen plan


@dataclass
class Recommendation:
    projection: xp_mod.Projection
    state: TeamState
    plan: Plan                         # what to do
    hold: Plan                         # same squad, no transfers, no chips
    horizon: int
    search: optimise.ChipSearch | None = None
    robust: optimise.Robustness | None = None
    deadline: pd.Timestamp | None = None
    meta: dict = field(default_factory=dict)


def auto_horizon(state: TeamState, last_gw: int = 38) -> int:
    """Nine gameweeks, stretched (to twelve at most) so the plan runs a few weeks past the expiry
    of any wildcard / free hit: otherwise playing them late looks worse than it is, because the
    weeks that benefit fall outside the plan."""
    ends = [max(c["gws"]) for c in state.chips
            if c["name"] in ("wildcard", "freehit") and min(c["gws"]) <= state.next_gw]
    need = (min(ends) - state.next_gw + 1 + TAIL) if ends else BASE_HORIZON
    return int(max(1, min(MAX_HORIZON, max(BASE_HORIZON, need), last_gw - state.next_gw + 1)))


_PLAIN = str.maketrans({"ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "Æ": "AE", "đ": "d", "Đ": "D", "ł": "l",
                        "Ł": "L", "ı": "i", "þ": "th", "ð": "d", "œ": "oe"})


def _fold(s: str) -> str:
    """Lower-case and strip accents, so "odegaard" finds Ødegaard and "gross" finds Groß."""
    s = unicodedata.normalize("NFKD", str(s).translate(_PLAIN))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def resolve_players(names, players: pd.DataFrame) -> list[int]:
    """Player names (or ids) -> element ids. A name must match exactly one player; add the team
    in brackets to disambiguate, e.g. "Martinez (AVL)"."""
    out = []
    for raw in names or []:
        raw = str(raw).strip()
        if raw.isdigit() and int(raw) in players.index:
            out.append(int(raw))
            continue
        name, team = raw, None
        if raw.endswith(")") and "(" in raw:
            name, team = raw[: raw.rindex("(")].strip(), raw[raw.rindex("(") + 1: -1].strip().upper()
        hit = players[players["name"].map(_fold) == _fold(name)]
        if team:
            hit = hit[hit["team"] == team]
        if len(hit) == 0:
            near = players[players["name"].map(_fold).str.contains(_fold(name), regex=False)]
            hint = ", ".join(f"{r['name']} ({r['team']})" for _, r in near.head(6).iterrows())
            raise ValueError(f"no player called '{raw}'" + (f"; did you mean: {hint}?" if hint else ""))
        if len(hit) > 1:
            opts = ", ".join(f"{r['name']} ({r['team']})" for _, r in hit.iterrows())
            raise ValueError(f"'{raw}' matches several players: {opts}. Add the team in brackets.")
        out.append(int(hit.index[0]))
    return out


def make(horizon: int | None = None, ft: int | None = None, use_chips: bool = True,
         wildcard: int | None = None, freehit: int | None = None, quick: bool = False, sims: int = 20,
         refresh: bool = True, banned=(), locked=(), max_hits: int = 2, workers: int | None = None,
         entry_id: int | None = None, log=print) -> Recommendation:
    entry_id = entry_id or team_id()
    bs = fpl_api.bootstrap(0.0 if refresh else 6.0)
    state = my_team.load(entry_id, bs, free_transfers=ft)
    H = horizon or auto_horizon(state)
    log(f"projecting GW{state.next_gw}-{min(38, state.next_gw + H - 1)} ...")
    P = xp_mod.project_live(horizon=H, refresh=refresh)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "projection.pkl", "wb") as f:
        pickle.dump(P, f)

    opt = SolverOptions(use_chips=use_chips, max_hits_per_gw=max_hits,
                        banned=tuple(resolve_players(banned, P.players)),
                        locked=tuple(resolve_players(locked, P.players)))
    hold = hold_value(P.xp, P.players, state, H, opt)
    windows = optimise.chip_windows(optimise.available_chips(state, opt), optimise.horizon_gws(P.xp, state, H))
    searchable = use_chips and (windows["wildcard"]["gws"] or windows["freehit"]["gws"])
    chosen = replace(opt, wildcard_gw=wildcard, freehit_gw=freehit)
    search = None
    if quick or not searchable:
        log("solving ...")
        plan = solve(P.xp, P.players, state, H, replace(chosen, time_limit=120))
    else:
        search = optimise.chip_search(P.xp, P.players, state, H, chosen, workers=workers, log=log)
        plan = search.choose()
        if plan.gap is None or plan.gap > 1.5 * opt.gap:
            # the search shares the machine between many solves; finish the chosen one on its own
            log("finishing the chosen plan ...")
            o = replace(opt, wildcard_gw=plan.wildcard_gw, freehit_gw=plan.freehit_gw, pool=plan.pool,
                        time_limit=FINISH_TIME)
            better = solve(P.xp, P.players, state, H, o, warm=plan.values)
            if better.objective >= plan.objective:
                search.scenarios[next(i for i, q in enumerate(search.scenarios) if q is plan)] = better
                search.scenarios.sort(key=lambda q: -q.objective)
                plan = better
    robust = None
    if sims and sims > 0:
        robust = optimise.robustness(P.xp, P.players, state, plan, H, opt, n=sims, workers=workers, log=log)

    ev = next((e for e in bs["events"] if e["id"] == state.next_gw), {})
    ent = fpl_api.entry(entry_id, 1.0)
    meta = {"team_name": ent.get("name"), "manager": f"{ent.get('player_first_name', '')} {ent.get('player_last_name', '')}".strip(),
            "total_players": bs.get("total_players"), "generated": str(pd.Timestamp.now(tz="UTC"))}
    deadline = pd.Timestamp(ev["deadline_time"]) if ev.get("deadline_time") else None
    return Recommendation(P, state, plan, hold, H, search, robust, deadline, meta)


def save(rec: Recommendation, text: str) -> dict:
    """Write the report and a small machine-readable record of what was recommended."""
    gw = rec.state.next_gw
    d = REPORTS / f"gw{gw:02d}"
    d.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d-%H%M")
    md = d / f"plan-{stamp}.md"
    md.write_text(text)
    (REPORTS / "latest.md").write_text(text)
    w = rec.plan.weeks[0]
    name = rec.projection.players["name"].to_dict()
    record = {
        "gw": gw, "generated": rec.meta.get("generated"), "horizon": rec.horizon, "chip": w.chip,
        "transfers_out": w.transfers_out, "transfers_in": w.transfers_in, "hits": w.hits,
        "lineup": w.lineup, "bench": w.bench, "captain": w.captain, "vice": w.vice,
        "xp": w.xp, "hold_xp": rec.hold.weeks[0].xp, "plan_total": rec.plan.total_xp,
        "hold_total": rec.hold.total_xp, "chips": rec.plan.chips,
        "names": {str(p): name.get(p) for p in set(w.lineup + w.bench + w.transfers_out)},
        "squad_before": rec.state.squad,
    }
    js = d / f"plan-{stamp}.json"
    js.write_text(json.dumps(record, indent=1))
    (d / "latest.json").write_text(json.dumps(record, indent=1))
    return {"markdown": md, "json": js, "latest": REPORTS / "latest.md"}
