"""The gameweek review: how the team did against its projection, where it stands on the way to
first place in the world, and what limits it most. Written to `reports/gwNN_review.md`.

Every number comes from FPL's public data for the gameweek, the projection the engine saved just
before the deadline (`reports/gwNN/projection-*.parquet`) and its plan record. The review ends
with a machine-readable copy of its findings (an HTML comment) that `fpl improve` reads.

Where the points went is measured against the top 50 managers in the world as they stood before
the gameweek (their squads, captains, chips and hits are public), because that is the pace first
place needs. The difference splits exactly into five parts:

    squad       the best eleven each team could have fielded from its 15 (who you own)
    selection   that best eleven minus the eleven that played (who you start, bench order)
    captaincy   the armband's extra points
    chips       Bench Boost / Triple Captain extra points
    hits        points spent on transfers

The part with the largest average gap over recent engine-managed gameweeks is the limiting factor.
Each part's average carries a standard error, and the factor counts as settled only when it leads the
runner-up by more than chance (`certainty`). Until then code changes are aimed by the model's errors
pooled over every review (`evidence.py`), which each review stores.
"""
from __future__ import annotations

import json
import math
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import evidence, fpl_api
from .config import CHIP_NAMES, POS, REPORTS, team_id
from .scoring import MIN_FORMATION, score_gw

VERSION = 1
FACTORS = ("squad", "selection", "captaincy", "chips", "hits")
WINDOW = 6                      # gameweeks the limiting factor is averaged over
ELITE_N = 50
OVERALL_LEAGUE = 314
THRESHOLD_RANKS = (1, 100, 1_000, 10_000, 100_000, 1_000_000)
RANK_BANDS = (10_000_000, 5_000_000, 2_000_000, 1_000_000, 500_000, 250_000, 100_000, 50_000, 10_000,
              1_000, 100, 10, 1)
COMPONENTS = ["app", "goals", "assists", "cs", "gc", "saves", "pensave", "defcon", "bonus", "cards"]
COMP_LABEL = {"app": "Appearance", "goals": "Goals", "assists": "Assists", "cs": "Clean sheets",
              "gc": "Goals conceded", "saves": "Saves", "pensave": "Penalty saves",
              "defcon": "Defensive contributions", "bonus": "Bonus", "cards": "Cards, own goals, missed pens",
              "other": "Other"}
IDENT = {"minutes": "app", "goals_scored": "goals", "assists": "assists", "clean_sheets": "cs",
         "goals_conceded": "gc", "saves": "saves", "penalties_saved": "pensave", "defensive_contribution": "defcon",
         "bonus": "bonus", "yellow_cards": "cards", "red_cards": "cards", "own_goals": "cards",
         "penalties_missed": "cards"}
FACTOR_TEXT = {
    "squad": "who you own: the best eleven the squad could field",
    "selection": "who starts: points left on the bench beyond what the best eleven needed",
    "captaincy": "the armband's extra points",
    "chips": "Bench Boost and Triple Captain points",
    "hits": "points spent on transfers",
    "operations": "points lost to the autopilot itself (unplanned hits, a missed deadline)",
}
LEVERS = {
    "squad": "points model (player and team ratings), then the transfer planner",
    "selection": "availability (injury flags, minutes), then the points model's ranking within the squad",
    "captaincy": "the points model at the top end (who hauls), then captaincy that weighs ownership",
    "chips": "chip timing (chip reserves; not covered by the replays)",
    "hits": "the transfer planner (hit cost, free-transfer value, horizon)",
    "operations": "the autopilot code",
}
NOTES_HEADING = "## Your notes"
NOTES_HINT = "_Anything written below this line is kept when the review is rewritten and is read by `fpl improve`'s skill._"
DATA_OPEN, DATA_CLOSE = "<!-- fpl-review-data", "-->"


def review_path(gw: int) -> Path:
    return REPORTS / f"gw{gw:02d}_review.md"


# ----------------------------------------------------------------------------- scoring helpers
def best_xi(squad: list[int], pts: dict, pos: dict) -> tuple[float, list[int]]:
    """The highest-scoring legal eleven from a squad, in hindsight."""
    by = {k: sorted([p for p in squad if pos.get(p) == k], key=lambda p: -float(pts.get(p, 0))) for k in POS}
    best: tuple[float, list[int]] | None = None
    for d in range(MIN_FORMATION[2], 6):
        for m in range(MIN_FORMATION[3], 6):
            f = 10 - d - m
            if not MIN_FORMATION[4] <= f <= 3 or len(by[2]) < d or len(by[3]) < m or len(by[4]) < f or not by[1]:
                continue
            ids = by[1][:1] + by[2][:d] + by[3][:m] + by[4][:f]
            s = float(sum(pts.get(p, 0) for p in ids))
            if best is None or s > best[0]:
                best = (s, ids)
    return best or (0.0, [])


def team_parts(picks: list[dict], chip: str | None, hits: int, pts: dict, mins: dict, pos: dict) -> dict:
    """One manager's gameweek split into the five parts (they add up to the net points)."""
    order = sorted(picks, key=lambda p: p["position"])
    lineup = [p["element"] for p in order if p["position"] <= 11]
    bench = [p["element"] for p in order if p["position"] > 11]
    cap = next((p["element"] for p in order if p.get("is_captain")), lineup[0])
    vice = next((p["element"] for p in order if p.get("is_vice_captain")), lineup[1])
    sc = score_gw(lineup, bench, cap, vice, pts, mins, pos, chip)
    squad = lineup + bench
    bx, bx_ids = best_xi(squad, pts, pos)
    if chip == "bboost":
        xi, sel, chip_extra = bx, 0.0, float(sum(pts.get(p, 0) for p in squad)) - bx
    else:
        xi = float(sum(pts.get(p, 0) for p in sc.lineup))
        sel, chip_extra = bx - xi, 0.0
    arm = sc.captain
    cap_extra = float(pts.get(arm, 0)) if arm is not None else 0.0
    if chip == "3xc" and arm is not None:
        chip_extra = cap_extra
    pool = lineup if chip == "bboost" else sc.lineup          # the armband can only sit in the eleven
    best_cap = max((float(pts.get(p, 0)) for p in pool), default=0.0)
    return {"lineup": lineup, "bench": bench, "captain": cap, "vice": vice, "armband": arm,
            "final": sc.lineup, "subs": sc.subs, "squad": squad, "best_xi": bx, "best_xi_ids": bx_ids,
            "xi": xi, "selection": sel, "captain_extra": cap_extra, "chip_extra": chip_extra, "hits": int(hits),
            "net": xi + cap_extra + chip_extra - int(hits), "best_captain": best_cap,
            "captain_loss": max(0.0, best_cap - cap_extra), "chip": chip,
            "bench_points": float(sum(pts.get(p, 0) for p in bench))}


def deficits(ours: dict, elite: list[dict]) -> dict:
    """How far behind the elite average each part was (positive = behind). They sum to the gap in net points."""
    if not elite:
        return {}
    avg = {k: float(np.mean([e[k] for e in elite])) for k in ("best_xi", "selection", "captain_extra", "chip_extra", "hits", "net")}
    return {"squad": avg["best_xi"] - ours["best_xi"], "selection": ours["selection"] - avg["selection"],
            "captaincy": avg["captain_extra"] - ours["captain_extra"], "chips": avg["chip_extra"] - ours["chip_extra"],
            "hits": ours["hits"] - avg["hits"], "total": avg["net"] - ours["net"]}


# ----------------------------------------------------------------------------- data
def live_frame(live: dict) -> pd.DataFrame:
    """One row per player: points, minutes, expected stats and points by component."""
    rows = []
    for e in live["elements"]:
        s = e["stats"]
        r = {"pid": e["id"], "pts": s["total_points"], "minutes": s["minutes"],
             "goals_n": s["goals_scored"], "assists_n": s["assists"], "cs_n": s["clean_sheets"],
             "bonus_n": s["bonus"], "saves_n": s["saves"], "dc_n": s.get("defensive_contribution", 0),
             "xg": float(s.get("expected_goals") or 0), "xa": float(s.get("expected_assists") or 0),
             "xgc": float(s.get("expected_goals_conceded") or 0), "starts": s.get("starts", 0)}
        for c in COMPONENTS + ["other"]:
            r[f"a_{c}"] = 0.0
        for f in e.get("explain") or []:
            for st in f["stats"]:
                r["a_" + IDENT.get(st["identifier"], "other")] += st["points"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("pid")


def snapshot_for(gw: int, deadline: pd.Timestamp | None) -> pd.DataFrame | None:
    """The last projection saved before the deadline of `gw` (None if there is none)."""
    d = REPORTS / f"gw{gw:02d}"
    cands = []
    for f in list(d.glob("projection-*.parquet")) + list(d.glob("projection-*.pkl")):
        m = re.search(r"(\d{8}-\d{4})", f.name)
        if not m:
            continue
        t = pd.to_datetime(m.group(1), format="%Y%m%d-%H%M", utc=True)
        if deadline is None or t <= deadline:
            cands.append((t, f))
    if not cands:
        return None
    f = max(cands)[1]
    if f.suffix == ".parquet":
        return pd.read_parquet(f)
    from .planner import snapshot
    with open(f, "rb") as fh:
        return snapshot(pickle.load(fh))


def per_player(snap: pd.DataFrame) -> pd.DataFrame:
    """Projection summed over the player's fixtures in the gameweek."""
    sums = {c: "sum" for c in COMPONENTS + ["xp", "x_goals", "x_assists", "p_cs", "emin"] if c in snap}
    firsts = {c: "first" for c in ("name", "team", "pos", "price", "own", "status", "chance", "news", "ep_next",
                                   "opp", "home") if c in snap}
    maxes = {c: "max" for c in ("p_play", "p60", "p_play_base") if c in snap}
    g = snap.groupby("pid").agg({**sums, **firsts, **maxes})
    g["n_fix"] = snap.groupby("pid").size()
    return g


def _standings_page(page: int, max_age_h: float) -> dict:
    return fpl_api._get(f"leagues-classic/{OVERALL_LEAGUE}/standings/?page_standings={page}",
                        f"overall/page_{page}.json", max_age_h)


def thresholds(max_age_h: float = 2.0) -> dict[str, dict]:
    """Total points at ranks 1, 100, 1k, 10k, 100k and 1M of the overall league, right now."""
    out = {}
    for r in THRESHOLD_RANKS:
        try:
            res = _standings_page(math.ceil(r / 50), max_age_h)["standings"]["results"]
        except Exception:  # noqa: BLE001  (a missing threshold is not worth failing the review for)
            continue
        if res:
            row = min(res, key=lambda x: abs(x["rank_sort"] - r))
            out[str(r)] = {"rank": row["rank"], "total": row["total"], "event_total": row["event_total"]}
    return out


def elite_entries(gw: int, previous: dict | None, now_is_gw: bool, max_age_h: float = 2.0) -> tuple[list[int], str]:
    """The top 50 as they stood before `gw`: from the last review if it saved them, else from the
    standings' `last_rank` (only valid while `gw` is the latest gameweek), else the current top 50."""
    if previous and (previous.get("elite") or {}).get("ids_next") and previous.get("gw") == gw - 1:
        return list(previous["elite"]["ids_next"]), f"top {ELITE_N} after GW{gw - 1} (saved by that review)"
    rows = []
    for page in (1, 2, 3, 4):
        try:
            rows += _standings_page(page, max_age_h)["standings"]["results"]
        except Exception:  # noqa: BLE001
            break
    if not rows:
        return [], "standings unavailable"
    if now_is_gw:
        before = sorted([r for r in rows if r.get("last_rank") and r["last_rank"] <= ELITE_N],
                        key=lambda r: r["last_rank"])[:ELITE_N]   # managers tied on rank: keep 50
        if len(before) >= 0.8 * ELITE_N:
            return [r["entry"] for r in before], f"top {ELITE_N} before GW{gw} ({len(before)} found)"
    top = sorted(rows, key=lambda r: r["rank_sort"])[:ELITE_N]
    return [r["entry"] for r in top], f"current top {ELITE_N} (biased upwards: they are there partly because of this gameweek)"


def _picks(entry: int, gw: int) -> dict | None:
    try:
        return fpl_api.entry_picks(entry, gw, 24.0 * 30)
    except Exception:  # noqa: BLE001
        return None


def load_reviews() -> list[dict]:
    """The machine-readable part of every review written so far, oldest first."""
    out = []
    for f in sorted(REPORTS.glob("gw*_review.md")):
        d = parse(f.read_text())
        if d:
            out.append(d)
    return sorted(out, key=lambda d: d["gw"])


def parse(text: str) -> dict | None:
    i = text.rfind(DATA_OPEN)
    if i < 0:
        return None
    j = text.find(DATA_CLOSE, i + len(DATA_OPEN))
    try:
        return json.loads(text[i + len(DATA_OPEN): j if j > 0 else None])
    except ValueError:
        return None


def window(reviews: list[dict], gw: int, n: int = WINDOW) -> list[dict]:
    """Recent engine-managed reviews up to `gw` that could be compared with the elite."""
    ok = [r for r in reviews if r["gw"] <= gw and r.get("engine_managed") and r.get("deficits")]
    return ok[-n:]


def choose_factor(win: list[dict], operations_lost: float = 0.0) -> tuple[str | None, dict]:
    """The limiting factor: unplanned operational losses first, else the part furthest behind the elite
    on average over the window."""
    avg = {f: float(np.mean([r["deficits"][f] for r in win])) for f in FACTORS} if win else {}
    if operations_lost > 0:
        return "operations", avg
    if not avg:
        return None, avg
    return max(avg, key=avg.get), avg


# how each team's own numbers enter its deficits (squad = elite best eleven - ours, selection = ours - elite, ...)
DEFICIT_SIGN = {"squad": ("best_xi", -1), "selection": ("selection", 1), "captaincy": ("captain_extra", -1),
                "chips": ("chip_extra", -1), "hits": ("hits", 1)}


def part_noise(elite: list[dict]) -> dict | None:
    """How much one manager's week varies, part by part: the covariance of the parts across the top managers
    this week, signed like the deficits. They share a template, so for a team further from it this is a
    lower bound."""
    if len(elite) < 3:
        return None
    X = np.array([[s * e[k] for k, s in (DEFICIT_SIGN[f] for f in FACTORS)] for e in elite], float)
    C = np.cov(X, rowvar=False)
    return {f: {g: float(C[i, j]) for j, g in enumerate(FACTORS)} for i, f in enumerate(FACTORS)}


def certainty(win: list[dict], factor: str | None) -> dict:
    """How sure the window is of its limiting factor: each part's standard error, and whether the factor leads
    the runner-up by more than chance.

    The noise is estimated two ways and the larger is used: the top managers' spread in each week of the
    window (`deficit_cov`, a lower bound for your team), and, from two weeks on, the spread of your own
    weeks. The lead must clear Student's t at 95% (two-sided) with n - 1 degrees of freedom, so a single
    week never settles it: your own week-to-week spread is not known yet."""
    n = len(win)
    out = {"n": n, "se": {}, "settled": False}
    if factor == "operations":
        return out | {"settled": True}           # points the autopilot lost are counted, not estimated
    if factor is None or not n:
        return out
    D = np.array([[float(r["deficits"][f]) for f in FACTORS] for r in win])
    avg = D.mean(axis=0)
    est = []
    covs = [np.array([[r["deficit_cov"][f][g] for g in FACTORS] for f in FACTORS]) for r in win if r.get("deficit_cov")]
    if covs:
        est.append(np.mean(covs, axis=0) / n)
    if n >= 2:
        est.append(np.cov(D, rowvar=False, ddof=1) / n)
    if not est:
        return out
    out["se"] = {f: float(np.sqrt(max(M[i, i] for M in est))) for i, f in enumerate(FACTORS)}
    a = FACTORS.index(factor)
    b = max((i for i in range(len(FACTORS)) if i != a), key=lambda i: avg[i])
    lead = float(avg[a] - avg[b])
    se = float(np.sqrt(max(max(M[a, a] + M[b, b] - 2 * M[a, b], 0.0) for M in est)))
    t = lead / se if se > 0 else 0.0
    need = float(stats.t.ppf(0.975, n - 1)) if n >= 2 else None
    out |= {"runner_up": FACTORS[b], "lead": lead, "lead_se": se, "t": float(t), "needed": need,
            "settled": bool(need is not None and t >= need)}
    return out


def certainty_text(f: str, cert: dict) -> str:
    """One sentence: is the limiting factor settled, and by how much."""
    if f == "operations":
        return "**Settled:** points lost to the autopilot itself come first."
    if "lead" not in cert:
        return "**Not settled:** there is no estimate of the noise yet."
    lead = f"{f} leads {cert['runner_up']} by {cert['lead']:.1f} ± {cert['lead_se']:.1f} points a week"
    if cert["needed"] is None:
        return (f"**Not settled:** {lead}, from one gameweek. The ± is the top managers' spread this week, a lower "
                "bound for your team; your own week-to-week spread needs two gameweeks or more.")
    word = "Settled" if cert["settled"] else "Not settled"
    return (f"**{word}:** {lead} ({cert['t']:.1f} standard errors; with {cert['n']} gameweeks it takes "
            f"{cert['needed']:.1f}).")


def _rank_band(rank: int | None) -> int | None:
    if not rank:
        return None
    inside = [b for b in RANK_BANDS if rank <= b]
    return min(inside) if inside else None


def _fmt_rank(r) -> str:
    return f"{int(r):,}" if r else "?"


def _signed(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}"


# ----------------------------------------------------------------------------- the review
@dataclass
class Review:
    gw: int
    final: bool
    data: dict
    sections: list[str] = field(default_factory=list)


def build(gw: int | None = None, entry_id: int | None = None, log=lambda m: None) -> Review:
    entry_id = entry_id or team_id()
    bs = fpl_api.bootstrap(0.5)
    events = {e["id"]: e for e in bs["events"]}
    gw = gw or default_gw(bs)
    ev = events[gw]
    cur_id = next((e["id"] for e in bs["events"] if e["is_current"]), None)
    now_is_gw = cur_id == gw
    final = bool(ev.get("finished") and ev.get("data_checked"))
    deadline = pd.Timestamp(ev["deadline_time"])
    els = {e["id"]: e for e in bs["elements"]}
    pos = {i: e["element_type"] for i, e in els.items()}
    short = {t["id"]: t["short_name"] for t in bs["teams"]}
    name = {i: e["web_name"] for i, e in els.items()}
    total_players = bs.get("total_players") or ev.get("ranked_count")

    log(f"GW{gw} review: reading the gameweek ...")
    live_raw = fpl_api.event_live(gw, 0.25 if not final else 6.0)
    L = live_frame(live_raw)
    fx_gw = [f for f in fpl_api.fixtures(0.25) if f.get("event") == gw]
    to_play = {f["id"] for f in fx_gw if not (f.get("finished") or f.get("finished_provisional"))}
    teams_to_play = {t for f in fx_gw if f["id"] in to_play for t in (f["team_h"], f["team_a"])}
    pending = {i for i, e in els.items() if e["team"] in teams_to_play}
    pts = L["pts"].to_dict()
    mins = L["minutes"].to_dict()
    hist = fpl_api.entry_history(entry_id, 0.25)
    rows = {g["event"]: g for g in hist.get("current", [])}
    me = rows.get(gw)
    picks = fpl_api.entry_picks(entry_id, gw, 0.25 if not final else 24.0)
    chips_used = {c["event"]: c["name"] for c in hist.get("chips", [])}
    chip = picks.get("active_chip") or chips_used.get(gw)
    eh = picks.get("entry_history") or me or {}
    hits = int(eh.get("event_transfers_cost") or 0)
    ours = team_parts(picks["picks"], chip, hits, pts, mins, pos)
    reviews = [r for r in load_reviews() if r["gw"] != gw]
    previous = next((r for r in reversed(reviews) if r["gw"] == gw - 1), None)

    # the squad before this gameweek's moves (the previous real squad if last week was a Free Hit)
    prev_gw = gw - 1
    while prev_gw >= 1 and chips_used.get(prev_gw) == "freehit":
        prev_gw -= 1
    old = None
    if prev_gw >= 1 and prev_gw in rows:
        try:
            pp = fpl_api.entry_picks(entry_id, prev_gw, 24.0 * 30)
            old = team_parts(pp["picks"], None, 0, pts, mins, pos)
        except Exception:  # noqa: BLE001
            old = None

    # what the engine recommended and whether the account followed it
    plan = None
    try:
        plan = json.loads((REPORTS / f"gw{gw:02d}" / "latest.json").read_text())
    except (OSError, ValueError):
        pass
    engine = bool(plan and set(plan["lineup"]) == set(ours["lineup"]) and set(plan["bench"]) == set(ours["bench"])
                  and plan["captain"] == ours["captain"])
    rec_score = None
    if plan and not engine:
        rec_picks = [{"element": p, "position": i + 1, "is_captain": p == plan["captain"],
                      "is_vice_captain": p == plan["vice"]} for i, p in enumerate(plan["lineup"] + plan["bench"])]
        rec_score = team_parts(rec_picks, plan.get("chip") if plan.get("chip") in ("bboost", "3xc") else None,
                               4 * int(plan.get("hits") or 0), pts, mins, pos)["net"]

    # operations: hits nobody planned, runs that stopped
    incidents = []
    for f in sorted((REPORTS / f"gw{gw:02d}").glob("applied-*.json")):
        try:
            a = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if a.get("error"):
            when = pd.Timestamp(a["when"]).strftime("%a %d %b %H:%M UTC") if a.get("when") else f.stem
            incidents.append(f"{when}: {a['error']}")
    planned_cost = 4 * int(plan.get("hits") or 0) if plan else hits
    ops_lost = float(max(0, hits - planned_cost)) if engine else 0.0

    # the projection saved before the deadline
    snap = snapshot_for(gw, deadline)
    proj = model = Pp = None
    if snap is not None and len(snap):
        snap = snap[~snap["fixture"].isin(to_play) & ~snap["pid"].isin(pending)]
        log(f"GW{gw} review: comparing with the projection ...")
        Pp = per_player(snap)
        proj = _team_vs_projection(ours, Pp, L, pos)
        model = _model_check(Pp, L, snap, bs, gw)

    # the elite: top 50 before the gameweek
    log(f"GW{gw} review: reading the top {ELITE_N} managers' teams ...")
    ids, elite_src = elite_entries(gw, previous, now_is_gw)
    with ThreadPoolExecutor(4) as ex:
        ep = list(ex.map(lambda e: _picks(e, gw), ids))
    elite = []
    for e, p in zip(ids, ep):
        if not p or not p.get("picks"):
            continue
        h = int((p.get("entry_history") or {}).get("event_transfers_cost") or 0)
        t = team_parts(p["picks"], p.get("active_chip"), h, pts, mins, pos)
        t["entry"] = e
        elite.append(t)
    dfc = deficits(ours, elite)
    elite_info = None
    if elite:
        own = pd.Series([p for t in elite for p in t["squad"]]).value_counts() / len(elite)
        capn = pd.Series([t["armband"] for t in elite if t["armband"]]).value_counts() / len(elite)
        chips_e = pd.Series([t["chip"] for t in elite if t["chip"]]).value_counts()
        elite_info = {
            "n": len(elite), "source": elite_src,
            "avg": {k: float(np.mean([t[k] for t in elite])) for k in ("net", "best_xi", "xi", "selection", "captain_extra", "chip_extra", "hits")},
            "best_net": float(max(t["net"] for t in elite)),
            "captains": {str(k): float(v) for k, v in capn.head(6).items()},
            "ownership": {str(k): float(v) for k, v in own[own >= 0.2].items()},
            "chips": {str(k): int(v) for k, v in chips_e.items()},
        }
    ids_next = []
    if now_is_gw:
        try:
            ids_next = [r["entry"] for r in sorted(_standings_page(1, 2.0)["standings"]["results"],
                                                   key=lambda r: r["rank_sort"])[:ELITE_N]]
        except Exception:  # noqa: BLE001
            ids_next = []
    if elite_info is not None:
        elite_info["ids_next"] = ids_next

    th = thresholds() if now_is_gw else {}
    dcov = part_noise(elite)
    win = window(reviews + ([{"gw": gw, "engine_managed": engine, "deficits": dfc, "deficit_cov": dcov}]
                            if dfc else []), gw)
    factor, avg = choose_factor(win, ops_lost)
    cert = certainty(win, factor)

    data = {
        "version": VERSION, "gw": gw, "final": final, "generated": pd.Timestamp.now(tz="UTC").strftime("%d %b %Y %H:%M UTC"),
        "matches_to_play": len(to_play), "squad_to_play": sorted(p for p in ours["squad"] if p in pending),
        "engine_managed": engine, "chip": chip,
        "summary": {"points": int(eh.get("points", ours["net"] + hits)), "hits": hits, "net": int(eh.get("points", 0)) - hits,
                    "average": ev.get("average_entry_score"), "highest": ev.get("highest_score"),
                    "gw_rank": eh.get("rank"), "overall_rank": (me or {}).get("overall_rank") or eh.get("overall_rank"),
                    "overall_rank_prev": (rows.get(gw - 1) or {}).get("overall_rank"),
                    "total_points": (me or {}).get("total_points") or eh.get("total_points"),
                    "total_players": total_players, "bench_points": eh.get("points_on_bench"),
                    "projected": proj["xi_projected"] if proj else (plan or {}).get("xp")},
        "team": {k: ours[k] for k in ("best_xi", "xi", "selection", "captain_extra", "chip_extra", "hits", "net",
                                       "best_captain", "captain_loss", "bench_points")}
                | {"captain": ours["captain"], "armband": ours["armband"],
                   "old_squad_net": old["xi"] + old["captain_extra"] if old else None,
                   "transfer_gain": (ours["xi"] + ours["captain_extra"] - hits) - (old["xi"] + old["captain_extra"]) if old else None,
                   "recommended_net": rec_score},
        "projection": {k: v for k, v in (proj or {}).items() if k not in ("table",)} if proj else None,
        "model": model,
        "elite": elite_info, "deficits": dfc, "deficit_cov": dcov,
        "operations": {"incidents": incidents, "points_lost": ops_lost},
        "thresholds": th,
        "limiting_factor": factor, "window": {"gws": [r["gw"] for r in win], "avg": avg, "certainty": cert},
    }
    rv = Review(gw, final, data)
    ctx = dict(ours=ours, old=old, plan=plan, proj=proj, model=model, elite=elite, name=name, short=short, Pp=Pp,
               els=els, pos=pos, pts=pts, mins=mins, L=L, hist=hist, rows=rows, reviews=reviews, ev=ev, pending=pending)
    rv.sections = _render_sections(rv, ctx)
    return rv


def default_gw(bs: dict) -> int:
    """The latest gameweek whose matches have all been played (checked or not)."""
    fx = fpl_api.fixtures(0.5)
    by_gw: dict[int, list] = {}
    for f in fx:
        if f.get("event"):
            by_gw.setdefault(f["event"], []).append(f)
    done = [e["id"] for e in bs["events"] if e.get("finished")]
    for e in sorted(bs["events"], key=lambda e: -e["id"]):
        fs = by_gw.get(e["id"], [])
        if e.get("finished") or (fs and all(f.get("finished") or f.get("finished_provisional") for f in fs)):
            return e["id"]
    if done:
        return max(done)
    raise RuntimeError("No gameweek has been played yet.")


# ----------------------------------------------------------------------------- analysis
def _team_vs_projection(ours: dict, Pp: pd.DataFrame, L: pd.DataFrame, pos: dict) -> dict:
    mult = {p: 1 for p in ours["lineup"]}
    if ours["captain"] in mult:
        mult[ours["captain"]] = 3 if ours["chip"] == "3xc" else 2
    if ours["chip"] == "bboost":
        mult.update({p: 1 for p in ours["bench"]})
    xp = lambda p: float(Pp["xp"].get(p, 0.0))  # noqa: E731
    projected = sum(xp(p) * m for p, m in mult.items())
    played = lambda p: L["minutes"].get(p, 0) > 0  # noqa: E731
    lines = {c: 0.0 for c in COMPONENTS + ["other"]}
    for p, m in mult.items():
        if not played(p):
            continue
        for c in COMPONENTS:
            lines[c] += m * (float(L.at[p, f"a_{c}"]) if p in L.index else 0.0) - m * float(Pp[c].get(p, 0.0))
        lines["other"] += m * (float(L.at[p, "a_other"]) if p in L.index else 0.0)
    absent = [p for p in mult if not played(p)]
    absent_loss = -sum(xp(p) * mult[p] for p in absent)
    subs_in = [b for _, b in ours["subs"]]
    subs_gain = float(sum(L["pts"].get(b, 0) for b in subs_in))
    vice_extra = 0.0
    if ours["captain"] in absent and ours["armband"] is not None and ours["chip"] != "bboost":
        vice_extra = float(L["pts"].get(ours["armband"], 0)) * (2 if ours["chip"] == "3xc" else 1)
    actual = sum(float(L["pts"].get(p, 0)) * mult[p] for p in mult if played(p)) + subs_gain + vice_extra
    gap = {"Starters who did not play": absent_loss, "Automatic substitutes": subs_gain,
           "Armband passed to the vice-captain": vice_extra}
    gap.update({COMP_LABEL[c]: v for c, v in lines.items() if abs(v) > 1e-9})
    # luck or model: projected goals / assists against the chances the eleven actually got
    w = {p: m for p, m in mult.items()}
    att = {"goals": {"projected": sum(float(Pp["x_goals"].get(p, 0)) * m for p, m in w.items()),
                     "chances (xG)": sum(float(L["xg"].get(p, 0)) * m for p, m in w.items()),
                     "scored": sum(float(L["goals_n"].get(p, 0)) * m for p, m in w.items())},
           "assists": {"projected": sum(float(Pp["x_assists"].get(p, 0)) * m for p, m in w.items()),
                       "chances (xA)": sum(float(L["xa"].get(p, 0)) * m for p, m in w.items()),
                       "made": sum(float(L["assists_n"].get(p, 0)) * m for p, m in w.items())}}
    exp_absent = float(sum(xp(p) for p in absent))
    exp_subs = float(sum(xp(b) for b in subs_in))
    return {"xi_projected": round(projected, 2), "xi_actual": round(actual, 2), "gap": {k: round(v, 2) for k, v in gap.items()},
            "attack": {k: {a: round(b, 2) for a, b in v.items()} for k, v in att.items()},
            "absent": absent, "minutes_loss": round(exp_absent - exp_subs, 2),
            "bench_projected": round(sum(xp(p) for p in ours["bench"]), 2)}


def _model_check(Pp: pd.DataFrame, L: pd.DataFrame, snap: pd.DataFrame, bs: dict, gw: int) -> dict:
    d = Pp.join(L, how="inner")
    if d.empty:
        return {}
    d["played"] = (d["minutes"] > 0).astype(float)
    likely = d[d["p_play"] > 0.5]
    y, x = likely["pts"].to_numpy(float), likely["xp"].to_numpy(float)
    out = {"n_players": int(len(d)), "n_likely": int(len(likely)),
           "mean_projected": float(x.mean()) if len(x) else None, "mean_actual": float(y.mean()) if len(y) else None,
           "bias": float((x - y).mean()) if len(x) else None,
           "mae": float(np.abs(x - y).mean()) if len(x) else None,
           "corr": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else None}
    if "ep_next" in likely and likely["ep_next"].notna().sum() > 20:
        f = likely[likely["ep_next"].notna()]
        out["fpl_corr"] = float(np.corrcoef(f["ep_next"].astype(float), f["pts"])[0, 1])
        out["corr_same"] = float(np.corrcoef(f["xp"], f["pts"])[0, 1])
        out["fpl_mae"] = float(np.abs(f["ep_next"].astype(float) - f["pts"]).mean())
    by_pos = {}
    for p, g in likely.groupby("pos"):
        by_pos[POS[int(p)]] = {"n": int(len(g)), "projected": float(g["xp"].mean()), "actual": float(g["pts"].mean()),
                               "corr": float(np.corrcoef(g["xp"], g["pts"])[0, 1]) if len(g) > 2 else None}
    out["by_pos"] = by_pos
    out["components"] = {c: {"projected": float(d[c].sum()), "actual": float(d[f"a_{c}"].sum())} for c in COMPONENTS}
    out["attack"] = {"x_goals": float(d["x_goals"].sum()), "xg": float(d["xg"].sum()), "goals": float(d["goals_n"].sum()),
                     "x_assists": float(d["x_assists"].sum()), "xa": float(d["xa"].sum()), "assists": float(d["assists_n"].sum())}
    # minutes: calibration of P(plays) and regulars who did not play
    bins = [0, 0.2, 0.5, 0.8, 0.95, 1.0001]
    cut = pd.cut(d["p_play"], bins, right=False)
    cal = d.groupby(cut, observed=True).agg(n=("played", "size"), predicted=("p_play", "mean"), actual=("played", "mean"))
    out["minutes_calibration"] = [{"bin": str(i), "n": int(r.n), "predicted": float(r.predicted), "actual": float(r.actual)}
                                  for i, r in cal.iterrows()]
    nailed = d[(d["p_play"] >= 0.85) & (d["minutes"] == 0)]
    out["nailed_n"] = int((d["p_play"] >= 0.85).sum())
    out["nailed_missed"] = [{"pid": int(i), "name": r["name"], "team": r["team"], "p_play": float(r["p_play"]),
                             "status": r.get("status", ""), "news": r.get("news", "")} for i, r in nailed.iterrows()]
    out["logloss_play"] = float(-np.mean(d["played"] * np.log(d["p_play"].clip(1e-4, 1 - 1e-4))
                                         + (1 - d["played"]) * np.log(1 - d["p_play"].clip(1e-4, 1 - 1e-4))))
    # the top end: the players a captain is picked from
    top = d.nlargest(10, "xp")
    out["top10"] = {"projected": float(top["xp"].mean()), "actual": float(top["pts"].mean()),
                    "players": [{"pid": int(i), "name": r["name"], "xp": float(r["xp"]), "pts": int(r["pts"])}
                                for i, r in top.iterrows()]}
    # team goals: the fixture's expected goals against the chances and goals the team got
    tid = {e["id"]: e["team"] for e in bs["elements"]}
    short = {t["id"]: t["short_name"] for t in bs["teams"]}
    lam = snap.drop_duplicates(["fixture", "team_code"]).groupby("team")["lam"].sum()
    L2 = L.assign(team=[short.get(tid.get(i)) for i in L.index])
    txg = L2.groupby("team")["xg"].sum()
    fx = [f for f in fpl_api.fixtures(1.0) if f.get("event") == gw and f.get("team_h_score") is not None]
    goals: dict[str, float] = {}
    for f in fx:
        goals[short[f["team_h"]]] = goals.get(short[f["team_h"]], 0) + f["team_h_score"]
        goals[short[f["team_a"]]] = goals.get(short[f["team_a"]], 0) + f["team_a_score"]
    T = pd.DataFrame({"projected": lam, "xg": txg, "goals": pd.Series(goals)}).dropna()
    if len(T):
        out["team_goals"] = {"mae_xg": float((T["projected"] - T["xg"]).abs().mean()),
                             "bias_xg": float((T["projected"] - T["xg"]).mean()),
                             "corr_xg": float(np.corrcoef(T["projected"], T["xg"])[0, 1]) if len(T) > 2 else None,
                             "worst": [{"team": t, "projected": float(r.projected), "xg": float(r.xg), "goals": int(r.goals)}
                                       for t, r in T.assign(e=(T["projected"] - T["xg"]).abs()).nlargest(3, "e").iterrows()]}
    miss = likely.assign(err=likely["pts"] - likely["xp"])
    out["under"] = [{"name": r["name"], "team": r["team"], "xp": float(r["xp"]), "pts": int(r["pts"])}
                    for _, r in miss.nlargest(5, "err").iterrows()]
    out["over"] = [{"name": r["name"], "team": r["team"], "xp": float(r["xp"]), "pts": int(r["pts"])}
                   for _, r in miss.nsmallest(5, "err").iterrows()]
    # the same checks by club, so `evidence.pool` can add them up over the season with standard errors
    out["errors"] = evidence.collect(likely, d, top, T if len(T) else None, COMPONENTS, bins, POS)
    return out


# ----------------------------------------------------------------------------- rendering
def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _what_happened(p: int, L: pd.DataFrame, pos: dict, pending=()) -> str:
    if p in pending:
        return "still to play"
    if p not in L.index:
        return "no match"
    r = L.loc[p]
    if r["minutes"] == 0:
        return "did not play"
    bits = []
    if r["goals_n"]:
        bits.append(f"{int(r['goals_n'])} goal" + ("s" if r["goals_n"] > 1 else ""))
    if r["assists_n"]:
        bits.append(f"{int(r['assists_n'])} assist" + ("s" if r["assists_n"] > 1 else ""))
    if r["cs_n"] and pos.get(p) in (1, 2, 3):
        bits.append("clean sheet")
    if r["a_gc"] < 0:
        bits.append(f"{int(r['a_gc'])} for goals conceded")
    if r["a_defcon"] > 0:
        bits.append("defensive contribution")
    if r["bonus_n"]:
        bits.append(f"{int(r['bonus_n'])} bonus")
    if r["a_saves"] > 0:
        bits.append(f"{int(r['saves_n'])} saves")
    if r["a_cards"] < 0:
        bits.append("card/own goal/missed pen")
    if r["minutes"] < 60:
        bits.append(f"{int(r['minutes'])} min")
    if r["xg"] + r["xa"] >= 0.3 and not r["goals_n"] and not r["assists_n"]:
        bits.append(f"{r['xg']:.2f} xG, {r['xa']:.2f} xA, no return")
    elif r["xg"] + r["xa"] >= 0.3:
        bits.append(f"{r['xg']:.2f} xG, {r['xa']:.2f} xA")
    return ", ".join(bits) or "blank"


def _render_sections(rv: Review, c: dict) -> list[str]:
    D, gw = rv.data, rv.gw
    S, T, ours, name = D["summary"], D["team"], c["ours"], c["name"]
    out: list[str] = []
    status = ("Final: FPL has checked the gameweek's data." if rv.final else
              "Provisional: FPL has not finished checking this gameweek, so bonus points and ranks may still "
              "change. The autopilot rewrites this review when the data is final."
              + (f" {D['matches_to_play']} match{'es' if D['matches_to_play'] != 1 else ''} still to play"
                 + (f", including {len(D['squad_to_play'])} of your players." if D["squad_to_play"] else
                    f", none of your players in {'it' if D['matches_to_play'] == 1 else 'them'}; those teams are "
                    "left out of the model check.")
                 if D["matches_to_play"] else ""))
    proj = c["proj"]
    head = [f"# Gameweek {gw} review", "", f"_{status} Written {D['generated']}._", ""]
    rank, prev = S["overall_rank"], S["overall_rank_prev"]
    move = ""
    if rank and prev:
        dlt = prev - rank
        move = (f", up {dlt:,} places" if dlt > 0 else f", down {-dlt:,} places" if dlt < 0 else ", unchanged")
    pct = lambda r: f"top {100 * r / S['total_players']:.0f}%" if r and S.get("total_players") else ""  # noqa: E731
    bullets = [f"**{S['net']} points**" + (f" ({S['points']} minus {S['hits']} for transfers)" if S["hits"] else "")
               + (f", projected {proj['xi_projected']:.1f}" if proj else "")
               + f"; game average {S['average']}, highest {S['highest']}."]
    if S.get("gw_rank"):
        bullets.append(f"Gameweek rank {_fmt_rank(S['gw_rank'])} ({pct(S['gw_rank'])}); overall rank "
                       f"{_fmt_rank(rank)} ({pct(rank)}{move}).")
    if D["chip"]:
        bullets.append(f"Chip: {CHIP_NAMES.get(D['chip'], D['chip'])}.")
    if D.get("elite"):
        e = D["elite"]
        gap = e["avg"]["net"] - S["net"]
        bullets.append(f"The top {e['n']} managers averaged {e['avg']['net']:.1f} points, {abs(gap):.1f} "
                       f"{'more' if gap > 0 else 'fewer'} than you.")
    th = D.get("thresholds") or {}
    if "1" in th and S.get("total_points"):
        left = 38 - gw
        gap = th["1"]["total"] - S["total_points"]
        bullets.append(f"First place has {th['1']['total']} points: {gap} ahead of you"
                       + (f", {gap / left:.1f} points a week over {left} gameweeks." if left > 0 and gap > 0 else "."))
    if D["limiting_factor"]:
        bullets.append(f"Limiting factor: **{D['limiting_factor']}** ({FACTOR_TEXT[D['limiting_factor']]}).")
    if not D["engine_managed"]:
        bullets.append("The team on the account was not the engine's recommendation this week"
                       + (f" (the recommendation would have scored {T['recommended_net']:.0f})" if T.get("recommended_net") is not None else "")
                       + ", so this week does not count towards the limiting factor.")
    out.append("\n".join(head + [f"- {b}" for b in bullets]))

    # ---- against the projection
    pos, L, snap = c["pos"], c["L"], c["Pp"]
    sec = ["## How the team did against the projection", ""]
    roles = {}
    for i, p in enumerate(ours["lineup"]):
        roles[p] = "C" if p == ours["captain"] else "V" if p == ours["vice"] else "XI"
    for i, p in enumerate(ours["bench"]):
        roles[p] = f"bench {i + 1}" if pos.get(p) != 1 else "bench GK"
    subs_in = {b: o for o, b in ours["subs"]}
    subs_out = {o: b for o, b in ours["subs"]}
    rows = []
    for p in ours["lineup"] + ours["bench"]:
        xp = float(snap["xp"].get(p, 0.0)) if snap is not None else None
        pp = float(snap["p_play"].get(p, 0.0)) if snap is not None else None
        a = int(c["pts"].get(p, 0))
        role = roles[p] + (f" (on for {name[subs_in[p]]})" if p in subs_in else "") \
            + (f" (off for {name[subs_out[p]]})" if p in subs_out else "")
        rows.append([POS[pos[p]], name[p], role, f"{xp:.1f}" if xp is not None else "-",
                     f"{pp:.0%}" if pp is not None else "-", int(c["mins"].get(p, 0)), a,
                     _signed(a - xp) if xp is not None else "-", _what_happened(p, L, pos, c["pending"])])
    sec.append(_table(["Pos", "Player", "Role", "Projected", "P(plays)", "Mins", "Points", "Diff", "What happened"], rows))
    sec.append("")
    sec.append("Points are before the armband; the captain's are doubled in the totals below.")
    if proj:
        sec += ["", f"The eleven as picked (captain counted twice) were projected **{proj['xi_projected']:.1f}** and "
                    f"scored **{proj['xi_actual']:.0f}**. Where the difference came from:", ""]
        g = proj["gap"]
        sec.append(_table(["Source", "Points"], [[k, _signed(v)] for k, v in g.items() if abs(v) >= 0.05]
                          + [["**Total**", f"**{_signed(proj['xi_actual'] - proj['xi_projected'])}**"]]))
        at = proj["attack"]
        sec += ["", "Luck or model? The eleven's goals and assists (captain counted twice):", "",
                _table(["", "Projected", "Chances they got", "Actual"],
                       [["Goals", f"{at['goals']['projected']:.2f}", f"{at['goals']['chances (xG)']:.2f} xG", f"{at['goals']['scored']:.0f}"],
                        ["Assists", f"{at['assists']['projected']:.2f}", f"{at['assists']['chances (xA)']:.2f} xA", f"{at['assists']['made']:.0f}"]]),
                "", "A projection close to the chances but far from the actual is finishing luck; a projection far "
                    "from the chances is the model."]
    else:
        sec += ["", "No projection was saved before this deadline, so only actual points are shown."]
    sec += ["", "In hindsight, from the same 15 players:", ""]
    hs = [f"Best possible eleven: **{T['best_xi']:.0f}** points, against {T['xi']:.0f} from the eleven that played "
          f"(**{T['selection']:.0f}** left on the bench beyond what was needed; bench total {T['bench_points']:.0f})."]
    arm = ours["armband"]
    hs.append(f"Best captain among the eleven: {T['best_captain']:.0f} points, against {T['captain_extra']:.0f} from "
              f"{name.get(arm, 'nobody')} (**{T['captain_loss']:.0f}** lost to the armband).")
    if T.get("old_squad_net") is not None:
        hs.append(f"The squad from before this week's transfers would have scored {T['old_squad_net']:.0f} "
                  f"(before chips): the moves {'gained' if T['transfer_gain'] >= 0 else 'cost'} "
                  f"**{abs(T['transfer_gain']):.0f}** points this week" + (", hits included." if T["hits"] else ".")
                  + (f" A {CHIP_NAMES[D['chip']]} is built for the weeks after it, so one week says little about it."
                     if D["chip"] in ("wildcard", "freehit") and D["chip"] != "freehit" else ""))
    if proj and proj["absent"]:
        hs.append(f"Starters who did not play: {', '.join(name[p] for p in proj['absent'])} "
                  f"(expected value lost after substitutes: {proj['minutes_loss']:.1f}).")
    sec += [f"- {h}" for h in hs]
    out.append("\n".join(sec))

    # ---- the model across all players
    m = c["model"]
    if m:
        sec = ["## How the model did across all players", ""]
        sec.append(f"Likely starters (P(plays) over 50%): {m['n_likely']}. Projected {m['mean_projected']:.2f} points each, "
                   f"scored {m['mean_actual']:.2f} (bias {_signed(m['bias'], 2)}); correlation with actual points "
                   f"**{m['corr']:.2f}**" + (f", FPL's own projection on the same players {m['fpl_corr']:.2f} "
                                              f"(ours {m['corr_same']:.2f})." if m.get("fpl_corr") is not None else "."))
        sec += ["", _table(["Position", "Players", "Projected", "Actual", "Correlation"],
                           [[k, v["n"], f"{v['projected']:.2f}", f"{v['actual']:.2f}",
                             f"{v['corr']:.2f}" if v["corr"] is not None else "-"] for k, v in m["by_pos"].items()])]
        comp_rows = [[COMP_LABEL[k], f"{v['projected']:.0f}", f"{v['actual']:.0f}", _signed(v["actual"] - v["projected"], 0)]
                     for k, v in m["components"].items() if abs(v["projected"]) + abs(v["actual"]) >= 1]
        sec += ["", "Points by component, all players:", "", _table(["Component", "Projected", "Actual", "Diff"], comp_rows)]
        a = m["attack"]
        sec += ["", f"Goals: projected {a['x_goals']:.0f}, chances worth {a['xg']:.0f} xG, scored {a['goals']:.0f}. "
                    f"Assists: projected {a['x_assists']:.0f}, {a['xa']:.0f} xA, made {a['assists']:.0f}."]
        if m.get("team_goals"):
            tg = m["team_goals"]
            worst = "; ".join(f"{w['team']} {w['projected']:.1f} projected, {w['xg']:.1f} xG, {w['goals']} scored" for w in tg["worst"])
            sec += ["", f"Team goals: the fixtures' expected goals missed the chances teams actually created by "
                        f"{tg['mae_xg']:.2f} a team on average (bias {_signed(tg['bias_xg'], 2)}). Furthest out: {worst}."]
        mc = m["minutes_calibration"]
        sec += ["", "Minutes: how often players played, by the chance the model gave them:", "",
                _table(["P(plays)", "Players", "Predicted", "Played"],
                       [[r["bin"], r["n"], f"{r['predicted']:.0%}", f"{r['actual']:.0%}"] for r in mc])]
        if m["nailed_missed"]:
            sec += ["", f"Regulars (85%+) who did not play: {len(m['nailed_missed'])} of {m['nailed_n']}: "
                        + "; ".join(f"{x['name']} ({x['team']}, {x['p_play']:.0%}"
                                    + (f", flagged {x['status']}: {x['news']}" if x["status"] not in ("a", "") else ", no flag")
                                    + ")" for x in m["nailed_missed"][:12]) + "."]
        t10 = m["top10"]
        sec += ["", f"The ten highest projections (where captains come from) averaged {t10['projected']:.1f} projected and "
                    f"{t10['actual']:.1f} scored: " + ", ".join(f"{x['name']} {x['xp']:.1f}→{x['pts']}" for x in t10["players"]) + "."]
        sec += ["", "Furthest under: " + ", ".join(f"{x['name']} ({x['team']}) {x['xp']:.1f}→{x['pts']}" for x in m["under"]) + ".",
                "Furthest over: " + ", ".join(f"{x['name']} ({x['team']}) {x['xp']:.1f}→{x['pts']}" for x in m["over"]) + "."]
        out.append("\n".join(sec))

    # ---- toward first place
    sec = ["## Progress towards first place in the world", ""]
    left = 38 - gw
    if th and S.get("total_points"):
        rows = []
        labels = {"1": "First place", "100": "Top 100", "1000": "Top 1k", "10000": "Top 10k", "100000": "Top 100k",
                  "1000000": "Top 1M"}
        for k, lab in labels.items():
            if k not in th:
                continue
            gap = th[k]["total"] - S["total_points"]
            rows.append([lab, th[k]["total"], f"{th[k]['total'] / gw:.1f}", th[k]["event_total"],
                         gap if gap > 0 else f"ahead by {-gap}", f"{gap / left:.1f}" if gap > 0 and left else "-"])
        rows.append(["**You**", S["total_points"], f"{S['total_points'] / gw:.1f}", S["net"], "", ""])
        sec.append(_table(["", "Points", "Per GW", f"GW{gw}", "Gap", f"Needed per GW ({left} left)"], rows))
        prev_th = (next((r for r in reversed(c["reviews"]) if r.get("thresholds")), None) or {}).get("thresholds", {})
        prev_r = next((r for r in reversed(c["reviews"]) if r.get("thresholds")), None)
        if prev_th and prev_r:
            ch = []
            for k, lab in (("1", "first place"), ("10000", "the top 10k"), ("1000000", "the top 1M")):
                if k in prev_th and k in th:
                    before = prev_th[k]["total"] - (prev_r["summary"]["total_points"] or 0)
                    now = th[k]["total"] - S["total_points"]
                    ch.append(f"{lab} {'closed by' if now < before else 'widened by'} {abs(before - now)}")
            if ch:
                sec += ["", f"Since the GW{prev_r['gw']} review the gap to " + ", ".join(ch) + "."]
    else:
        sec.append("Rank thresholds are read from the live standings, so they are only shown in a review written "
                   "while this gameweek is the latest one.")
    hist = [g for g in c["hist"].get("current", []) if g["event"] <= gw]
    if hist:
        trows, prev_or = [], None
        averages = {e["id"]: e["average_entry_score"] for e in fpl_api.bootstrap(1.0)["events"]}
        for g in hist:
            o = g.get("overall_rank")
            arrow = "" if prev_or is None or not o else ("▲" if o < prev_or else "▼" if o > prev_or else "=")
            avg = averages.get(g["event"])
            trows.append([g["event"], g["points"] - g["event_transfers_cost"], avg, _fmt_rank(g.get("rank")),
                          f"{_fmt_rank(o)} {arrow}".strip()])
            prev_or = o or prev_or
        sec += ["", "<details><summary>Season so far</summary>", "",
                _table(["GW", "Points", "Average", "GW rank", "Overall rank"], trows), "", "</details>"]
        sec += ["", "Milestones:", ""] + [f"- {x}" for x in _milestones(hist, gw, c["reviews"], D)]
    if D.get("elite"):
        e = D["elite"]
        dfc = D["deficits"]
        sec += ["", f"Against the top {e['n']} ({e['source']}):", "",
                _table(["Part", "You", f"Top {e['n']} average", "Behind by"],
                       [["Squad: best possible eleven", f"{T['best_xi']:.0f}", f"{e['avg']['best_xi']:.1f}", f"{dfc['squad']:.1f}"],
                        ["Selection: left on the bench", f"{T['selection']:.0f}", f"{e['avg']['selection']:.1f}", f"{dfc['selection']:.1f}"],
                        ["Captaincy: armband", f"{T['captain_extra']:.0f}", f"{e['avg']['captain_extra']:.1f}", f"{dfc['captaincy']:.1f}"],
                        ["Chips: BB / TC", f"{T['chip_extra']:.0f}", f"{e['avg']['chip_extra']:.1f}", f"{dfc['chips']:.1f}"],
                        ["Hits", f"{T['hits']:.0f}", f"{e['avg']['hits']:.1f}", f"{dfc['hits']:.1f}"],
                        ["**Net points**", f"**{T['net']:.0f}**", f"**{e['avg']['net']:.1f}**", f"**{dfc['total']:.1f}**"]])]
        caps = ", ".join(f"{name.get(int(k), k)} {v:.0%}" for k, v in e["captains"].items())
        sec += ["", f"Their captains: {caps}. Yours: {name.get(ours['captain'])}."]
        own = {int(k): v for k, v in e["ownership"].items()}
        missing = sorted([p for p, v in own.items() if v >= 0.4 and p not in ours["squad"]], key=lambda p: -own[p])
        diffs = sorted([p for p in ours["squad"] if own.get(p, 0) < 0.2], key=lambda p: -c["pts"].get(p, 0))
        if missing:
            sec.append("Owned by 40%+ of them, not by you: " + ", ".join(
                f"{name[p]} ({own[p]:.0%}, {c['pts'].get(p, 0)} pts)" for p in missing[:10]) + ".")
        if diffs:
            sec.append("Yours, owned by under 20% of them: " + ", ".join(
                f"{name[p]} ({own.get(p, 0):.0%}, {c['pts'].get(p, 0)} pts)" for p in diffs[:10]) + ".")
        if e["chips"]:
            sec.append("Chips they played: " + ", ".join(f"{CHIP_NAMES.get(k, k)} {v}" for k, v in e["chips"].items()) + ".")
    out.append("\n".join(sec))

    # ---- what can be improved
    sec = ["## What can be improved", ""]
    dfc, win = D["deficits"], D["window"]
    cert = win.get("certainty") or {}
    se = cert.get("se") or {}
    if dfc:
        n = len(win["gws"])
        rows = [[f, f"{dfc[f]:+.1f}",
                 (f"{win['avg'][f]:+.1f}" + (f" ± {se[f]:.1f}" if f in se else "")) if win["avg"] else "-", LEVERS[f]]
                for f in FACTORS]
        sec += [f"Points behind the top {D['elite']['n']} by part (positive = behind), this week and averaged over the "
                f"last {n} engine-managed gameweek{'s' if n != 1 else ''} ({', '.join(f'GW{g}' for g in win['gws']) or 'none'}):", "",
                _table(["Part", f"GW{gw}", "Average ± s.e.", "What moves it"], rows), ""]
    f = D["limiting_factor"]
    if f:
        settled = cert.get("settled", False)
        sec.append(f"**Limiting factor: {f}**{'' if settled else ' (not settled)'} — {FACTOR_TEXT[f]}. Lever: {LEVERS[f]}.")
        sec.append("")
        sec.append(certainty_text(f, cert))
        if not settled:
            sec.append("Until it is, code changes are aimed by the model's errors pooled over every review (below), "
                       "not by the factor.")
        sec.append("")
    found = evidence.pool(c["reviews"] + [D])
    sec += ["Where the projections were wrong, pooled over the season's reviews:", "", evidence.summary(found)]
    if evidence.leads(found):
        sec += ["", _table(evidence.HEADERS, evidence.table(found, 5))]
    sec.append("")
    obs = _observations(D, c)
    if obs:
        sec += ["Observations:", ""] + [f"- {o}" for o in obs] + [""]
    if D["operations"]["incidents"]:
        sec += ["Autopilot incidents this gameweek:", ""] + [f"- {x}" for x in D["operations"]["incidents"]]
        sec += ["", f"Points they cost in the end: {D['operations']['points_lost']:.0f}.", ""]
    sec.append("Next step: `fpl improve` reads this review, tests changes to the settings behind the limiting "
               "factor on past seasons and keeps one only if it is clearly better; if none is, it writes a brief for a "
               f"code change (`reports/gw{gw:02d}_improvement.md`).")
    from . import tuning
    hist_t = tuning.history()
    if hist_t:
        sec += ["", "Changes in effect from earlier reviews: " + "; ".join(
            f"#{h['id']} {h['key']} {h['old']:g}→{h['new']:g} ({h['when'][:10]})" for h in hist_t[-5:]) + "."]
    out.append("\n".join(sec))
    return out


def _milestones(hist: list[dict], gw: int, reviews: list[dict], D: dict) -> list[str]:
    out = []
    g = {h["event"]: h for h in hist}
    me, prev = g.get(gw), g.get(gw - 1)
    ranked = [h for h in hist if h.get("overall_rank")]
    if me and me.get("overall_rank"):
        best = min(ranked, key=lambda h: h["overall_rank"])
        if best["event"] == gw and len(ranked) > 1:
            out.append(f"New season-best overall rank: {_fmt_rank(me['overall_rank'])}.")
        else:
            out.append(f"Season-best overall rank: {_fmt_rank(best['overall_rank'])} (GW{best['event']}).")
        if prev and prev.get("overall_rank"):
            b0, b1 = _rank_band(prev["overall_rank"]), _rank_band(me["overall_rank"])
            if b1 and (b0 is None or b1 < b0):
                out.append(f"Entered the top {b1:,}.")
            elif b0 and (b1 is None or b1 > b0):
                out.append(f"Dropped out of the top {b0:,}.")
        streak = 0
        for a, b in zip(reversed(ranked[:-1]), reversed(ranked)):
            if b["overall_rank"] < a["overall_rank"]:
                streak += 1
            else:
                break
        if streak >= 1:
            out.append(f"Green arrows in a row: {streak}.")
        nxt = max((b for b in RANK_BANDS if b < me["overall_rank"]), default=None)
        if nxt:
            out.append(f"Next milestone: the top {nxt:,} ({me['overall_rank'] - nxt:,} places to go).")
    if me:
        best_gw = min((h for h in hist if h.get("rank")), key=lambda h: h["rank"], default=None)
        if best_gw and best_gw["event"] == gw and len(hist) > 1:
            out.append(f"Best gameweek rank of the season: {_fmt_rank(me['rank'])}.")
    bs = fpl_api.bootstrap(1.0)
    avg = {e["id"]: e["average_entry_score"] for e in bs["events"]}
    beat = sum(1 for h in hist if h["points"] - h["event_transfers_cost"] > (avg.get(h["event"]) or 0))
    out.append(f"Beat the game average in {beat} of {len(hist)} gameweeks.")
    eng = [r for r in reviews if r.get("engine_managed")] + ([D] if D.get("engine_managed") else [])
    if eng:
        tot = sum(r["summary"]["net"] for r in eng)
        av = sum(r["summary"]["average"] or 0 for r in eng)
        el = [r["elite"]["avg"]["net"] for r in eng if r.get("elite")]
        out.append(f"Since the engine took over ({len(eng)} gameweek{'s' if len(eng) != 1 else ''}): {tot} points, "
                   f"{_signed(tot - av, 0)} against the average" + (f", {_signed(tot - sum(el), 0)} against the top 50." if el else "."))
    return out


def _observations(D: dict, c: dict) -> list[str]:
    """Plain statements of what stood out, each tied to what could change it."""
    o, T, m, proj, name = [], D["team"], D.get("model") or {}, D.get("projection"), c["name"]
    if T["selection"] >= 8:
        why = ""
        if proj and not proj["absent"]:
            why = (f" Every starter played, so the bench order played no part: the four on the bench were projected "
                   f"{proj['bench_projected']:.1f} between them and outscored starters projected higher, which is the "
                   "projections' ranking within the squad (or plain variance).")
        elif proj:
            why = " Some starters did not play; see the substitutes in the table above."
        o.append(f"{T['selection']:.0f} points were left on the bench beyond the best eleven." + why)
    if T["captain_loss"] >= 6:
        o.append(f"The armband earned {T['captain_extra']:.0f}; the best pick in the eleven would have earned "
                 f"{T['best_captain']:.0f}. One week of this is luck; the top-end check below says whether it is a pattern.")
    if proj:
        at = proj["attack"]["goals"]
        if at["projected"] > 0.5 and at["scored"] < at["chances (xG)"] - 0.8:
            o.append(f"The eleven got chances worth {at['chances (xG)']:.1f} goals and scored {at['scored']:.0f}: "
                     "finishing luck rather than the model.")
        elif at["projected"] - at["chances (xG)"] > 1.0:
            o.append(f"The eleven was projected {at['projected']:.1f} goals but created chances worth only "
                     f"{at['chances (xG)']:.1f}: the model overrated their attacking roles or fixtures.")
    if m:
        if m.get("fpl_corr") is not None and m["fpl_corr"] > m["corr_same"] + 0.05:
            o.append(f"FPL's own projection ranked players better this week ({m['fpl_corr']:.2f} against "
                     f"{m['corr_same']:.2f}). Worth watching: if it repeats, the points model is behind a simple benchmark.")
        if m.get("bias") is not None and abs(m["bias"]) > 0.4:
            o.append(f"Across likely starters the projections were {'high' if m['bias'] > 0 else 'low'} by "
                     f"{abs(m['bias']):.2f} points a player.")
        if len(m.get("nailed_missed", [])) >= 0.08 * max(1, m.get("nailed_n", 1)):
            unflag = [x for x in m["nailed_missed"] if x["status"] in ("a", "")]
            o.append(f"{len(m['nailed_missed'])} regulars the model gave 85%+ did not play"
                     + (f", {len(unflag)} of them without an injury flag (rotation or late news)." if unflag else "."))
        t10 = m.get("top10")
        if t10 and t10["actual"] < t10["projected"] - 2:
            o.append(f"The ten highest projections scored {t10['actual']:.1f} on average against {t10['projected']:.1f} "
                     "projected.")
    e = D.get("elite")
    if e:
        top_cap = max(e["captains"], key=e["captains"].get) if e["captains"] else None
        if top_cap and int(top_cap) != T["captain"] and e["captains"][top_cap] >= 0.4:
            o.append(f"{e['captains'][top_cap]:.0%} of the top {e['n']} captained {name.get(int(top_cap))} "
                     f"({c['pts'].get(int(top_cap), 0)} points); the engine picked {name.get(T['captain'])}. Going "
                     "against the leaders' captain is a large swing either way.")
    if T.get("transfer_gain") is not None and T["transfer_gain"] <= -6:
        o.append(f"The squad from before this week's moves would have scored {-T['transfer_gain']:.0f} more.")
    return o


# ----------------------------------------------------------------------------- writing
def render(rv: Review, notes: str = "") -> str:
    body = "\n\n".join(rv.sections)
    notes = notes.strip("\n")
    data = json.dumps(rv.data, default=_jsonable, separators=(",", ":"))
    return (body + f"\n\n{NOTES_HEADING}\n\n{NOTES_HINT}\n\n" + (notes + "\n\n" if notes else "")
            + f"{DATA_OPEN}\n{data}\n{DATA_CLOSE}\n")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def existing_notes(text: str) -> str:
    i = text.find(NOTES_HEADING)
    if i < 0:
        return ""
    rest = text[i + len(NOTES_HEADING):]
    j = rest.find(DATA_OPEN)
    rest = rest[: j if j >= 0 else None]
    return rest.replace(NOTES_HINT, "").strip("\n")


def write(gw: int | None = None, log=lambda m: None) -> tuple[Path, Review, str]:
    rv = build(gw, log=log)
    path = review_path(rv.gw)
    notes = existing_notes(path.read_text()) if path.exists() else ""
    text = render(rv, notes)
    REPORTS.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path, rv, text


def summary_lines(rv: Review) -> list[str]:
    """A few lines for the Discord report."""
    D, S = rv.data, rv.data["summary"]
    lines = [f"{S['net']} points" + (f" against {D['projection']['xi_projected']:.0f} projected" if D.get("projection") else "")
             + f"; average {S['average']}."]
    if D.get("elite"):
        lines.append(f"Top {D['elite']['n']} average: {D['elite']['avg']['net']:.1f}.")
    if D.get("limiting_factor"):
        settled = ((D.get("window") or {}).get("certainty") or {}).get("settled", False)
        lines.append(f"Limiting factor: {D['limiting_factor']} ({FACTOR_TEXT[D['limiting_factor']]})"
                     + ("." if settled else "; not settled yet."))
    lines.append(f"Full review: reports/{review_path(rv.gw).name}")
    return lines
