"""Command line entry point: `fpl <command>`.

    fpl plan        recommend transfers, line-up, captain and chips for the next deadline
    fpl team        your squad with projections
    fpl top         best projected players
    fpl player      one player's projection, broken into components
    fpl backtest    replay the optimiser over a past season
    fpl review      how past recommendations scored, next to your actual points
    fpl auto        run `plan` only when a deadline is near (for cron)
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings

import pandas as pd

from .config import POS, PROCESSED

POS_BY_NAME = {v: k for k, v in POS.items()}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _projection(max_age_h: float = 6.0, horizon: int = 9, refresh: bool = False):
    """Use the saved projection if it is recent and still for the upcoming gameweek."""
    from . import fpl_api, xp
    f = PROCESSED / "projection.pkl"
    if not refresh and f.exists() and (time.time() - f.stat().st_mtime) < max_age_h * 3600:
        with open(f, "rb") as fh:
            P = pickle.load(fh)
        if P.next_gw == fpl_api.current_and_next_gw()[1] and len(P.xp.columns) >= min(horizon, 39 - P.next_gw):
            return P
    _log("building projections ...")
    P = xp.project_live(horizon=horizon, refresh=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(f, "wb") as fh:
        pickle.dump(P, fh)
    return P


def _show(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 500):
        print(df.to_string())


def _xp_table(P, ids, gws: int) -> pd.DataFrame:
    cols = list(P.xp.columns[:gws])
    t = P.players.loc[ids, ["name", "team", "pos", "price", "own", "status"]].copy()
    t["pos"] = t["pos"].map(POS)
    t = t.join(P.xp.loc[ids, cols].round(1))
    t["total"] = P.xp.loc[ids, cols].sum(axis=1).round(1)
    return t.rename(columns={c: f"GW{c}" for c in cols})


# ----------------------------------------------------------------------------- commands
def cmd_plan(a) -> int:
    from . import planner, report
    try:
        rec = planner.make(horizon=a.horizon, ft=a.ft, use_chips=not a.no_chips, wildcard=a.wildcard,
                           freehit=a.freehit, quick=a.quick, sims=a.sims, refresh=not a.no_refresh,
                           banned=a.ban, locked=a.lock, max_hits=a.max_hits, workers=a.workers, log=_log)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    text = report.render(rec)
    print(text)
    if not a.no_save:
        paths = planner.save(rec, text)
        _log(f"saved {paths['markdown']}")
    return 0


def cmd_team(a) -> int:
    from . import my_team
    from .config import team_id
    P = _projection(horizon=a.gws)
    st = my_team.load(team_id())
    t = _xp_table(P, st.squad, a.gws)
    t.insert(4, "sell", [st.sell_price[p] / 10 for p in st.squad])
    print(f"GW{st.next_gw}: bank £{st.bank / 10:.1f}m, {st.free_transfers} free transfer(s), "
          f"chips: {', '.join(sorted({c['name'] for c in st.chips if min(c['gws']) <= st.next_gw})) or 'none'}")
    _show(t.sort_values(["pos", "total"], ascending=[True, False]))
    for n in st.notes:
        print("note:", n)
    return 0


def cmd_top(a) -> int:
    P = _projection(horizon=a.gws)
    pl = P.players
    mask = pd.Series(True, index=pl.index)
    if a.pos:
        mask &= pl["pos"] == POS_BY_NAME[a.pos.upper()]
    if a.max_price:
        mask &= pl["price"] <= a.max_price
    if a.team:
        mask &= pl["team"] == a.team.upper()
    tot = P.xp.loc[mask[mask].index, list(P.xp.columns[:a.gws])].sum(axis=1)
    if a.value:
        tot = tot / pl.loc[tot.index, "price"]
    _show(_xp_table(P, tot.nlargest(a.n).index, a.gws))
    return 0


def cmd_player(a) -> int:
    from . import planner
    P = _projection()
    try:
        pid = planner.resolve_players([" ".join(a.name)], P.players)[0]
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    r = P.players.loc[pid]
    print(f"{r['name']} ({r['team']}, {POS[int(r['pos'])]}) £{r['price']:.1f}m, owned by {r['own']:.1f}%, "
          f"status {r['status']}" + (f", news: {r['news']}" if r["news"] else ""))
    d = P.detail[P.detail["pid"] == pid].sort_values("kickoff")
    opp = P.fixtures.drop_duplicates(["fixture", "team_code"]).set_index(["fixture", "team_code"])["opp"]
    out = pd.DataFrame({
        "GW": d["gw"].astype(int), "opp": [opp.get((f, t), "?") for f, t in zip(d["fixture"], d["team_code"])],
        "H/A": d["home"].map({True: "H", False: "A"}), "team xG": d["lam"].round(2), "opp xG": d["lam_a"].round(2),
        "P(start60)": d["p60"].round(2), "mins": d["emin"].round(0),
    })
    for c in ["app", "goals", "assists", "cs", "defcon", "bonus", "saves", "gc", "cards", "xp"]:
        out[c] = d[c].round(2)
    _show(out.set_index("GW"))
    return 0


def cmd_backtest(a) -> int:
    from . import backtest
    res = backtest.run(a.season, strategies=a.strategy, horizon=a.horizon, first_gw=a.first_gw,
                       last_gw=a.last_gw, workers=a.workers, log=_log)
    print(backtest.render(res))
    return 0


def cmd_review(a) -> int:
    """How past recommendations did: their line-ups scored with actual points, next to what you scored."""
    import json

    from . import fpl_api
    from .config import REPORTS, team_id
    from .scoring import score_gw
    bs = fpl_api.bootstrap()
    pos = {e["id"]: e["element_type"] for e in bs["elements"]}
    done = {e["id"] for e in bs["events"] if e["finished"]}
    avg = {e["id"]: e["average_entry_score"] for e in bs["events"]}
    mine = {g["event"]: g["points"] - g["event_transfers_cost"]
            for g in fpl_api.entry_history(team_id(), 1.0).get("current", [])}
    rows = []
    for f in sorted(REPORTS.glob("gw*/latest.json")):
        r = json.loads(f.read_text())
        if r["gw"] not in done:
            continue
        live = {e["id"]: e["stats"] for e in fpl_api.event_live(r["gw"], 24.0 * 365)["elements"]}
        pts = {p: s["total_points"] for p, s in live.items()}
        mins = {p: s["minutes"] for p, s in live.items()}
        sc = score_gw(r["lineup"], r["bench"], r["captain"], r["vice"], pts, mins, pos, r.get("chip"))
        got = sc.points - 4 * r.get("hits", 0)
        rows.append({"GW": r["gw"], "projected": round(r["xp"], 1), "recommended team": int(got),
                     "your team": mine.get(r["gw"]), "game average": avg.get(r["gw"]),
                     "captain": r["names"].get(str(r["captain"]), r["captain"]), "captain pts": pts.get(r["captain"], 0)})
    if not rows:
        print("No finished gameweek has a saved recommendation yet (they are written by `fpl plan`).")
        return 0
    t = pd.DataFrame(rows).set_index("GW")
    _show(t)
    both = t.dropna(subset=["your team"])
    print(f"\nTotal over {len(both)} gameweek(s): recommended {int(both['recommended team'].sum())}, "
          f"yours {int(both['your team'].sum())}, average {int(both['game average'].sum())}")
    return 0


def cmd_auto(a) -> int:
    from . import fpl_api
    bs = fpl_api.bootstrap(0.0)
    ev = next((e for e in bs["events"] if e["is_next"]), None)
    if ev is None:
        _log("season is over; nothing to do")
        return 0
    left = (pd.Timestamp(ev["deadline_time"]) - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400.0
    if left > a.within_days and not a.force:
        _log(f"GW{ev['id']} deadline is {left:.1f} days away (running within {a.within_days}); nothing to do")
        return 0
    _log(f"GW{ev['id']} deadline in {left:.1f} days: planning")
    a.horizon = a.ft = a.wildcard = a.freehit = a.workers = None
    a.no_chips = a.quick = a.no_refresh = a.no_save = False
    a.sims, a.ban, a.lock, a.max_hits = 20, [], [], 2
    return cmd_plan(a)


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fpl", description="Fantasy Premier League decision engine.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="recommend transfers, line-up, captain and chips for the next deadline")
    p.add_argument("--horizon", type=int, help="gameweeks to plan (default: 9, up to 12 while a chip is about to expire)")
    p.add_argument("--ft", type=int, help="free transfers you have, if the reconstructed number is wrong")
    p.add_argument("--no-chips", action="store_true", help="plan without any chip")
    p.add_argument("--wildcard", type=int, metavar="GW", help="force the wildcard in this gameweek")
    p.add_argument("--freehit", type=int, metavar="GW", help="force the free hit in this gameweek")
    p.add_argument("--quick", action="store_true", help="skip the chip-timing search (about a minute instead of several)")
    p.add_argument("--sims", type=int, default=20, help="re-solves for the confidence check (0 to skip)")
    p.add_argument("--ban", action="append", default=[], metavar="PLAYER", help="never buy this player (repeatable)")
    p.add_argument("--lock", action="append", default=[], metavar="PLAYER", help="always own this player (repeatable)")
    p.add_argument("--max-hits", type=int, default=2, help="most -4 hits allowed in one gameweek")
    p.add_argument("--no-refresh", action="store_true", help="use cached data instead of downloading")
    p.add_argument("--no-save", action="store_true", help="do not write the report to reports/")
    p.add_argument("--workers", type=int, help="parallel solver processes")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("team", help="your squad with projections")
    p.add_argument("--gws", type=int, default=6)
    p.set_defaults(fn=cmd_team)

    p = sub.add_parser("top", help="best projected players")
    p.add_argument("--pos", choices=["GKP", "DEF", "MID", "FWD", "gkp", "def", "mid", "fwd"])
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--gws", type=int, default=6)
    p.add_argument("--max-price", type=float)
    p.add_argument("--team", help="three-letter club code, e.g. ARS")
    p.add_argument("--value", action="store_true", help="rank by points per £m")
    p.set_defaults(fn=cmd_top)

    p = sub.add_parser("player", help="one player's projection, broken into components")
    p.add_argument("name", nargs="+")
    p.set_defaults(fn=cmd_player)

    p = sub.add_parser("backtest", help="replay the optimiser over a past season")
    p.add_argument("--season", default="2025-26")
    p.add_argument("--strategy", action="append", choices=["plan", "myopic", "hold"],
                   help="repeatable; default: all three")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--first-gw", type=int, default=1)
    p.add_argument("--last-gw", type=int, default=38)
    p.add_argument("--workers", type=int)
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("review", help="how past recommendations scored, next to your actual points")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("auto", help="run `plan` only when the deadline is near (for cron)")
    p.add_argument("--within-days", type=float, default=2.0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_auto)
    return ap


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
