"""Player model: turns history into per-player rates, then rates + fixture + minutes into
expected FPL points, component by component.

Rates are time-decayed and shrunk towards a prior set by position and price (empirical
Bayes), so a £4.5m defender with one lucky goal is not treated like a striker.
Attacking output is modelled as a *share of the team's goals while on the pitch*, so
team strength and fixture difficulty enter through the team model's expected goals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from . import history

LN2 = np.log(2.0)
T0 = pd.Timestamp("2022-07-01", tz="UTC")
PRICE_EDGES = np.array([0, 42, 47, 52, 57, 62, 67, 72, 77, 85, 95, 110, 130, 999])
# the parts of expected points; they sum to xp
COMPONENTS = ["app", "goals", "assists", "cs", "gc", "saves", "pensave", "defcon", "bonus", "cards"]

# stat -> (numerator column, exposure column)
STATS = {
    "xg": ("xg", "x_g"), "goal": ("goals", "x_gf"),
    "xa": ("xa", "x_g"), "ast": ("assists", "x_gf"),
    "dc": ("dc", "dc_e90"), "yc": ("yc", "e90x"), "sv": ("saves", "x_sv"),
    "bres": ("bres", "e90x"), "bps": ("bps", "e90x"),
}
SAVE_BASE, SAVE_SLOPE = 1.2, 1.1  # saves per 90 ~ base + slope * opponent xG


@dataclass
class PlayerParams:
    hl_days: float = 300.0     # half-life of a player's past output
    k_goal: float = 8.0        # prior strength, in units of team goals while on the pitch
    k_ast: float = 8.0
    xg_blend: float = 0.75     # weight on xG-based share vs actual-goal share
    xa_blend: float = 0.6
    k_dc: float = 4.0          # prior strength in 90s
    k_yc: float = 15.0
    k_sv: float = 15.0
    k_bres: float = 15.0
    k_bps: float = 8.0


@dataclass
class Calibration:
    priors: dict = field(default_factory=dict)      # stat -> pos -> (price points, ratios)
    bonus_coef: dict = field(default_factory=dict)  # pos -> coefficient dict (on actual match events)
    bonus2: dict = field(default_factory=dict)      # pos -> coefficients on *expected* events
    cs: tuple = (0.0, 1.0)                           # P(clean sheet) = exp(-(a + b * lam_against))
    dc: dict = field(default_factory=dict)           # pos -> (a, b) logistic on mean count per 90
    mult: dict = field(default_factory=dict)         # component -> pos -> multiplier
    flat: dict = field(default_factory=dict)         # rare events per 90 by pos
    team_sums: tuple = (0.97, 0.88)                  # player goals / assists per team goal


# ----------------------------------------------------------------------------- preparation
def prepare(ms: pd.DataFrame, team_wf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add exposures and team context. `ms` should be sorted by (code, kickoff) with column `i`
    (see minutes.add_state). `team_wf` supplies pre-match expected goals (lam, lam_a) if given."""
    tm = history.team_matches(ms)
    key = ["season", "fixture", "team_code"]
    cols = tm[key + ["xg", "xga", "gf", "ga"]].rename(columns={"xg": "txg", "xga": "txga", "gf": "tgf", "ga": "tga"})
    ms = ms.drop(columns=[c for c in ("txg", "txga", "tgf", "tga", "lam", "lam_a") if c in ms], errors="ignore")
    ms = ms.merge(cols, on=key, how="left")
    if team_wf is not None:
        ms = ms.merge(team_wf[key + ["lam", "lam_a"]], on=key, how="left")
    e90 = ms["minutes"] / 90.0
    hx = ms["has_xg"].astype(float)
    ms["e90"] = e90
    ms["e90x"] = e90 * hx
    ms["x_g"] = ms["txg"] * e90 * hx
    ms["x_gf"] = ms["tgf"] * e90 * hx
    ms["x_sv"] = (SAVE_BASE + SAVE_SLOPE * ms["txga"]) * e90 * hx * (ms["pos"] == 1)
    hd = (ms["has_defcon"] & (ms["pos"] > 1)).astype(float)
    ms["dc"] = ms["defcon"].fillna(0) * hd
    ms["dc_e90"] = e90 * hd
    ms["dc_hit"] = ((ms["dc"] >= np.where(ms["pos"] == 2, 10, 12)) & (hd > 0)).astype(float)
    ms["cs60"] = ((ms["cs"] > 0) & (ms["minutes"] >= 60)).astype(float)
    ms["days"] = (ms["kickoff"] - T0).dt.total_seconds() / 86400.0
    ms["gw_start"] = ms.groupby(["season", "gw"])["days"].transform("min")
    if "bres" not in ms:
        ms["bres"] = 0.0
    return ms


# ----------------------------------------------------------------------------- priors / bonus
def fit_priors(tr: pd.DataFrame) -> dict:
    """Pooled rate by position and price band for every stat."""
    out: dict = {}
    b = np.digitize(tr["value"].to_numpy(), PRICE_EDGES) - 1
    for stat, (num, den) in STATS.items():
        out[stat] = {}
        for pos in (1, 2, 3, 4):
            mk = (tr["pos"] == pos).to_numpy()
            n_all, d_all = tr.loc[mk, num].sum(), tr.loc[mk, den].sum()
            base = n_all / d_all if d_all > 0 else 0.0
            xs, ys = [], []
            for k in range(len(PRICE_EDGES) - 1):
                mb = mk & (b == k)
                d = tr.loc[mb, den].sum()
                if d <= 0:
                    continue
                w = tr.loc[mb, den].to_numpy()
                xs.append(float(np.average(tr.loc[mb, "value"], weights=w)))
                c = 0.05 * d_all / max(1, len(PRICE_EDGES) - 1)   # light pooling towards position mean
                ys.append(float((tr.loc[mb, num].sum() + c * base) / (d + c)))
            out[stat][pos] = (np.array(xs), np.array(ys)) if xs else (np.array([50.0]), np.array([base]))
    return out


def prior(priors: dict, stat: str, pos, value) -> np.ndarray:
    pos, value = np.asarray(pos), np.asarray(value, dtype=float)
    res = np.zeros(len(pos))
    for p in (1, 2, 3, 4):
        mk = pos == p
        if mk.any():
            xs, ys = priors[stat][p]
            res[mk] = np.interp(value[mk], xs, ys)
    return res


BONUS_X = ["goals", "assists", "cs60", "saves", "gc", "p60", "e90", "dc_hit"]


def fit_bonus(tr: pd.DataFrame) -> dict:
    """Per-position linear model of bonus points on match events (played rows only)."""
    coef = {}
    d = tr[(tr["minutes"] > 0) & tr["has_xg"]]
    for pos in (1, 2, 3, 4):
        x = d[d["pos"] == pos]
        X = x[BONUS_X].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ x["bonus"].to_numpy(dtype=float), rcond=None)
        coef[pos] = dict(zip(BONUS_X, beta))
    return coef


def bonus_fitted(d: pd.DataFrame, coef: dict) -> np.ndarray:
    out = np.zeros(len(d))
    for pos, c in coef.items():
        mk = (d["pos"] == pos).to_numpy()
        out[mk] = sum(c[k] * d.loc[mk, k].to_numpy(dtype=float) for k in BONUS_X)
    return out * (d["minutes"].to_numpy() > 0)


# ----------------------------------------------------------------------------- decayed state
def _weighted_cumsums(ms: pd.DataFrame, hl_days: float) -> pd.DataFrame:
    a = LN2 / hl_days
    w = np.exp(a * ms["days"].to_numpy())
    cols = sorted({c for pair in STATS.values() for c in pair})
    tmp = pd.DataFrame({c: ms[c].to_numpy(dtype=float) * w for c in cols})
    tmp["code"] = ms["code"].to_numpy()
    cs = tmp.groupby("code")[cols].cumsum()
    cs.index = ms.index
    return cs


def state_backtest(ms: pd.DataFrame, hl_days: float) -> pd.DataFrame:
    """Decayed sums available at each row's gameweek deadline (rows are aligned with ms)."""
    a = LN2 / hl_days
    cs = _weighted_cumsums(ms, hl_days)
    cs["code"], cs["i"] = ms["code"].to_numpy(), ms["i"].to_numpy()
    k = ms.groupby(["code", "season", "gw"]).cumcount() + 1
    look = pd.DataFrame({"code": ms["code"].to_numpy(), "i": (ms["i"] - k).to_numpy()}, index=ms.index)
    S = look.reset_index().merge(cs, on=["code", "i"], how="left").set_index("index").drop(columns=["code", "i"])
    S = S.reindex(ms.index).fillna(0.0)
    return S.mul(np.exp(-a * ms["gw_start"].to_numpy()), axis=0)


def state_asof(ms: pd.DataFrame, when: pd.Timestamp, hl_days: float) -> pd.DataFrame:
    """Decayed sums per player code using every row before `when` (for live prediction)."""
    a = LN2 / hl_days
    d = ms[ms["kickoff"] < when]
    cs = _weighted_cumsums(d, hl_days)
    last = d.groupby("code").tail(1)
    S = cs.loc[last.index]
    S.index = last["code"].to_numpy()
    return S * np.exp(-a * (when - T0).total_seconds() / 86400.0)


# ----------------------------------------------------------------------------- rates
def rates(S: pd.DataFrame, pos, value, cal: Calibration, p: PlayerParams) -> pd.DataFrame:
    pr = lambda stat: prior(cal.priors, stat, pos, value)  # noqa: E731
    R = pd.DataFrame(index=S.index)

    def shrunk(stat, k):
        num, den = STATS[stat]
        return (S[num].to_numpy() + k * pr(stat)) / (S[den].to_numpy() + k)

    def parts(stat, k):
        """(evidence part, prior part) of a shrunk rate; they add up to shrunk(stat, k)."""
        num, den = STATS[stat]
        d = S[den].to_numpy() + k
        return S[num].to_numpy() / d, k * pr(stat) / d

    p_xg, p_goal, p_xa, p_ast = pr("xg"), pr("goal"), pr("xa"), pr("ast")
    conv_g = np.divide(p_goal, p_xg, out=np.ones_like(p_goal), where=p_xg > 0)
    conv_a = np.divide(p_ast, p_xa, out=np.ones_like(p_ast), where=p_xa > 0)
    (xg_e, xg_p), (go_e, go_p) = parts("xg", p.k_goal), parts("goal", p.k_goal)
    (xa_e, xa_p), (as_e, as_p) = parts("xa", p.k_ast), parts("ast", p.k_ast)
    R["g_ev"] = p.xg_blend * xg_e * conv_g + (1 - p.xg_blend) * go_e
    R["g_pr"] = p.xg_blend * xg_p * conv_g + (1 - p.xg_blend) * go_p
    R["a_ev"] = p.xa_blend * xa_e * conv_a + (1 - p.xa_blend) * as_e
    R["a_pr"] = p.xa_blend * xa_p * conv_a + (1 - p.xa_blend) * as_p
    R["g_share"] = R["g_ev"] + R["g_pr"]
    R["a_share"] = R["a_ev"] + R["a_pr"]
    R["dc_mu"] = shrunk("dc", p.k_dc)
    R["yc90"] = shrunk("yc", p.k_yc)
    R["sv_mult"] = shrunk("sv", p.k_sv)          # saves per unit of expected save load (prior ~1)
    R["bres90"] = S["bres"].to_numpy() / (S["e90x"].to_numpy() + p.k_bres)
    R["bps90"] = shrunk("bps", p.k_bps)
    return R


# ----------------------------------------------------------------------------- expected points
def _e_floor_div(mu: np.ndarray, k: int) -> np.ndarray:
    """E[floor(N / k)] for N ~ Poisson(mu)."""
    mu = np.clip(np.asarray(mu, dtype=float), 0, 12)
    n = np.arange(0, 31)
    pmf = poisson.pmf(n[None, :], mu[:, None])
    return (pmf * (n // k)[None, :]).sum(axis=1)


def p_clean_sheet(lam_a, cal: Calibration) -> np.ndarray:
    a, b = cal.cs
    return np.exp(-(a + b * np.asarray(lam_a, dtype=float)))


def bonus_features(out: pd.DataFrame, R: pd.DataFrame, x_saves, p60, e90) -> np.ndarray:
    xg, xa = out["x_goals"].to_numpy(), out["x_assists"].to_numpy()
    return np.column_stack([
        xg, xa, out["p_cs"].to_numpy(), np.asarray(x_saves),
        np.asarray(p60), np.asarray(e90), R["bres90"].to_numpy() * e90, R["bps90"].to_numpy() * e90,
        (xg + 0.5 * xa) ** 2,   # bonus is convex in attacking threat: braces usually mean 3 bonus
    ])


DEFAULT_SCORING = {
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
    "assists": 3, "saves": 1, "penalties_saved": 5, "penalties_missed": -2,
    "yellow_cards": -1, "red_cards": -3, "own_goals": -2, "bonus": 1, "long_play": 2, "short_play": 1,
}
_POSN = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _by_pos(pos: np.ndarray, table: dict) -> np.ndarray:
    return np.array([table[_POSN[int(x)]] for x in pos], dtype=float)


def expected_points(R: pd.DataFrame, pos, lam, lam_a, p_play, p60, emin, cal: Calibration,
                    scoring: dict | None = None, defcon: bool = True) -> pd.DataFrame:
    """Expected points per component for one fixture. All inputs are aligned arrays."""
    sc = scoring or DEFAULT_SCORING
    pos = np.asarray(pos)
    lam, lam_a = np.asarray(lam, float), np.asarray(lam_a, float)
    p_play, p60, emin = np.asarray(p_play, float), np.asarray(p60, float), np.asarray(emin, float)
    e90 = emin / 90.0
    on = np.divide(e90, p_play, out=np.zeros_like(e90), where=p_play > 1e-6)  # 90s played, given he plays
    m = lambda comp: np.array([cal.mult.get(comp, {}).get(int(x), 1.0) for x in pos])  # noqa: E731
    fl = lambda ev: np.array([cal.flat.get(ev, {}).get(int(x), 0.0) for x in pos])  # noqa: E731
    out = pd.DataFrame(index=R.index)
    out["x_goals"] = R["g_share"].to_numpy() * lam * e90 * m("goals")
    out["x_assists"] = R["a_share"].to_numpy() * lam * e90 * m("assists")
    pcs = p_clean_sheet(lam_a, cal)
    out["p_cs"] = p60 * pcs * m("cs")
    out["app"] = sc["short_play"] * p_play + (sc["long_play"] - sc["short_play"]) * p60
    out["goals"] = out["x_goals"] * _by_pos(pos, sc["goals_scored"])
    out["assists"] = out["x_assists"] * sc["assists"]
    out["cs"] = out["p_cs"] * _by_pos(pos, sc["clean_sheets"])
    out["gc"] = p_play * _e_floor_div(lam_a * on, 2) * _by_pos(pos, sc["goals_conceded"]) * m("gc")
    is_gk = pos == 1
    x_saves = R["sv_mult"].to_numpy() * (SAVE_BASE + SAVE_SLOPE * lam_a) * on * is_gk
    out["saves"] = p_play * _e_floor_div(x_saves, 3) * sc["saves"] * m("saves")
    out["pensave"] = fl("ps") * e90 * (lam_a / 1.4) * sc["penalties_saved"]
    if defcon:
        a = np.array([cal.dc.get(int(x), (-9.0, 0.0))[0] for x in pos])
        b = np.array([cal.dc.get(int(x), (-9.0, 0.0))[1] for x in pos])
        out["defcon"] = p60 * (1 / (1 + np.exp(-(a + b * R["dc_mu"].to_numpy())))) * _by_pos(pos, sc["defensive_contribution"])
    else:
        out["defcon"] = 0.0
    # bonus: linear in expected events and the player's own bonus-point habits
    F = bonus_features(out, R, x_saves * p_play, p60, e90)
    bon = np.zeros(len(pos))
    if cal.bonus2:
        for ps, beta in cal.bonus2.items():
            k = pos == ps
            if k.any():
                bon[k] = F[k] @ beta
    else:  # before calibration: fall back to the model fitted on actual match events
        for ps, c in cal.bonus_coef.items():
            k = pos == ps
            if k.any():
                bon[k] = (c["goals"] * F[k, 0] + c["assists"] * F[k, 1] + c["cs60"] * F[k, 2]
                          + c["saves"] * F[k, 3] + c["gc"] * lam_a[k] * e90[k] + c["p60"] * p60[k]
                          + c["e90"] * e90[k] + c["dc_hit"] * (out["defcon"].to_numpy()[k] / 2.0) + F[k, 6])
    out["bonus"] = np.clip(bon, 0, 3) * sc["bonus"]
    out["cards"] = (sc["yellow_cards"] * R["yc90"].to_numpy() + sc["red_cards"] * fl("rc")
                    + sc["own_goals"] * fl("og") + sc["penalties_missed"] * fl("pm")) * e90
    out["xp"] = out[COMPONENTS].sum(axis=1)
    return out


def expected_points_team(R: pd.DataFrame, team_key, pos, lam, lam_a, p_play, p60, emin, cal: Calibration,
                         scoring: dict | None = None, defcon: bool = True) -> pd.DataFrame:
    """expected_points with each team's goal and assist shares balanced so they add up to the team's
    expected goals. If a talisman is out, his goals go to whoever is on the pitch instead of vanishing.

    Only the *prior* part of each share is rescaled, so players with a long record keep their
    evidence-based share and unproven ones absorb the adjustment.
    `team_key` identifies the (team, fixture) of every row."""
    key = pd.Series(list(team_key) if not isinstance(team_key, pd.Series) else team_key.to_numpy(), index=R.index)
    posa = np.asarray(pos)
    w = np.asarray(emin, float) / 90.0
    R = R.copy()
    for share, ev, pr_, comp, target in (("g_share", "g_ev", "g_pr", "goals", cal.team_sums[0]),
                                         ("a_share", "a_ev", "a_pr", "assists", cal.team_sums[1])):
        mult = np.array([cal.mult.get(comp, {}).get(int(x), 1.0) for x in posa])
        ww = pd.Series(w * mult, index=R.index)
        s_ev = (ww * R[ev]).groupby(key).transform("sum").to_numpy()
        s_pr = (ww * R[pr_]).groupby(key).transform("sum").to_numpy()
        c = np.clip((target - s_ev) / np.where(s_pr > 1e-6, s_pr, 1.0), 0.3, 3.0)
        R[share] = R[ev].to_numpy() + c * R[pr_].to_numpy()
        tot = (ww * R[share]).groupby(key).transform("sum").to_numpy()
        # mop up what the prior could not absorb
        R[share] = R[share].to_numpy() * np.clip(target / np.where(tot > 0.05, tot, target), 0.9, 1.1)
    return expected_points(R, pos, lam, lam_a, p_play, p60, emin, cal, scoring, defcon).assign(
        g_share=R["g_share"], a_share=R["a_share"])


# ----------------------------------------------------------------------------- calibration
def fit_calibration(ms: pd.DataFrame, train_mask: np.ndarray, p: PlayerParams) -> tuple[Calibration, pd.DataFrame]:
    """Fit priors, bonus model and calibration constants on `train_mask` rows (which need lam, lam_a
    for the multipliers; rows without them only inform priors and the bonus model).
    Also returns `ms` with the bonus-residual column filled, as later states depend on it."""
    cal = Calibration()
    tr = ms[train_mask]
    cal.bonus_coef = fit_bonus(tr)
    ms = ms.copy()
    ms["bres"] = (ms["bonus"] - bonus_fitted(ms, cal.bonus_coef)) * (ms["minutes"] > 0) * ms["has_xg"]
    tr = ms[train_mask]
    cal.priors = fit_priors(tr)
    played = tr[(tr["minutes"] > 0) & tr["has_xg"]]
    e = played.groupby("pos")["minutes"].sum() / 90.0
    cal.flat = {ev: (played.groupby("pos")[ev].sum() / e).to_dict() for ev in ("rc", "og", "pm", "ps")}
    tgf = played.drop_duplicates(["season", "fixture", "team_code"])["tgf"].sum()
    if tgf > 0:
        cal.team_sums = (float(played["goals"].sum() / tgf), float(played["assists"].sum() / tgf))

    # everything below is calibrated conditional on the minutes actually played
    S = state_backtest(ms, p.hl_days)
    c = tr[tr["lam"].notna() & tr["has_xg"]] if "lam" in tr else tr.iloc[0:0]
    if len(c):
        # clean sheet curve from team-level outcomes
        t = c.drop_duplicates(["season", "fixture", "team_code"])
        y, la = (t["tga"] == 0).to_numpy(float), t["lam_a"].to_numpy(float)

        # one-parameter curve: goals are a little over-dispersed vs Poisson, so P(0) = exp(-(a + lam))
        def nll(a):
            pr_ = np.clip(np.exp(-(a + la)), 1e-6, 1 - 1e-6)
            return -np.sum(y * np.log(pr_) + (1 - y) * np.log(1 - pr_))
        cal.cs = (float(minimize_scalar(nll, bounds=(-0.3, 0.3), method="bounded").x), 1.0)
        # defensive-contribution logistic
        R = rates(S.loc[c.index], c["pos"], c["value"], cal, p)
        d = c[(c["dc_e90"] > 0) & (c["minutes"] >= 60)]
        for group, members in ((2, (2,)), (3, (3, 4))):
            x = d[d["pos"].isin(members)]
            if len(x) < 200:
                continue
            mu, hit = R.loc[x.index, "dc_mu"].to_numpy(), x["dc_hit"].to_numpy()

            def nll2(th):
                q = np.clip(1 / (1 + np.exp(-(th[0] + th[1] * mu))), 1e-6, 1 - 1e-6)
                return -np.sum(hit * np.log(q) + (1 - hit) * np.log(1 - q))
            ab = tuple(minimize(nll2, [-5.0, 0.5], method="Nelder-Mead").x)
            for ps in members:
                cal.dc[ps] = ab
        # multiplicative calibration of each component given actual minutes
        played_flag = (c["minutes"] > 0).to_numpy(float)
        p60 = (c["minutes"] >= 60).to_numpy(float)
        E = expected_points(R, c["pos"], c["lam"], c["lam_a"], played_flag, p60, c["minutes"], cal,
                            defcon=bool((c["dc_e90"] > 0).any()))
        actual = {
            "goals": c["goals"], "assists": c["assists"], "cs": c["cs60"],
            "gc": -(c["gc"] // 2) * (c["pos"] <= 2), "saves": c["saves"] // 3,
        }
        pred = {"goals": E["x_goals"], "assists": E["x_assists"], "cs": E["p_cs"], "gc": E["gc"],
                "saves": E["saves"]}
        for comp in actual:
            cal.mult[comp] = {}
            for ps in (1, 2, 3, 4):
                k = (c["pos"] == ps).to_numpy()
                a_, p_ = float(np.sum(actual[comp].to_numpy()[k])), float(np.sum(pred[comp].to_numpy()[k]))
                cal.mult[comp][ps] = a_ / p_ if abs(p_) > 1e-9 and abs(a_) > 0 else 1.0
        # bonus: regress actual bonus on the *expected* events (now calibrated), per position
        E = expected_points(R, c["pos"], c["lam"], c["lam_a"], played_flag, p60, c["minutes"], cal,
                            defcon=bool((c["dc_e90"] > 0).any()))
        e90 = c["minutes"].to_numpy(float) / 90.0
        is_gk = (c["pos"] == 1).to_numpy()
        on = np.where(played_flag > 0, e90, 0.0)
        x_saves = R["sv_mult"].to_numpy() * (SAVE_BASE + SAVE_SLOPE * c["lam_a"].to_numpy(float)) * on * is_gk
        F = bonus_features(E, R, x_saves, p60, e90)
        y = c["bonus"].to_numpy(float)
        for ps in (1, 2, 3, 4):
            k = ((c["pos"] == ps) & (c["minutes"] > 0)).to_numpy()
            if k.sum() < 500:
                continue
            A = F[k]
            cal.bonus2[ps] = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ y[k])
    return cal, ms
