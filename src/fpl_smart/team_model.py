"""Team strength model.

Each team has an attack rating and a defensive-weakness rating; expected goals in a
fixture are  exp(c + home_adv + attack[team] + weakness[opponent]).  Ratings are fitted
by time-decayed, ridge-penalised quasi-Poisson likelihood on a blend of xG and goals,
so a 3-0 win on 0.8 xG moves a team less than a 1-0 win on 3.0 xG.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class TeamParams:
    halflife_days: float = 100.0   # how fast old matches lose weight
    xg_weight: float = 0.85        # blend of xG vs actual goals in the target
    market_weight: float = 0.5     # blend of bookmaker-implied goals into the target (if present)
    odds_blend: float = 0.7        # weight on live odds vs ratings when a fixture has odds
    ridge: float = 1.5             # shrinkage of ratings towards their prior
    promoted_attack: float = -0.22  # prior for teams with no recent top-flight data
    promoted_weak: float = 0.20
    new_team_gap_days: float = 300.0
    max_age_days: float = 1100.0


@dataclass
class Ratings:
    c: float
    home: float
    attack: dict[int, float]
    weak: dict[int, float]
    asof: pd.Timestamp | None = None
    extra: dict = field(default_factory=dict)

    def lam(self, team: int, opp: int, home: bool) -> float:
        """Expected goals scored by `team` against `opp`."""
        return float(np.exp(self.c + (self.home if home else 0.0)
                            + self.attack.get(team, 0.0) + self.weak.get(opp, 0.0)))

    def predict(self, team, opp, home) -> np.ndarray:
        team, opp, home = np.asarray(team), np.asarray(opp), np.asarray(home, dtype=bool)
        a = np.array([self.attack.get(int(t), 0.0) for t in team])
        w = np.array([self.weak.get(int(o), 0.0) for o in opp])
        return np.exp(self.c + self.home * home + a + w)

    def table(self) -> pd.DataFrame:
        t = pd.DataFrame({"attack": self.attack, "weak": self.weak})
        t["net"] = t["attack"] - t["weak"]
        return t.sort_values("net", ascending=False)


def fit(tm: pd.DataFrame, asof, teams: list[int] | None = None, season_start=None,
        p: TeamParams | None = None) -> Ratings:
    """Fit ratings on team-match rows strictly before `asof`.

    `teams`: codes that must get a rating (current-season clubs), even with no data yet.
    `season_start`: clubs with no match in the `new_team_gap_days` before it are treated
    as newly promoted: old data dropped, prior set to the promoted prior.
    """
    p = p or TeamParams()
    asof = pd.Timestamp(asof)
    d = tm[(tm["kickoff"] < asof)]
    age = (asof - d["kickoff"]).dt.total_seconds() / 86400.0
    d = d[age <= p.max_age_days].copy()
    age = age[age <= p.max_age_days]
    codes = sorted(set(d["team_code"]) | set(d["opp_code"]) | set(teams or []))

    new = set()
    if season_start is not None and teams:
        ss = pd.Timestamp(season_start)
        before = d[d["kickoff"] < ss]
        gap = (ss - before.groupby("team_code")["kickoff"].max()).dt.total_seconds() / 86400.0
        for t in teams:
            if t not in gap.index or gap[t] > p.new_team_gap_days:
                new.add(t)
        if new:  # forget their stale history (a different squad in a different league era)
            stale = (d["kickoff"] < ss) & (d["team_code"].isin(new) | d["opp_code"].isin(new))
            d, age = d[~stale], age[~stale]

    idx = {c: i for i, c in enumerate(codes)}
    n = len(codes)
    ti = d["team_code"].map(idx).to_numpy()
    oi = d["opp_code"].map(idx).to_numpy()
    hm = d["home"].to_numpy(dtype=float)
    xw = np.where(d["has_xg"].to_numpy(), p.xg_weight, 0.0)
    y = xw * d["xg"].to_numpy() + (1 - xw) * d["gf"].to_numpy()
    w = 0.5 ** (age.to_numpy() / p.halflife_days)
    if "mkt_lam" in d and p.market_weight > 0:  # closing odds are a low-noise read on strength
        has = d["mkt_lam"].notna().to_numpy()
        y = np.where(has, p.market_weight * d["mkt_lam"].fillna(0).to_numpy() + (1 - p.market_weight) * y, y)

    a0 = np.array([p.promoted_attack if c in new else 0.0 for c in codes])
    w0 = np.array([p.promoted_weak if c in new else 0.0 for c in codes])

    def loss(theta):
        c, h, a, wk = theta[0], theta[1], theta[2:2 + n], theta[2 + n:]
        eta = c + h * hm + a[ti] + wk[oi]
        mu = np.exp(eta)
        ll = np.sum(w * (y * eta - mu))
        pen = p.ridge * (np.sum((a - a0) ** 2) + np.sum((wk - w0) ** 2))
        r = w * (y - mu)
        g = np.empty_like(theta)
        g[0] = r.sum()
        g[1] = (r * hm).sum()
        g[2:2 + n] = np.bincount(ti, r, n) - 2 * p.ridge * (a - a0)
        g[2 + n:] = np.bincount(oi, r, n) - 2 * p.ridge * (wk - w0)
        return -(ll - pen), -g

    theta0 = np.concatenate([[np.log(1.35), 0.15], a0, w0])
    if len(d) == 0:
        theta = theta0
    else:
        theta = minimize(loss, theta0, jac=True, method="L-BFGS-B").x
    return Ratings(c=float(theta[0]), home=float(theta[1]),
                   attack={c: float(theta[2 + i]) for c, i in idx.items()},
                   weak={c: float(theta[2 + n + i]) for c, i in idx.items()},
                   asof=asof, extra={"new": new, "n": len(d)})


def walk_forward(tm: pd.DataFrame, seasons: list[str], p: TeamParams | None = None) -> pd.DataFrame:
    """Out-of-sample expected goals for every team-match in `seasons`: ratings are refitted
    before each gameweek using only earlier matches. Adds `rat_lam`/`rat_lam_a` (ratings only)
    and `lam`/`lam_a` (blended with that match's odds when available, as in live use)."""
    p = p or TeamParams()
    out = []
    for s in seasons:
        ds = tm[tm["season"] == s]
        teams = sorted(ds["team_code"].unique())
        start = ds["kickoff"].min()
        for gw, dg in ds.groupby("gw"):
            r = fit(tm, dg["kickoff"].min(), teams, start, p)
            dg = dg.copy()
            dg["rat_lam"] = r.predict(dg["team_code"], dg["opp_code"], dg["home"])
            dg["rat_lam_a"] = r.predict(dg["opp_code"], dg["team_code"], ~dg["home"])
            dg["lam"], dg["lam_a"] = dg["rat_lam"], dg["rat_lam_a"]
            if "mkt_lam" in dg:
                has = dg["mkt_lam"].notna()
                b = p.odds_blend
                dg.loc[has, "lam"] = b * dg.loc[has, "mkt_lam"] + (1 - b) * dg.loc[has, "rat_lam"]
                dg.loc[has, "lam_a"] = b * dg.loc[has, "mkt_lam_a"] + (1 - b) * dg.loc[has, "rat_lam_a"]
            out.append(dg)
    return pd.concat(out).sort_index()


def poisson_deviance(y, mu) -> float:
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))
