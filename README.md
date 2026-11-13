# FPL Smart Predictor

A decision engine for Fantasy Premier League. Each gameweek it tells you which transfers to
make, who to start, bench and captain, and when to play each chip, by projecting every
player's points and solving for the best plan over the coming weeks.

## Quick start

```bash
cd FPL-Smart-Predictor
.venv/bin/fpl plan            # full recommendation for the next deadline (about 9 minutes)
.venv/bin/fpl plan --quick    # skip the chip-timing search (about a minute)
```

The report is printed and saved to `reports/gwNN/plan-<time>.md` (latest copy: `reports/latest.md`).

Settings live in `.env` (not committed):

```
FPL_TEAM_ID=1234567          # the number in your FPL "Points" page URL
ODDS_API_KEY=...             # the-odds-api.com key (free tier: 500 credits a month; a run uses 2)
```

Without an odds key the tool still works, using its own team ratings for every fixture.

## Commands

| Command | What it does |
|---|---|
| `fpl plan` | Transfers, line-up, bench order, captain and chip advice for the next deadline |
| `fpl team` | Your squad with selling prices and projected points |
| `fpl top --pos MID -n 20` | Best projected players (`--max-price`, `--team`, `--value`) |
| `fpl player Haaland` | One player's projection per fixture, split into goals, assists, clean sheets, bonus... |
| `fpl review` | How past recommendations scored, next to your real score and the game average |
| `fpl backtest --season 2025-26` | Replay the optimiser over a finished season |
| `fpl auto` | Run `plan` only if the deadline is within two days (for cron) |

Useful `fpl plan` options:

- `--ft N`: free transfers you have. The public API does not expose this, so it is
  reconstructed from your history; check the number in the report.
- `--wildcard GW`, `--freehit GW`, `--no-chips`: force or forbid chips.
- `--ban "Player"`, `--lock "Player"`: never buy / always keep a player (repeatable; add the
  club for namesakes, e.g. `"Martinez (AVL)"`).
- `--horizon N`, `--max-hits N`, `--sims N`, `--no-refresh`.

## How it works

1. **Data** (`fpl_api.py`, `history.py`, `market.py`, `odds.py`). The official FPL API for
   the current season, your team and fixtures; the vaastav archive for four past seasons;
   football-data.co.uk for historical bookmaker odds; the-odds-api.com for upcoming odds.
   Everything is cached under `data/raw/`.
2. **Team strength** (`team_model.py`). Attack and defence ratings per club, fitted on a
   time-decayed blend of expected goals, actual goals and bookmaker-implied goals. For
   fixtures that have live odds, expected goals are 70% odds and 30% ratings.
3. **Minutes** (`minutes.py`). Gradient-boosted models for the chance a player appears, the
   chance he plays 60 minutes, and his expected minutes, from his recent pattern of starts.
   Live injury and suspension flags are applied on top (`xp.py`).
4. **Points** (`player_model.py`). Each player's share of his team's goals and assists
   (time-decayed, shrunk towards a prior for his position and price), combined with the
   fixture's expected goals and his minutes, then converted to FPL points component by
   component: appearance, goals, assists, clean sheets, goals conceded, saves, defensive
   contributions, bonus and cards.
5. **Optimiser** (`solver.py`). A mixed-integer programme (HiGHS) that chooses transfers,
   line-up, bench order, captain, bench boost and triple captain for every gameweek of the
   plan at once, with banked free transfers, -4 hits, selling prices and the three-per-club
   rule.
6. **Decisions** (`optimise.py`, `planner.py`, `report.py`). The wildcard and free hit weeks
   are searched from outside the solver (each candidate week is solved in parallel). The
   chosen plan is then re-solved many times with randomly shaken projections to show which
   of this week's moves hold up.

## How good is it

**Points model.** Trained only on earlier data and tested on each later season, its
projections correlate 0.37 (2024-25), 0.34 (2025-26) and 0.38 (2026-27 so far) with the
actual points of likely starters, and run about 0.1 points per player-match low
(`scripts/eval_xp.py`).

**Optimiser.** `fpl backtest` replays a season one gameweek at a time, using only what was
known before each deadline, starting from the engine's own GW1 squad. No chips are played
and no injury news is available in the archive, so read the totals as a comparison between
strategies, not as a forecast of your score.

| Season | Multi-week plan (default) | Next gameweek only | Never transfer |
|---|---|---|---|
| 2023-24 | 2,298 | 2,254 | 1,793 |
| 2024-25 | 2,396 | 2,437 | 2,186 |
| 2025-26 | 2,098 | 2,166 | 1,707 |
| 2026-27, GW1-10 | 601 | 614 | 631 |

- Over the three full seasons, making transfers added about 370 points a season over
  holding the opening squad (between 210 and 505).
- Planning several weeks ahead was **not** measurably better than looking one week ahead:
  the one-week version finished 78 points ahead over these 124 gameweeks, which is inside
  the noise (a single captain haul is 20 points). The same held across eleven settings of
  horizon, weighting and free-transfer value (`scripts/tune_optimiser.py`): none separated
  from the rest by more than about 1.5 standard errors.
- The multi-week plan is kept as the default because chip timing needs it and because the
  live tool, unlike the replay, knows injury flags and return dates. Its weights lean
  towards the near weeks (each week counts 75% of the one before).
- In GW1-10 of this season the replay scored 601 against a game average of 563; holding
  its opening squad would have scored 631.

Known limits:

- Projections run a few percent low overall and somewhat more for the very best players, so
  a premium captain can be slightly undervalued.
- Bookmaker odds usually cover only the next one or two gameweeks.
- The engine knows FPL's injury flags but not press-conference news or predicted line-ups.
- Free transfers are inferred, not read (see `--ft`).
- It maximises expected points. It does not yet weigh ownership (differentials against
  template picks) when you are chasing or protecting rank.

## Automation

`scripts/auto.sh` runs `fpl auto`, which does nothing unless the next deadline is within two
days, and logs to `reports/auto.log`. A crontab line that checks twice a day (08:00 and 20:00
local time; a run near the deadline keeps ten solver processes busy for about nine minutes):

```
0 8,20 * * * /home/peace/PROJECTS/Personal/FPL-Smart-Predictor/scripts/auto.sh
```

On WSL, cron only runs while the WSL instance is up.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                       # about a minute, no network needed
.venv/bin/python scripts/eval_minutes.py         # walk-forward check of the minutes model
.venv/bin/python scripts/eval_xp.py              # walk-forward check of the points model
.venv/bin/python scripts/tune_optimiser.py       # compare optimiser settings on past seasons
```

Layout: `src/fpl_smart/` is the package, `tests/` the test suite, `scripts/` evaluation and
tuning scripts, `data/` caches (not committed), `reports/` generated plans (not committed).
