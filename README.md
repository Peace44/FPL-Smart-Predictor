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

To have it make the moves itself, see *Letting it act for you*.

Without an odds key the tool still works, using its own team ratings for every fixture.

## Commands

| Command | What it does |
|---|---|
| `fpl plan` | Transfers, line-up, bench order, captain and chip advice for the next deadline |
| `fpl team` | Your squad with selling prices and projected points |
| `fpl top --pos MID -n 20` | Best projected players (`--max-price`, `--team`, `--value`) |
| `fpl player Haaland` | One player's projection per fixture, split into goals, assists, clean sheets, bonus... |
| `fpl review` | The gameweek review in `reports/gwNN_review.md`: result against projection, progress towards first place, the limiting factor (`--gw N`; `--history` for the old table of past plans) |
| `fpl improve` | Read the latest final review and test changes to the settings behind its limiting factor; keep one only if it is clearly better (`--dry-run`, `--revert ID`) |
| `fpl evaluate` | Walk-forward accuracy of the points model; `--save NAME` / `--against NAME` compare two versions of the code |
| `fpl backtest --season 2025-26` | Replay the optimiser over a finished season |
| `fpl auto` | Unattended mode: plan when a deadline is near and, if allowed, update your team (for a scheduler) |
| `fpl apply` | Plan now and carry the plan out on your FPL account (`--dry-run` shows the requests without sending them) |
| `fpl login` | Check the FPL sign-in and show what FPL says about your team (reads only) |
| `fpl status` | What unattended mode did last, and the recent actions on your account |

Useful `fpl plan` options:

- `--ft N`: free transfers you have. The public API does not expose this, so it is
  reconstructed from your history; check the number in the report.
- `--wildcard GW`, `--freehit GW`, `--no-chips`: force or forbid chips.
- `--ban "Player"`, `--lock "Player"`: never buy / always keep a player (repeatable; add the
  club for namesakes, e.g. `"Martinez (AVL)"`).
- `--horizon N`, `--max-hits N`, `--sims N`, `--no-refresh`.

## How it works

1. **Data** (`fpl_api.py`, `history.py`, `market.py`, `odds.py`, `account.py`). The official FPL API for
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
- Free transfers are inferred, not read (see `--ft`), unless a sign-in is set up.
- Every run refits the team ratings, player rates and calibration on all matches played so far,
  and the minutes model is retrained after each gameweek, so the projections keep up with the
  season. The settings behind them change only through the weekly review loop (see *The weekly
  review and self-improvement*), one step at a time and only on clear evidence. Chip reserves are
  not tuned by it, because the replays leave chips out.
- It maximises expected points. It does not yet weigh ownership (differentials against
  template picks) when you are chasing or protecting rank.

## The weekly review and self-improvement

When a gameweek's data is final, the autopilot writes **`reports/gwNN_review.md`** (also copied to
`Documents\FPL` and announced on Discord). Run `fpl review` to write one yourself. A review written
before FPL has checked the data is marked provisional and is rewritten later. It covers:

- **Against the projection:** each player's projected and actual points, and where the team's gap
  came from (minutes, goals, assists, clean sheets, bonus...). It separates finishing luck (chances
  created but not scored) from the model's error (chances that were never there). It also gives the
  hindsight numbers: best eleven from the same 15, best captain, and what the squad from before the
  week's moves would have scored.
- **The model across all players:** bias and correlation against FPL's own projection, calibration of
  P(plays), regulars who did not play, the ten highest projections, and team expected goals against
  the chances teams created.
- **Towards first place:** points of first place and the top 100 / 1k / 10k / 100k / 1M, the gap and
  the points a week needed to close it, the season so far, and milestones. It also compares your week
  with the top 50 managers as they stood before the gameweek: their captains, the players most of
  them owned that you did not, and your differentials.
- **What can be improved:** the gap to the top 50 split into five parts that add up exactly: squad
  (best possible eleven), selection (left on the bench), captaincy, chips and hits. The part with the
  largest average gap over the last six engine-managed gameweeks is the **limiting factor**. An
  autopilot mishap that cost points comes first. Below that are plain observations, each tied to
  what could change it.
- **Your notes:** anything you write under that heading is kept when the review is rewritten, and
  the self-improvement skill reads it.

**`fpl improve`** takes the latest final review and works on its limiting factor. One week of
points is mostly luck, so the review only chooses where to look. Each candidate setting is one step
up or down for one knob in `tuning.KNOBS` (player and team ratings, availability, optimiser). It is
replayed walk-forward over the last two seasons and this one, gameweek by gameweek, against the
settings in effect. A setting is kept only if it beats the current value by three standard errors
without the guard measure or the current season getting worse. At most one setting changes per run.
Kept values go to `tuning/params.json`, the live code reads them from there, and `tuning/ledger.md`
records every run. `fpl improve --revert ID` undoes a change. The experiments behind each factor:

| Factor | Tested first | Then |
|---|---|---|
| squad | points model: accuracy of every likely starter | optimiser replay (about 30 minutes) |
| selection | points model | availability: P(plays) against who played, from the projections saved before each deadline (needs three gameweeks of them) |
| captaincy | points model at the top end: actual points of each week's ten best projections | |
| hits | optimiser replay | |
| chips, operations | (no setting covers them) | |

When no setting passes, the limit is in the code. The run then writes a brief for a code change in
`reports/gwNN_improvement.md`. The Claude Code skill **`/fpl-improve`**
(`.claude/skills/fpl-improve/SKILL.md`) works from that brief: it implements the most promising
untried idea on a branch, proves it with `fpl evaluate --against` and the tests, records the result in
the ledger, and asks before merging. When a review missed something, the skill also extends what the
review measures.

`scripts/auto.sh` runs `fpl improve --if-due` after each check. It works on a new final review
only, and only while the next deadline is more than 50 hours away. `FPL_SELF_IMPROVE=off` in the
environment of the scheduled task switches it off.

## Letting it act for you

By default the tool only writes reports. With a sign-in and `FPL_AUTOPILOT=on` it also makes the
transfers, plays the chips and sets the line-up, bench order and captain.

**1. Give it a sign-in (once).** FPL has no API keys; the tool uses the same refresh token your
browser holds after you log in.

1. Open a **private / incognito** window and log in at fantasy.premierleague.com.
2. Press F12, open the *Application* tab (Firefox: *Storage*), then *Local Storage* →
   `https://fantasy.premierleague.com`.
3. Find the key that starts with `oidc.user:` and copy its whole value (it starts with `{` and
   contains `"refresh_token"`).
4. Run `.venv/bin/fpl login --paste`, paste, then press Enter and Ctrl-D. It prints your bank,
   free transfers and chips if the sign-in works. (Or put just the `refresh_token` text in `.env`
   on one line as `FPL_REFRESH_TOKEN=...` and run `.venv/bin/fpl login`.)
5. Close the private window **without logging out**. The token is replaced every time it is
   used, so a browser session that keeps using the same one would cut the tool off (and logging
   out cancels it).

The live token is kept in `data/auth/token.json`, readable only by you and never committed.
It is valid for 180 days and every use renews it; unattended mode uses it about once a day, so
it should not lapse while the schedule keeps running. If it ever does, the tool says so (see
*Staying informed*) and you repeat the steps above.

**2. Switch it on.** In `.env`:

```
FPL_AUTOPILOT=on             # without this it signs in to read your exact squad, but changes nothing
FPL_APPLY_WITHIN_HOURS=30    # how long before the deadline it first acts (default 30)
FPL_AUTO_MAX_HITS=1          # most -4 hits it may take in one gameweek (default 1; 0 = never)
FPL_CHECK_EVERY_HOURS=2      # how often your scheduler runs the check (default 2; see Automation)
FPL_BAN=Player One,Player Two (AVL)     # never buy these
FPL_LOCK=Player Three                   # never sell these
```

**When it acts** (`autopilot.py`). Two days before a deadline it writes a plan. The first run
inside the last 30 hours carries the plan out, because nobody knows whether the computer will be
on later; it waits if the previous gameweek still has matches to play, unless the deadline is
under six hours away. After that the squad is settled: later runs skip the chip search, re-check
the line-up and captain, and happen when one of your players' injury flags changes, every twelve
hours, and once more in the last five hours.

**What stops it doing damage** (`execute.py`):

- It works from what your account looks like at that moment, not from what the plan assumed,
  and refuses if the plan is not legal under the game's rules, if prices have moved it out of
  budget, or if the transfers would cost more points than the plan expected.
- Unattended, it takes at most one -4 hit a gameweek (`FPL_AUTO_MAX_HITS`). The replays do not
  show hits paying for themselves, so this is a cap on risk, not a tuned value.
- A wildcard or free hit is played with a single transfer first, one that a free transfer covers
  anyway. Only when FPL then shows unlimited transfers are the rest sent. If the chip did not
  register, it stops having spent nothing.
- After every change it reads your team back and compares it with what it asked for. Anything
  unexpected stops the run and is reported; it never tries to repair a surprise by itself.
- It never starts within ten minutes of a deadline.

`fpl apply --dry-run` prints exactly what would be sent, without sending it.

**What you should know before relying on it:**

- FPL's sign-in and "my team" endpoints are undocumented and were written from how other
  community tools use them. Signing in, reading the team and saving a line-up have been checked
  against a real account (November 2026). Making a transfer and playing a chip have not: the
  first real one is the test, and the checks above are built so that a wrong guess ends in
  "nothing happened" plus a warning.
- A transfer cannot be undone, and a wildcard or free hit cannot be cancelled once played.
- Whether FPL's game terms allow automated play has not been checked: the Premier League's
  website terms do not mention it, and the game's own terms page could not be read by a script.
  Read them yourself if an account sanction would matter to you.
- If you change your team by hand, the next run plans from the new squad but may change your
  line-up and captain back. Set `FPL_AUTOPILOT=off` for a week you want to manage yourself.

## Automation

`scripts/auto.sh` runs `fpl auto` and logs to `reports/auto.log`. Trigger it as often as you
like: far from a deadline it exits in a few seconds (and, once a day, uses the sign-in so it does
not lapse); a second copy started while one is running exits at once. Near a deadline a full
run keeps ten solver processes busy for about nine minutes.

**Windows (the computer does not need WSL open).** In a Windows PowerShell window, once:

```powershell
Register-ScheduledTask -TaskName "FPL Smart Predictor" `
  -Action (New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu -- /home/peace/PROJECTS/Personal/FPL-Smart-Predictor/scripts/auto.sh") `
  -Trigger (New-ScheduledTaskTrigger -Once -At 00:10 -RepetitionInterval (New-TimeSpan -Hours 2)) `
  -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew)
```

It checks every two hours while you are logged in to Windows, and catches up on a missed check
when you next log in. A console window is visible while it runs. Remove it with
`Unregister-ScheduledTask -TaskName "FPL Smart Predictor"`. The computer still has to be on and
logged in at some point in the 30 hours before each deadline; if it is not, nothing happens that
week.

**Every 11 hours instead.** If a console window every two hours is too much:

```powershell
Set-ScheduledTask -TaskName "FPL Smart Predictor" -Trigger (New-ScheduledTaskTrigger -Once -At 20:00 -RepetitionInterval (New-TimeSpan -Hours 11))
```

and in `.env` set `FPL_CHECK_EVERY_HOURS=11` and `FPL_APPLY_WITHIN_HOURS=24`. An interval under
half the acting window always puts at least two checks inside it: the first carries the plan
out, and the last one that still leaves an hour before the deadline re-checks the line-up and
captain. Because 11 does not divide 24, the checks fall two hours earlier each day. One that
falls while the computer is off or asleep runs when you next log in, so both looks happen only
if the computer is on twice in those 24 hours; used once, it acts once.

**Once a day.** The fewest checks that still work: `-Trigger (New-ScheduledTaskTrigger -Daily
-At 20:00)` with `FPL_CHECK_EVERY_HOURS=24` and `FPL_APPLY_WITHIN_HOURS=24`. Exactly one check
falls inside the acting window (20:00 the evening before, at least 16 hours ahead of any
deadline), so it acts once and never looks again: an injury announced after that run is not
acted on.

Either way `scripts/auto.sh` retries a failed run twice, ten minutes apart, because the next
trigger may come too late.

**cron (only while WSL is running):**

```
10 */2 * * * /home/peace/PROJECTS/Personal/FPL-Smart-Predictor/scripts/auto.sh
```

### Staying informed

After every run that planned or changed something:

- `reports/status.md` says what it did and whether anything needs you; `reports/actions.log`
  lists every change made to your account; `reports/gwNN/applied-*.json` has the exact requests.
- On WSL the status and the latest plan are also copied to `Documents\FPL` on the Windows side
  (`FPL_WINDOWS_DIR` changes the folder, `off` disables it), and a Windows notification appears
  when your team was changed or something needs your attention, such as an expired sign-in.
- With `DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...` in `.env` (Discord: channel
  settings → Integrations → Webhooks), a report goes to that channel for:
  - every run that planned or acted: what it did, the plan, the line-up, anything that needs you;
  - a run that failed, and a sign-in that has expired;
  - a check with nothing to do yet, at most every six hours, so you can see the schedule is alive;
  - each finished gameweek: points, the game average, overall rank and how far it moved;
- each gameweek review, with its limiting factor, and each self-improvement run's result.

  `reports/auto.log` records whether each report was delivered. Anyone who has the webhook
  address can post to the channel, so keep it out of anything you share.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                       # about a minute, no network needed
.venv/bin/python scripts/eval_minutes.py         # walk-forward check of the minutes model
.venv/bin/python scripts/eval_xp.py              # walk-forward check of the points model
.venv/bin/python scripts/tune_optimiser.py       # compare optimiser settings on past seasons
```

`fpl evaluate --save before` and, after a code change, `fpl evaluate --against before` compare the
points model's walk-forward accuracy gameweek by gameweek (z-scores per season).

Layout: `src/fpl_smart/` is the package, `tests/` the test suite, `tuning/` the self-tuned settings
and their ledger (committed), `scripts/` evaluation and tuning scripts, `data/` caches (not
committed), `reports/` generated plans and reviews (not committed).
