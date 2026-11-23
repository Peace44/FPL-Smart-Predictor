---
name: fpl-improve
description: Weekly self-improvement loop for FPL Smart Predictor. Reads the latest finished gameweek review (reports/gwNN_review.md), runs `fpl improve` to test settings behind the review's limiting factor, and when no setting helps, implements and validates a code change aimed at that factor on a branch. Use when the user asks to improve the predictor from a review, to address the limiting factor, to act on a gameweek review, or runs /fpl-improve.
---

# FPL self-improvement loop

Goal of the project: finish first in the world. Each finished gameweek gets a review; this skill turns
the review into one improvement aimed at whatever limits the team most, and proves the change helps
before keeping it. Work from evidence, one change at a time.

## 1. Read the review

- Use the review the user names. Otherwise use the newest `reports/gw*_review.md` whose embedded data
  (the `<!-- fpl-review-data ... -->` block at the end) has `"final": true`. If only a provisional one
  exists, say so. Work from it only if the user wants that.
- Read all of it: the limiting factor, the "What can be improved" table (each part's average ± its
  standard error, and whether the factor is settled), the model errors pooled over the season, the
  observations, the model check, and the **"Your notes"** section. The user's notes outrank the automatic diagnosis
  when they disagree. Say which one you followed.
- Read `tuning/ledger.md` so you do not repeat a change that was already tried and rejected.

## 2. Settings first: `fpl improve`

- If `reports/gwNN_improvement.md` already exists for this review (the scheduler runs
  `fpl improve --if-due` on its own), read it. Otherwise run `.venv/bin/fpl improve <review>`. It
  takes 5 to 10 minutes, or about 30 if it replays the optimiser; use `--light` to skip that replay.
- If it changed a setting: report the knob, old and new values, and the evidence (z-scores). Run
  `.venv/bin/python -m pytest -q`. Offer to commit `tuning/params.json` and `tuning/ledger.md`.
  Then stop: one change per review.
- If no setting passed, the limit is structural. Continue with step 3 using its
  "Needs a code change" brief.

## 3. A code change for the limiting factor

1. **Choose.** The brief's bold line says what aims the change this week. Skip anything already
   tried (ledger lines starting `- Code change:`).
   - **A row marked *act*** in its evidence table (a model error pooled over three or more
     gameweeks, at least 3 standard errors, big enough to matter): take the first one. Its "Where in
     the code" column says where to look. First check that the replays show the same error on the
     current code (`fpl evaluate`); a live error the replays do not show is probably a run of luck.
   - **No *act* row, but the factor is settled:** take the factor's idea most likely to move it.
   - **Neither:** say that no change is well aimed this week, name the strongest *watch* rows, and
     stop unless the user wants to go ahead. If they do, say the change is a guess.
   A bug visible in the review (a player projected for a team he left, minutes that ignore a flag)
   comes before all of these: one instance is enough evidence. Say which case applied and why.
2. **Baseline.** On the current code: `.venv/bin/fpl evaluate --save before-gwNN`
   (walk-forward points-model metrics by gameweek, about 30 seconds).
3. **Branch.** `git checkout -b improve/gwNN-<short-slug>`. Never work on `main`: the autopilot runs
   from the working tree, so check out `main` again when you finish unless the user says otherwise.
4. **Implement** the smallest change that tests the idea. Keep the code style of the module you
   touch. If the idea adds a tunable number, add a `Knob` for it in `src/fpl_smart/tuning.py` and
   read it through `tuning.*` so later `fpl improve` runs can tune it.
5. **Validate**, all of it:
   - `.venv/bin/python -m pytest -q` passes. Add a test for the new behaviour.
   - Points-model changes: `.venv/bin/fpl evaluate --against before-gwNN`. Keep the change only if
     the target measure improves (z ≥ 2 for a change you designed from evidence; there is no
     multiple-testing correction because you test one idea). Overall accuracy (`mse_rel`) and the
     current season must not get clearly worse (z > -1).
   - Planner or captaincy changes: `.venv/bin/fpl backtest --season 2025-26` and the current season
     before and after (slow; run in the background). Report the totals with the noise caveat from
     the README.
   - Autopilot or operations fixes: a test in `tests/test_autopilot.py` that reproduces the incident
     with `FakeFPL` and passes after the fix.
   - Diagnosis changes (step 4): `.venv/bin/fpl review --gw NN` still writes the review.
6. **Record** in `tuning/ledger.md`, under the gameweek's entry:
   `- Code change: <what> (<branch>): <evidence, e.g. top10 +0.06 pts/GW, z 2.4>; kept|rejected`.
   Record rejected ideas too, so they are not tried again.
7. **Commit** on the branch with a message that states the factor and the evidence. Show the user
   the diff summary and the numbers, and ask before merging into `main`.

If the change does not pass validation, revert it on the branch, record it as rejected, and try the
next idea only if the user wants another round.

## 4. Improve the loop itself

The loop is only as good as what the review measures. If the user's notes or your investigation find
a failure the review did not show (e.g. a hit taken for a player who was already injured, or a
captain picked against 90% of the top 50 every week), extend `src/fpl_smart/gwreview.py` so every
future review measures it. Add an observation, a field in the embedded data, or a new factor with
its `LEVERS` and `improve.PLAN`/`STRUCTURAL` entries. Add a test in `tests/test_review.py`. A new
factor must keep the parts adding up to the gap with the top 50, or be reported beside them.

## Ground rules

- Never send anything to the FPL account from this skill. `fpl apply` and `fpl auto` are not part
  of it.
- Never weaken the checks in `execute.py`, the hit cap, or the adoption thresholds in `improve.py`
  to make a change pass.
- One week of points is mostly luck. A single review chooses where to look; the walk-forward tests
  decide what to keep. An unsettled limiting factor is not a reason to change code.
- Report outcomes plainly: what was tried, the numbers, and what was kept.
