#!/usr/bin/env bash
# Unattended run: plans when a deadline is near and, if FPL_AUTOPILOT=on and a sign-in is set up,
# carries the plan out. Safe to trigger as often as you like: most runs exit in a few seconds,
# and a second copy started while one is running exits at once.
#   Windows Task Scheduler:  wsl.exe -d Ubuntu -- /path/to/FPL-Smart-Predictor/scripts/auto.sh
#   cron:                    0 */2 * * * /path/to/FPL-Smart-Predictor/scripts/auto.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
exec 9> reports/.auto.lock
flock -n 9 || exit 0
exec >> reports/auto.log 2>&1
echo "=== $(date -u '+%Y-%m-%d %H:%M UTC')"
# A run that fails (no network yet, FPL briefly down) is tried again: on a once-a-day schedule
# the next trigger may come after the deadline.
tries="${FPL_AUTO_TRIES:-3}"
for attempt in $(seq 1 "$tries"); do
  if .venv/bin/fpl auto > /dev/null; then
    break
  fi
  echo "run ended with an error (attempt $attempt of $tries)"
  if [ "$attempt" -lt "$tries" ]; then
    sleep "${FPL_AUTO_RETRY_SECONDS:-600}"
  fi
done
# After a gameweek's review is final, and while no deadline is close, test one-step changes to every
# setting (fpl improve). When none passes, Claude Code works on the code-change brief it wrote
# (fpl improve --code: /fpl-improve unattended, on a branch in its own worktree, never merged), and
# stops before the planning window opens. Both run inside the lock, so no plan runs meanwhile, and
# report their own result or failure to Discord. FPL_SELF_IMPROVE=off switches both off;
# FPL_SELF_IMPROVE_CODE=off only the second.
if [ "${FPL_SELF_IMPROVE:-on}" != "off" ]; then
  .venv/bin/fpl improve --if-due > /dev/null || echo "self-improvement run ended with an error"
  .venv/bin/fpl improve --code --if-due > /dev/null || echo "code-change run ended with an error"
fi
