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
