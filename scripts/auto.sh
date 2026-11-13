#!/usr/bin/env bash
# Run from cron: writes a fresh plan to reports/ when the next deadline is close, otherwise exits at once.
#   0 8,20 * * * /path/to/FPL-Smart-Predictor/scripts/auto.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports
exec >> reports/auto.log 2>&1
echo "=== $(date -u '+%Y-%m-%d %H:%M UTC')"
.venv/bin/fpl auto --within-days "${FPL_WITHIN_DAYS:-2}" > /dev/null
