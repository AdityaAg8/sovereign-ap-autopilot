#!/usr/bin/env bash
# Nightly audit verifier job (PRD §6.4, §10 M10.4): runs `python -m apx.audit verify`
# (hash-chain + head-anchor check, apx/audit/anchor.py) via `uv run`, tees the structured
# JSON result plus a timestamped human summary into logs/audit-verify/, and exits with the
# CLI's own exit code (0 ok, 1 chain/anchor mismatch, 2 could not run) so cron, systemd or a
# monitoring wrapper can alert on failure. Only depends on `uv` and `DATABASE_URL`/
# `AUDIT_ANCHOR_FILE` already being set the way the running app is configured -- no Temporal
# client exists yet in this repo (apx/workflows has only Protocol interfaces), so this host
# cron entry is the fit-for-purpose "compose-friendly" scheduling path called out in PRD §10
# M10.4; once a Temporal worker exists, wire the same `python -m apx.audit verify` command
# into a Temporal Schedule instead of cron without changing anything in apx/audit/.
#
# Suggested crontab entry (nightly at 02:00, matching PRD §6.4 "walks the chain nightly"):
#   0 2 * * * /usr/bin/env bash /path/to/repo/scripts/audit_verify_cron.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/audit-verify"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +'%Y%m%dT%H%M%SZ')"
LOG_FILE="$LOG_DIR/$STAMP.log"

echo "audit_verify_cron: running python -m apx.audit verify (UTC $STAMP), logging to $LOG_FILE"

set +e
uv run python -m apx.audit verify >"$LOG_FILE" 2>&1
status=$?
set -e

cat "$LOG_FILE"

if [ "$status" -eq 0 ]; then
  echo "audit_verify_cron: OK"
else
  echo "audit_verify_cron: FAILED (exit $status) -- see $LOG_FILE" >&2
fi

exit "$status"
