#!/usr/bin/env bash
# Fetch whatever the current race weekend has produced so far — practice,
# qualifying, the race — and rebuild the warehouse, so the planner works
# from Saturday's real grid rather than a season-average one. Also catches
# up any earlier rounds that were never ingested. See
# ingestion/refresh_weekend.py for what it fetches and why the backfill
# can't do this.
#
# Usage:
#   scripts/refresh_weekend.sh              # whatever needs it
#   scripts/refresh_weekend.sh --round 16   # one round
#
# Restart scripts/dev.sh afterwards: the API caches plans in memory.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run python -m ingestion.refresh_weekend "$@"
