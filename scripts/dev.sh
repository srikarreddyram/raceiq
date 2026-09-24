#!/usr/bin/env bash
# Start the RaceIQ API and the Pit Wall frontend together, wired to each
# other, and stop both with one Ctrl-C.
#
# Usage:
#   scripts/dev.sh
#   API_PORT=8100 WEB_PORT=5200 scripts/dev.sh    # pick your own ports
#
# Why this exists: the two halves find each other through two separate
# settings, and each fails silently when they disagree.
#
#   1. The frontend calls the API at VITE_RACEIQ_API, defaulting to
#      http://127.0.0.1:8000. If something else already holds 8000 — a
#      stale uvicorn from an earlier session, which really happened — a
#      fresh API can't bind, and the page carries on talking to the OLD
#      code with no sign anything is wrong.
#   2. The API only accepts browser requests from the origins in
#      RACEIQ_CORS_ORIGINS, defaulting to port 5173. If 5173 is busy, Vite
#      quietly moves to 5174, every request is then blocked by CORS, and
#      the page reports "Could not reach the RaceIQ API" — pointing at the
#      wrong half.
#
# So this picks free ports for both, writes the API's address into
# frontend/.env.local (gitignored), tells the API exactly which frontend
# origin to allow, and starts Vite with --strictPort so it fails loudly
# instead of drifting off the port the API was told about.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

port_in_use() {
  # Something accepting connections on localhost:$1? Check IPv4 and IPv6:
  # Vite binds "localhost", which on macOS is ::1 only.
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null || (exec 3<>"/dev/tcp/::1/$1") 2>/dev/null
}

free_port_from() {
  local p=$1
  while port_in_use "$p"; do p=$((p + 1)); done
  echo "$p"
}

for tool in uv npm curl; do
  command -v "$tool" >/dev/null || { echo "dev.sh: '$tool' is not installed" >&2; exit 1; }
done

if [[ -n "${API_PORT:-}" ]] && port_in_use "$API_PORT"; then
  echo "dev.sh: API_PORT=$API_PORT is already in use" >&2; exit 1
fi
if [[ -n "${WEB_PORT:-}" ]] && port_in_use "$WEB_PORT"; then
  echo "dev.sh: WEB_PORT=$WEB_PORT is already in use" >&2; exit 1
fi
API_PORT="${API_PORT:-$(free_port_from 8000)}"
WEB_PORT="${WEB_PORT:-$(free_port_from 5173)}"
[[ "$API_PORT" == 8000 ]] || echo "dev.sh: port 8000 is busy — API on $API_PORT instead"
[[ "$WEB_PORT" == 5173 ]] || echo "dev.sh: port 5173 is busy — frontend on $WEB_PORT instead"

# The frontend reads this at startup (Vite loads .env.local automatically).
cat > frontend/.env.local <<EOF
# Written by scripts/dev.sh — overwritten on every run.
VITE_RACEIQ_API=http://127.0.0.1:$API_PORT
EOF

[[ -d frontend/node_modules ]] || (cd frontend && npm install)

# Each server runs in its own process group (set -m), so cleanup can stop
# a server together with the python/node processes `uv run` and `npm`
# spawn under it — and nothing else. (`kill 0` would also reach whatever
# launched this script when it isn't run from an interactive terminal.)
set -m
API_PID=""
WEB_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  echo
  echo "dev.sh: stopping API and frontend"
  for pid in $API_PID $WEB_PID; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "dev.sh: starting API on http://127.0.0.1:$API_PORT"
RACEIQ_CORS_ORIGINS="http://localhost:$WEB_PORT,http://127.0.0.1:$WEB_PORT" \
  uv run uvicorn serving.api.main:app --host 127.0.0.1 --port "$API_PORT" --reload &
API_PID=$!

# The API loads its models on import, which takes a few seconds; the
# frontend's first requests would fail if it came up first.
for _ in $(seq 1 90); do
  if curl -fs "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "dev.sh: the API exited during startup — see the error above" >&2; exit 1
  fi
  sleep 1
done
curl -fs "http://127.0.0.1:$API_PORT/health" >/dev/null || { echo "dev.sh: the API didn't come up within 90s" >&2; exit 1; }

echo "dev.sh: starting frontend on http://localhost:$WEB_PORT"
(cd frontend && npm run dev -- --port "$WEB_PORT" --strictPort) &
WEB_PID=$!

cat <<EOF

  RaceIQ is running
    Pit Wall   http://localhost:$WEB_PORT/pitwall
    Splash     http://localhost:$WEB_PORT/
    API docs   http://127.0.0.1:$API_PORT/docs

  Ctrl-C stops both.

EOF

# Return (and clean up) as soon as either server stops. A polling loop
# rather than `wait -n`: macOS ships bash 3.2, and `wait -n` needs 4.3.
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 1
done
echo "dev.sh: a server stopped — shutting the other down"
