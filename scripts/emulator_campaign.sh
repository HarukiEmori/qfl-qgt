#!/bin/bash
# Runs one campaign of scripts/run_emulator.py on the emulator until it has finished.
#
#     scripts/emulator_campaign.sh <name> [options of run_emulator.py]
#
# A campaign needs one job submission per round, so a process has to stay alive for hours.  This
# script starts run_emulator.py for the campaign <name> and starts it again whenever it stops before
# the end (a lost connection, a sleep of the computer, an error of the service).  Every start reads
# the finished rounds from the cache and re-attaches to the jobs that are already submitted, so that
# nothing is submitted twice.  The output is appended to results/<name>/campaign.log.
#
# Environment variables
#     PYTHON        interpreter (default: python)
#     DEVICE        Nexus device (default: reimei-E)
#     MAX_RETRIES   restarts before the script gives up (default: 200)
#     RETRY_DELAY   seconds between a stop and the next start (default: 90)
#
# On macOS the computer is kept awake while the script runs.  A closed lid still puts a laptop to
# sleep; the campaign then pauses and continues after the computer wakes up.
# Progress and completion:  python scripts/emulator_status.py --name <name>

set -u
NAME=${1:?usage: scripts/emulator_campaign.sh <name> [options of run_emulator.py]}
shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1
PY=${PYTHON:-python}
DEVICE=${DEVICE:-reimei-E}
MAX_RETRIES=${MAX_RETRIES:-200}
RETRY_DELAY=${RETRY_DELAY:-90}
OUT="results/$NAME"
LOG="$OUT/campaign.log"
PIDFILE="$OUT/campaign.pid"
mkdir -p "$OUT"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "campaign $NAME is already running (process $(cat "$PIDFILE"))"
  exit 2
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -ims -w $$ &
fi

say() { echo "[campaign $NAME] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

n=0
say "started (device $DEVICE, options: $*)"
while true; do
  "$PY" -u -W ignore scripts/run_emulator.py --backend nexus --device "$DEVICE" --name "$NAME" "$@" 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    say "finished after $n restarts"
    exit 0
  fi
  n=$((n + 1))
  if [ "$n" -gt "$MAX_RETRIES" ]; then
    say "gave up after $MAX_RETRIES restarts (last exit status $status)"
    exit 1
  fi
  say "stopped with exit status $status; restart $n in $RETRY_DELAY s"
  sleep "$RETRY_DELAY"
done
