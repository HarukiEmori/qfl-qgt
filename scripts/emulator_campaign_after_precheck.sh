#!/bin/bash
# Starts a campaign on the emulator after a short pre-check run has passed its tests.
#
#     scripts/emulator_campaign_after_precheck.sh <pre-check name> <campaign name> [options of the pre-check run]
#
# 1. The pre-check run <pre-check name> is executed to its end with scripts/emulator_campaign.sh and
#    the given options of run_emulator.py (for example: --scenarios emulator2_iid --seeds 900 901
#    --rounds 1 --no-eval).  Jobs of this run that are already submitted are re-attached, not
#    submitted again.
# 2. Its results are tested with scripts/emulator_precheck.py: all jobs completed, finite results,
#    and the noise strengths estimated from the echo circuits as expected for the devices.
# 3. Only if the tests pass, the campaign <campaign name> is started with scripts/emulator_campaign.sh.
#
# If the pre-check run does not finish or its tests fail, the campaign is not started, and the file
# results/<pre-check name>/precheck.failed is written.  Environment variables: as in
# scripts/emulator_campaign.sh.  Progress:  python scripts/emulator_status.py --name <name>

set -u
PRE=${1:?usage: scripts/emulator_campaign_after_precheck.sh <pre-check name> <campaign name> [options of the pre-check run]}
CAMPAIGN=${2:?usage: scripts/emulator_campaign_after_precheck.sh <pre-check name> <campaign name> [options of the pre-check run]}
shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 1
PY=${PYTHON:-python}
export PYTHON="$PY"
mkdir -p "results/$PRE"
FAILED="results/$PRE/precheck.failed"
rm -f "$FAILED"

say() { echo "[pre-check $PRE] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "results/$PRE/precheck.log"; }

say "pre-check run started; the campaign $CAMPAIGN follows if its tests pass"
if ! scripts/emulator_campaign.sh "$PRE" "$@"; then
  say "the pre-check run did not finish; the campaign $CAMPAIGN is not started"
  echo "the pre-check run did not finish" > "$FAILED"
  exit 1
fi
"$PY" -W ignore scripts/emulator_precheck.py --name "$PRE" 2>&1 | tee -a "results/$PRE/precheck.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  say "the tests of the pre-check run failed; the campaign $CAMPAIGN is not started"
  echo "the tests of the pre-check run failed" > "$FAILED"
  exit 1
fi
say "the tests passed; starting the campaign $CAMPAIGN"
exec scripts/emulator_campaign.sh "$CAMPAIGN"
