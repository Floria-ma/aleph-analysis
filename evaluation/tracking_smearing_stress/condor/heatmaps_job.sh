#!/bin/bash
# Condor executable: run make_heatmaps.py on the nominal + stress scores.
# usage: heatmaps_job.sh [extra make_heatmaps.py args, e.g. --limit-files 5]

source "$(dirname "$0")/config.sh"

args=("$@"); set --
source "${ENV_SETUP}"
set -- "${args[@]}"
# strict mode only after sourcing the environment (setup scripts are not -e safe)
set -euo pipefail

mkdir -p "${RESULTS}"
echo "host: $(hostname)"
echo "python: $(which python)"

cd "${BUNDLE_DIR}"
python -u make_heatmaps.py \
  --mc-dir "${MC_DIR}" \
  --nominal-score-dir "${NOMINAL_SCORES}" \
  --stress-score-dir "${STRESS_SCORES}" \
  --thrust-script "${THRUST_SCRIPT}" \
  --output-dir "${RESULTS}" \
  "$@"
