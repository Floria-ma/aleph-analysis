#!/bin/bash
# Submit the full workflow as a DAG: all stress-inference jobs, then make_heatmaps
# once every inference job has succeeded.
#
# usage: ./submit_all.sh [--dry-run]
set -euo pipefail

source "$(dirname "$0")/config.sh"

SUBMIT=1
[ "${1:-}" = "--dry-run" ] && SUBMIT=0

"${CONDOR_DIR}/submit_stress_inference.sh" --no-submit
"${CONDOR_DIR}/submit_heatmaps.sh" --no-submit

DAGFILE=${CONDOR_DIR}/tracking_stress.dag
rm -f "${DAGFILE}".*
if [ -s "${CONDOR_DIR}/stress_inference_jobs.txt" ]; then
  cat > "${DAGFILE}" <<DAG
JOB inference ${CONDOR_DIR}/stress_inference.sub
JOB heatmaps ${CONDOR_DIR}/heatmaps.sub
PARENT inference CHILD heatmaps
DAG
else
  echo "JOB heatmaps ${CONDOR_DIR}/heatmaps.sub" > "${DAGFILE}"
fi
echo "Wrote ${DAGFILE}"

if [ ${SUBMIT} -eq 1 ]; then
  cd "${CONDOR_DIR}"
  condor_submit_dag "${DAGFILE}"
else
  echo "Not submitting (dry run). Submit with: condor_submit_dag ${DAGFILE}"
fi
