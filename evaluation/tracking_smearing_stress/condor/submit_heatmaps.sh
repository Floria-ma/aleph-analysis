#!/bin/bash
# Submit make_heatmaps.py as a single condor job, after the stress inference is done.
#
# usage: ./submit_heatmaps.sh [--dry-run] [--force] [--limit-files N]
#   --dry-run       write the submit file, but do not submit
#   --force         submit even if some stress outputs are still missing
#   --limit-files N five-file style preview (passed to make_heatmaps.py)
set -euo pipefail

source "$(dirname "$0")/config.sh"

SUBMIT=1
FORCE=0
EXTRA=""
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run|--no-submit) SUBMIT=0 ;;
    --force) FORCE=1 ;;
    --limit-files) EXTRA="--limit-files $2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

# check that the stress inference is complete
if [ ${FORCE} -eq 0 ] && [ ${SUBMIT} -eq 1 ]; then
  nmissing=0
  for strength in ${STRENGTHS}; do
    tag=$(printf "%03dpct" "${strength}")
    for sample in "${MC_DIR}"/output_qqb_*.root; do
      [ -s "${STRESS_SCORES}/${tag}/$(basename "${sample}" .root).pkl" ] || nmissing=$((nmissing + 1))
    done
  done
  if [ ${nmissing} -gt 0 ] && [ -z "${EXTRA}" ]; then
    echo "ERROR: ${nmissing} stress outputs missing in ${STRESS_SCORES}." >&2
    echo "Run ./submit_stress_inference.sh again to resubmit, or pass --force." >&2
    exit 1
  fi
fi

mkdir -p "${LOG_DIR}"
SUBFILE=${CONDOR_DIR}/heatmaps.sub
cat > "${SUBFILE}" <<SUB
executable = ${CONDOR_DIR}/heatmaps_job.sh
arguments = ${EXTRA}
output = ${LOG_DIR}/heatmaps_\$(ClusterId).out
error = ${LOG_DIR}/heatmaps_\$(ClusterId).err
log = ${LOG_DIR}/heatmaps_\$(ClusterId).log
getenv = False
transfer_executable = False
request_cpus = ${HEATMAP_CPUS}
request_memory = ${HEATMAP_MEMORY}
+JobFlavour = "${HEATMAP_FLAVOUR}"
queue
SUB
echo "Wrote ${SUBFILE}"

if [ ${SUBMIT} -eq 1 ]; then
  condor_submit "${SUBFILE}"
else
  echo "Not submitting (dry run). Submit with: condor_submit ${SUBFILE}"
fi
