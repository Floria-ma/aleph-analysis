#!/bin/bash
# Submit the tracking-smearing inference to condor: one job per (strength, MC file).
# Resumable: only (strength, file) pairs without a non-empty output pkl are submitted.
#
# usage: ./submit_stress_inference.sh [--dry-run] [--test] [--no-submit]
#   --dry-run    write the submit file and job list, but do not submit
#   --test       only submit the first pending job
#   --no-submit  same as --dry-run (used by submit_all.sh to build the DAG)
set -euo pipefail

source "$(dirname "$0")/config.sh"

SUBMIT=1
TEST=0
for arg in "$@"; do
  case ${arg} in
    --dry-run|--no-submit) SUBMIT=0 ;;
    --test) TEST=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 1 ;;
  esac
done

for f in "${MODEL}" "${PREPROCESS}" "${TRANSLATION}"; do
  [ -f "${f}" ] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

mapfile -t SAMPLES < <(ls "${MC_DIR}"/output_qqb_*.root | sort -t_ -k3 -n)
[ ${#SAMPLES[@]} -gt 0 ] || { echo "ERROR: no output_qqb_*.root in ${MC_DIR}" >&2; exit 1; }

mkdir -p "${LOG_DIR}"
JOBLIST=${CONDOR_DIR}/stress_inference_jobs.txt
: > "${JOBLIST}"
ntotal=0
for strength in ${STRENGTHS}; do
  tag=$(printf "%03dpct" "${strength}")
  card=${CARDS_DIR}/tracking_smearing_${tag}.json
  [ -f "${card}" ] || { echo "ERROR: missing variation card ${card}" >&2; exit 1; }
  for sample in "${SAMPLES[@]}"; do
    ntotal=$((ntotal + 1))
    out=${STRESS_SCORES}/${tag}/$(basename "${sample}" .root).pkl
    [ -s "${out}" ] && continue
    echo "${strength} ${sample}" >> "${JOBLIST}"
  done
done
[ ${TEST} -eq 1 ] && sed -i '2,$d' "${JOBLIST}"
npending=$(wc -l < "${JOBLIST}")
echo "Found ${#SAMPLES[@]} MC files x strengths (${STRENGTHS}) = ${ntotal} jobs; ${npending} to submit."

SUBFILE=${CONDOR_DIR}/stress_inference.sub
cat > "${SUBFILE}" <<SUB
executable = ${CONDOR_DIR}/stress_inference_job.sh
arguments = \$(strength) \$(sample)
output = ${LOG_DIR}/inference_\$(strength)_\$(ClusterId)_\$(ProcId).out
error = ${LOG_DIR}/inference_\$(strength)_\$(ClusterId)_\$(ProcId).err
log = ${LOG_DIR}/inference_\$(ClusterId).log
getenv = False
transfer_executable = False
request_cpus = ${INFERENCE_CPUS}
request_memory = ${INFERENCE_MEMORY}
+JobFlavour = "${INFERENCE_FLAVOUR}"
max_retries = 2
queue strength, sample from ${JOBLIST}
SUB
echo "Wrote ${SUBFILE}"

if [ "${npending}" -eq 0 ]; then
  echo "Nothing to submit."
elif [ ${SUBMIT} -eq 1 ]; then
  condor_submit "${SUBFILE}"
else
  echo "Not submitting (dry run). Submit with: condor_submit ${SUBFILE}"
fi
