#!/bin/bash
# Condor executable: run tracking-smearing inference for one (strength, MC file).
# Mirrors the command built in run_stress_inference.py, but for a single file.
# usage: stress_inference_job.sh <strength> <sample.root>

STRENGTH=$1
SAMPLE=$2
source "$(dirname "$0")/config.sh"

args=("$@"); set --
source "${ENV_SETUP}"
set -- "${args[@]}"
# strict mode only after sourcing the environment (setup scripts are not -e safe)
set -euo pipefail

STRENGTH_TAG=$(printf "%03dpct" "${STRENGTH}")
CARD=${CARDS_DIR}/tracking_smearing_${STRENGTH_TAG}.json
STEM=$(basename "${SAMPLE}" .root)
FINAL_DIR=${STRESS_SCORES}/${STRENGTH_TAG}
FINAL_FILE=${FINAL_DIR}/${STEM}.pkl

# write to local scratch first so a crashed job never leaves a partial pkl on EOS
TMP_DIR=${_CONDOR_SCRATCH_DIR:-$(mktemp -d)}/stress_${STRENGTH_TAG}
mkdir -p "${TMP_DIR}" "${FINAL_DIR}"

echo "host: $(hostname)"
echo "strength: ${STRENGTH_TAG}, sample: ${SAMPLE}, card: ${CARD}"
echo "python: $(which python)"

cd "${ANALYSIS_DIR}"
python -u evaluation/inference.py \
  -s "${SAMPLE}" -m "${MODEL}" -p "${PREPROCESS}" \
  -o "${TMP_DIR}" -t "${TREE}" \
  --translation "${TRANSLATION}" \
  --event-batch-size "${EVENT_BATCH_SIZE}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --onnx-threads "${ONNX_THREADS}" \
  --tracking-variation-card "${CARD}" \
  --tracking-variation "${VARIATION}" \
  --tracking-variation-apply-to mc \
  --tracking-variation-seed "${SEED}"

if [ ! -s "${TMP_DIR}/${STEM}.pkl" ]; then
  echo "ERROR: expected output ${TMP_DIR}/${STEM}.pkl not found or empty" >&2
  exit 1
fi
cp "${TMP_DIR}/${STEM}.pkl" "${FINAL_FILE}.tmp"
mv "${FINAL_FILE}.tmp" "${FINAL_FILE}"
echo "wrote ${FINAL_FILE}"
