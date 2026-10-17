#!/bin/bash
#SBATCH --job-name=tracking-stress
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=tracking-stress-%j.out
#SBATCH --error=tracking-stress-%j.err

set -euo pipefail

# Add the module/virtual-environment setup used by your aleph-analysis checkout.
# module load python/3.11
# source /path/to/venv/bin/activate

ANALYSIS_DIR=/path/to/R_b
BUNDLE_DIR=${ANALYSIS_DIR}/evaluation/tracking_smearing_stress
MC_DIR=/path/to/eventlevel/mc
MODEL=/path/to/model.onnx
PREPROCESS=/path/to/preprocess.json
TRANSLATION=${ANALYSIS_DIR}/evaluation/translations/translations.json
NOMINAL_SCORES=/path/to/output_scores_model_20260305_withnewks_standardized
STRESS_SCORES=/path/to/tracking_stress_scores
THRUST_SCRIPT=${BUNDLE_DIR}/thrust.py
RESULTS=/path/to/full_stress_results

cd "${BUNDLE_DIR}"

python -u run_stress_inference.py \
  --analysis-dir "${ANALYSIS_DIR}" \
  --mc-dir "${MC_DIR}" \
  --model "${MODEL}" \
  --preprocess "${PREPROCESS}" \
  --translation "${TRANSLATION}" \
  --output-dir "${STRESS_SCORES}" \
  --strengths 100 200 300 \
  --workers 2 \
  --onnx-threads 2

python -u make_heatmaps.py \
  --mc-dir "${MC_DIR}" \
  --nominal-score-dir "${NOMINAL_SCORES}" \
  --stress-score-dir "${STRESS_SCORES}" \
  --thrust-script "${THRUST_SCRIPT}" \
  --output-dir "${RESULTS}"
