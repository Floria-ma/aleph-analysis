#!/bin/bash
# Shared settings for the condor submission of the tracking-smearing stress test.
# Sourced by the submit scripts and by the job executables. Edit as needed.

CONDOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "${CONDOR_DIR}/.." && pwd)"
ANALYSIS_DIR="$(cd "${BUNDLE_DIR}/../.." && pwd)"

# environment (key4hep + optional local venv overlay, see setup.sh)
ENV_SETUP="${ANALYSIS_DIR}/setup.sh"

# inputs
MODELTAG=20260305_withnewks_standardized
MC_DIR=/eos/user/l/llambrec/aleph-data/ntuples-withks/eventlevel/mc
MODEL=/eos/user/l/llambrec/aleph-data/models/output_${MODELTAG}/model.onnx
PREPROCESS=/eos/user/l/llambrec/aleph-data/models/output_${MODELTAG}/preprocess.json
TRANSLATION=${ANALYSIS_DIR}/evaluation/translations/translations.json
CARDS_DIR=${BUNDLE_DIR}/cards
NOMINAL_SCORES=/eos/user/l/llambrec/aleph-data/model_output_scores/output_scores_model_${MODELTAG}
THRUST_SCRIPT=${BUNDLE_DIR}/thrust.py

# outputs
STRESS_SCORES=/eos/user/z/zima/aleph-data/tracking_stress_scores_${MODELTAG}
RESULTS=/eos/user/z/zima/aleph-data/tracking_stress_results_${MODELTAG}

# inference settings (same defaults as run_stress_inference.py)
STRENGTHS="100 200 300"
VARIATION=refined_hybrid_non_gaussian_tracking_candidate
SEED=12345
TREE=events
EVENT_BATCH_SIZE=1000
INFERENCE_BATCH_SIZE=50
ONNX_THREADS=2

# condor settings (job logs must live on AFS, not EOS)
LOG_DIR=${CONDOR_DIR}/logs
INFERENCE_FLAVOUR=workday
INFERENCE_CPUS=2
INFERENCE_MEMORY=4000
HEATMAP_FLAVOUR=tomorrow
HEATMAP_CPUS=1
HEATMAP_MEMORY=8000
