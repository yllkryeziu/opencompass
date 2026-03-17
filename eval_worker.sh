#!/bin/bash
# eval_worker.sh — SLURM worker for OpenCompass evaluation
#
# Submitted by launch_eval.sh (do not run directly).
#
# Required env:
#   CKPT_PATH    - path to HuggingFace-format checkpoint directory
#   MODEL_ABBR   - short name for the model (used in output paths)
#   TP_SIZE      - tensor parallel size
#   BENCHMARKS   - comma-separated list of benchmarks to run
#   WORK_DIR     - OpenCompass output directory
#
# Optional env:
#   OPENCOMPASS_DIR  - path to opencompass repo (default: script's parent dir)
#   HF_HOME          - HuggingFace cache root

set -euo pipefail

echo "=== OpenCompass Evaluation ==="
echo "  Checkpoint: $CKPT_PATH"
echo "  Model:      $MODEL_ABBR"
echo "  TP size:    $TP_SIZE"
echo "  Benchmarks: $BENCHMARKS"
echo "  Node:       $(hostname)"
echo "  GPUs:       ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "  Start:      $(date)"
echo ""

# Module setup (JUWELS Booster)
module --force purge
module load Stages/2026
module load CUDA

# Environment
export HF_HOME="${HF_HOME:-/p/project1/envcomp/yll/.cache/huggingface}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

# Activate OpenCompass venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENCOMPASS_DIR="${OPENCOMPASS_DIR:-$SCRIPT_DIR}"
source "$OPENCOMPASS_DIR/.venv/bin/activate"

cd "$OPENCOMPASS_DIR"

# Prefix the output leaf dir with the SLURM job ID so results dirs
# are naturally sorted by submission order.
WORK_DIR="${WORK_DIR%/*}/${SLURM_JOB_ID}_${WORK_DIR##*/}"
echo "  Work dir:   $WORK_DIR"
echo ""

python run.py eval_checkpoints.py \
    -w "$WORK_DIR"

echo ""
echo "=== Done: $(date) ==="
