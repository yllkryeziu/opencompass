#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_eval.sh — Submit OpenCompass evaluation jobs for Qwen3 checkpoints
#
# Evaluates checkpoints on: math500, aime2025, ifeval, livecodebench
# Supports: auto-discovery, Megatron .distcp conversion, base model eval
#
# Usage:
#   # Evaluate all checkpoints in an experiment + base model:
#   bash launch_eval.sh \
#     --checkpoint-dir /p/scratch/envcomp/yll/checkpoints/distill-topk512-qwen3-8b-1node
#
#   # Evaluate specific steps only:
#   bash launch_eval.sh \
#     --checkpoint-dir /p/scratch/envcomp/yll/checkpoints/distill-topk512-qwen3-8b-1node \
#     --steps "100 250"
#
#   # Evaluate just the base (untrained) model:
#   bash launch_eval.sh \
#     --checkpoint-dir /p/scratch/envcomp/yll/checkpoints/distill-topk512-qwen3-8b-1node \
#     --base-only
#
#   # Quick test on develbooster:
#   bash launch_eval.sh \
#     --checkpoint-dir /p/scratch/envcomp/yll/checkpoints/distill-topk512-qwen3-1b7 \
#     --partition develbooster --time 02:00:00
#
#   # Select specific benchmarks:
#   bash launch_eval.sh --checkpoint-dir ... --benchmarks math500,ifeval
#
#   # Evaluate a single checkpoint directly:
#   bash launch_eval.sh \
#     /p/scratch/envcomp/yll/checkpoints/distill-topk512-qwen3-8b-1node/step_250/consolidated
#
#   # Dry run (show sbatch commands without submitting):
#   bash launch_eval.sh --checkpoint-dir ... --dry-run
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults (configurable via CLI flags) ───────────────────────────────────
PARTITION="booster"
ACCOUNT="envcomp"
TIME="08:00:00"
RESULTS_ROOT="/p/scratch/envcomp/yll/opencompass"
OPENCOMPASS_DIR="/p/project1/envcomp/yll/opencompass"
NEMO_DIR="/p/project1/envcomp/yll/nemo-rl"
BENCHMARKS="math500,aime2025,ifeval,livecodebench"
CHECKPOINT_DIR=""
STEPS=""
BASE_ONLY=0
DRY_RUN=0
CONVERT_PARTITION="develbooster"
CONVERT_TIME="01:00:00"

# Direct checkpoint paths (legacy mode: ./launch_eval.sh /path/to/ckpt)
DIRECT_PATHS=()

# ── Usage ───────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Usage:
  bash launch_eval.sh --checkpoint-dir <path> [options]
  bash launch_eval.sh <checkpoint_path> [checkpoint_path ...]

Options:
  --checkpoint-dir PATH    Root experiment checkpoint dir (contains step_* dirs)
  --steps "N1 N2 ..."     Evaluate only these steps (space-separated)
  --base-only              Evaluate only the base (untrained) model
  --partition NAME         SLURM partition (default: booster)
  --account NAME           SLURM account (default: envcomp)
  --time HH:MM:SS          SLURM time limit (default: 08:00:00)
  --benchmarks LIST        Comma-separated benchmarks (default: math500,aime2025,ifeval,livecodebench)
  --results-root PATH      Output root directory
  --convert-partition NAME SLURM partition for conversion jobs (default: develbooster)
  --dry-run                Show commands without submitting
  --list                   List available checkpoints
  -h, --help               Show this help
EOF
    exit "${1:-0}"
}

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir)  CHECKPOINT_DIR="$2"; shift 2 ;;
        --steps)           STEPS="$2"; shift 2 ;;
        --base-only)       BASE_ONLY=1; shift ;;
        --partition)       PARTITION="$2"; shift 2 ;;
        --account)         ACCOUNT="$2"; shift 2 ;;
        --time)            TIME="$2"; shift 2 ;;
        --benchmarks)      BENCHMARKS="$2"; shift 2 ;;
        --results-root)    RESULTS_ROOT="$2"; shift 2 ;;
        --convert-partition) CONVERT_PARTITION="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --list)
            echo "Available experiments with consolidated checkpoints:"
            echo ""
            find /p/scratch/envcomp/yll/checkpoints -maxdepth 3 -name "consolidated" -type d 2>/dev/null | sort | while read -r p; do
                step_dir="$(dirname "$p")"
                run_dir="$(dirname "$step_dir")"
                echo "  $(basename "$run_dir")/$(basename "$step_dir")"
            done
            exit 0
            ;;
        -h|--help)         usage 0 ;;
        -*)                echo "[ERROR] Unknown flag: $1" >&2; usage 2 ;;
        *)
            # Legacy mode: direct checkpoint path
            DIRECT_PATHS+=("$1"); shift ;;
    esac
done

if [[ -z "$CHECKPOINT_DIR" && ${#DIRECT_PATHS[@]} -eq 0 ]]; then
    echo "[ERROR] Either --checkpoint-dir or direct checkpoint paths required." >&2
    usage 2
fi

# ── Helper functions ────────────────────────────────────────────────────────

parse_job_id() {
    local out="$1"
    if [[ "$out" =~ ^([0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    if [[ "$out" =~ Submitted[[:space:]]+batch[[:space:]]+job[[:space:]]+([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

# Auto-detect base model path from config.yaml in any step dir
detect_base_model() {
    local ckpt_dir="$1"
    local config_yaml=""

    for d in "$ckpt_dir"/step_*/; do
        if [[ -f "$d/config.yaml" ]]; then
            config_yaml="$d/config.yaml"
            break
        fi
    done

    if [[ -z "$config_yaml" ]]; then
        echo "[ERROR] No config.yaml found in any step_* dir under $ckpt_dir" >&2
        return 1
    fi

    python3 -c "
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg['policy']['model_name'])
" "$config_yaml"
}

# Detect model size from base model or consolidated config.json → set TP size
# Falls back to inferring from the model path name if config.json is not accessible.
detect_tp_size() {
    local model_path="$1"
    local config_file="$model_path/config.json"

    if [[ -f "$config_file" ]]; then
        local hidden_size
        hidden_size=$(python3 -c "import json; print(json.load(open('$config_file'))['hidden_size'])")

        # hidden_size -> TP mapping:
        #   2048 (1.7B) -> TP=1
        #   2560 (4B)   -> TP=1
        #   4096 (8B)   -> TP=4
        #   5120 (14B)  -> TP=4
        if [[ "$hidden_size" -le 2560 ]]; then
            echo 1
        else
            echo 4
        fi
        return
    fi

    # Fallback: infer from path name
    case "$model_path" in
        *[Qq]wen3-1.7[Bb]*|*[Qq]wen3-1[Bb]7*|*1b7*) echo 1 ;;
        *[Qq]wen3-4[Bb]*|*4b*)                        echo 1 ;;
        *[Qq]wen3-8[Bb]*|*8b*)                        echo 4 ;;
        *[Qq]wen3-14[Bb]*|*14b*)                      echo 4 ;;
        *)
            echo "[WARN] Cannot detect model size from $model_path, defaulting to TP=1" >&2
            echo 1
            ;;
    esac
}

# Derive experiment name from checkpoint dir
derive_experiment_name() {
    basename "$1"
}

# Derive model abbreviation
derive_model_abbr() {
    local experiment="$1"
    local label="$2"  # "base" or "step_NNN"
    echo "${experiment}__${label}"
}

# Submit a single OpenCompass eval job
submit_eval_job() {
    local ckpt_path="$1"
    local model_abbr="$2"
    local experiment="$3"
    local label="$4"      # "base" or "step_NNN"
    local dependency="$5"  # "" or "afterok:JOBID"

    local tp_size
    tp_size="$(detect_tp_size "$ckpt_path")"
    local num_gpus="$tp_size"

    local work_dir="${RESULTS_ROOT}/${experiment}/${label}"
    local log_dir="${RESULTS_ROOT}/${experiment}/slurm-logs"
    mkdir -p "$log_dir"

    local dep_args=()
    if [[ -n "$dependency" ]]; then
        dep_args=(--dependency "$dependency")
    fi

    echo "  [$label]"
    echo "    checkpoint: $ckpt_path"
    echo "    model_abbr: $model_abbr"
    echo "    TP size:    $tp_size"
    echo "    benchmarks: $BENCHMARKS"
    echo "    output:     $work_dir"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "    [DRY RUN] Would submit: sbatch -p $PARTITION --gres=gpu:$num_gpus ..."
        echo ""
        return
    fi

    local job_out
    job_out=$(sbatch \
        --parsable \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --time="$TIME" \
        --nodes=1 \
        --ntasks=1 \
        --gres="gpu:$num_gpus" \
        --cpus-per-task=32 \
        --job-name="oc-eval-${label}" \
        --output="${log_dir}/eval_${label}_%j.out" \
        --error="${log_dir}/eval_${label}_%j.err" \
        "${dep_args[@]}" \
        --export=ALL,CKPT_PATH="$ckpt_path",MODEL_ABBR="$model_abbr",TP_SIZE="$tp_size",BENCHMARKS="$BENCHMARKS",WORK_DIR="$work_dir",OPENCOMPASS_DIR="$OPENCOMPASS_DIR" \
        "$OPENCOMPASS_DIR/eval_worker.sh")

    local job_id
    job_id="$(parse_job_id "$job_out")" || {
        echo "    [ERROR] Failed to parse job ID from: $job_out" >&2
        return 1
    }
    echo "    job ID:     $job_id"
    echo ""
}

# ── Main: --checkpoint-dir mode ─────────────────────────────────────────────
if [[ -n "$CHECKPOINT_DIR" ]]; then
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        echo "[ERROR] Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
        exit 1
    fi

    EXPERIMENT="$(derive_experiment_name "$CHECKPOINT_DIR")"
    echo "════════════════════════════════════════════════════════════════"
    echo "  Experiment:      $EXPERIMENT"
    echo "  Checkpoint dir:  $CHECKPOINT_DIR"
    echo "  Partition:       $PARTITION"
    echo "  Time limit:      $TIME"
    echo "  Benchmarks:      $BENCHMARKS"
    echo "  Results root:    $RESULTS_ROOT"
    echo "════════════════════════════════════════════════════════════════"
    echo ""

    # Detect base model
    BASE_MODEL="$(detect_base_model "$CHECKPOINT_DIR")"
    echo "[INFO] Base model: $BASE_MODEL"

    # Discover steps
    ALL_STEPS=()
    for d in "$CHECKPOINT_DIR"/step_*/; do
        [[ ! -d "$d" ]] && continue
        step_name="$(basename "$d")"
        step_num="${step_name#step_}"
        ALL_STEPS+=("$step_num")
    done

    # Filter by --steps if specified
    if [[ -n "$STEPS" ]]; then
        SELECTED_STEPS=()
        for s in $STEPS; do
            # Normalize: accept both "100" and "step_100"
            s="${s#step_}"
            if [[ -d "$CHECKPOINT_DIR/step_${s}" ]]; then
                SELECTED_STEPS+=("$s")
            else
                echo "[WARN] Step $s not found, skipping" >&2
            fi
        done
    else
        SELECTED_STEPS=("${ALL_STEPS[@]}")
    fi

    # Sort steps numerically
    IFS=$'\n' SELECTED_STEPS=($(printf '%s\n' "${SELECTED_STEPS[@]}" | sort -n)); unset IFS

    echo "[INFO] Found steps: ${ALL_STEPS[*]:-none}"
    echo "[INFO] Selected steps: ${SELECTED_STEPS[*]:-none}"
    echo ""

    # ── Check for steps needing conversion ──────────────────────────────
    CONVERT_JOB_ID=""
    DEPENDENCY=""

    if [[ "$BASE_ONLY" -eq 0 && ${#SELECTED_STEPS[@]} -gt 0 ]]; then
        NEED_CONVERT=()
        for step in "${SELECTED_STEPS[@]}"; do
            consolidated_cfg="$CHECKPOINT_DIR/step_${step}/consolidated/config.json"
            if [[ ! -f "$consolidated_cfg" ]]; then
                NEED_CONVERT+=("$step")
            fi
        done

        if [[ ${#NEED_CONVERT[@]} -gt 0 ]]; then
            echo "[INFO] Steps needing conversion: ${NEED_CONVERT[*]}"
            CONVERT_STEPS_STR="${NEED_CONVERT[*]}"

            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "[DRY RUN] Would submit conversion job for steps: $CONVERT_STEPS_STR"
                CONVERT_JOB_ID="DRYRUN_CONVERT"
            else
                CONVERT_SCRIPT="$NEMO_DIR/scripts/ensure_consolidated_checkpoints.sh"
                if [[ ! -f "$CONVERT_SCRIPT" ]]; then
                    echo "[ERROR] Conversion script not found: $CONVERT_SCRIPT" >&2
                    echo "[ERROR] Cannot convert .distcp checkpoints without nemo-rl." >&2
                    exit 1
                fi

                convert_out=$(sbatch \
                    --parsable \
                    --partition="$CONVERT_PARTITION" \
                    --account="$ACCOUNT" \
                    --nodes=1 \
                    --ntasks=1 \
                    --cpus-per-task=8 \
                    --time="$CONVERT_TIME" \
                    --job-name="convert_${EXPERIMENT}" \
                    --output="${RESULTS_ROOT}/${EXPERIMENT}/slurm-logs/convert_%j.out" \
                    --error="${RESULTS_ROOT}/${EXPERIMENT}/slurm-logs/convert_%j.err" \
                    --export=ALL,NEMO_DIR="$NEMO_DIR",CHECKPOINT_DIR="$CHECKPOINT_DIR",BASE_MODEL="$BASE_MODEL",FORCE_EVAL_STEPS="$CONVERT_STEPS_STR",UV_PROJECT_ENVIRONMENT="$NEMO_DIR/.venv" \
                    "$NEMO_DIR/scripts/slurm_convert_checkpoint_run.sh")

                CONVERT_JOB_ID="$(parse_job_id "$convert_out")" || {
                    echo "[ERROR] Failed to parse conversion job ID" >&2
                    exit 1
                }
                echo "[INFO] Conversion job submitted: $CONVERT_JOB_ID"
                DEPENDENCY="afterok:${CONVERT_JOB_ID}"
            fi
            echo ""
        else
            echo "[INFO] All selected steps already have consolidated checkpoints."
            echo ""
        fi
    fi

    # ── Submit base model eval ──────────────────────────────────────────
    BASE_ABBR="$(derive_model_abbr "$EXPERIMENT" "base")"
    echo "[INFO] Submitting evaluation jobs..."
    echo ""
    submit_eval_job "$BASE_MODEL" "$BASE_ABBR" "$EXPERIMENT" "base" ""

    # ── Submit checkpoint eval jobs ─────────────────────────────────────
    if [[ "$BASE_ONLY" -eq 0 ]]; then
        for step in "${SELECTED_STEPS[@]}"; do
            label="step_${step}"
            ckpt_path="$CHECKPOINT_DIR/${label}/consolidated"
            step_abbr="$(derive_model_abbr "$EXPERIMENT" "$label")"
            submit_eval_job "$ckpt_path" "$step_abbr" "$EXPERIMENT" "$label" "$DEPENDENCY"
        done
    fi

    echo "════════════════════════════════════════════════════════════════"
    echo "  All jobs submitted. Monitor with: squeue -u \$USER"
    echo "  Results: $RESULTS_ROOT/$EXPERIMENT/"
    echo "════════════════════════════════════════════════════════════════"

# ── Main: direct paths mode (legacy) ────────────────────────────────────────
elif [[ ${#DIRECT_PATHS[@]} -gt 0 ]]; then
    echo "Submitting ${#DIRECT_PATHS[@]} evaluation job(s)..."
    echo ""

    for raw_path in "${DIRECT_PATHS[@]}"; do
        # Resolve to consolidated/ dir
        raw_path="${raw_path%/}"
        if [[ "$(basename "$raw_path")" == "consolidated" ]] && [[ -d "$raw_path" ]]; then
            ckpt_path="$raw_path"
        elif [[ -d "$raw_path/consolidated" ]]; then
            ckpt_path="$raw_path/consolidated"
        else
            echo "[ERROR] No consolidated/ directory found at $raw_path" >&2
            continue
        fi

        # Derive names
        step_dir="$(dirname "$ckpt_path")"
        run_dir="$(dirname "$step_dir")"
        experiment="$(basename "$run_dir")"
        label="$(basename "$step_dir")"
        model_abbr="$(derive_model_abbr "$experiment" "$label")"

        submit_eval_job "$ckpt_path" "$model_abbr" "$experiment" "$label" ""
    done

    echo "All jobs submitted. Monitor with: squeue -u \$USER"
    echo "Results: $RESULTS_ROOT/"
fi
