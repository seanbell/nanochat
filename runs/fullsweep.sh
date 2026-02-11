#!/bin/bash
set -o pipefail

# Full pipeline sweep: pretrain → eval → SFT → eval → RL → eval across multiple depths.
# Generates scaling law data across the entire training pipeline for comparing configs.
#
# Usage: bash runs/fullsweep.sh [series_name] [--from FROM_SERIES] [--sft-datamix DATAMIX]
# Example: bash runs/fullsweep.sh baseline
#
# Post-training only (reuse pretrained models from another run):
#   bash runs/fullsweep.sh dolci --from baseline2 --sft-datamix allenai/Dolci-Instruct-SFT
#
# Compare configs by re-running with different series names:
#   bash runs/fullsweep.sh baseline 2>&1 | tee runs/fullsweep_baseline.log
#   bash runs/fullsweep.sh new_data 2>&1 | tee runs/fullsweep_newdata.log
#
# Env vars:
#   DEPTHS            - override depth array (default: "12 14 16 18 20 24")
#   SERIES_NAME       - label for this config (also 1st arg)
#   NPROC_PER_NODE    - GPU count (auto-detected)
#   WANDB_RUN         - wandb run prefix
#   SKIP_SETUP        - skip env setup for re-runs

# HOME may not be set in some environments (e.g. Lambda NFS)
export HOME="${HOME:-$(eval echo ~$(whoami))}"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Setup (skip with SKIP_SETUP=1)
if [ -z "$SKIP_SETUP" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate

    # Tokenizer + dataset
    python -m nanochat.dataset -n 1000
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768

    # Identity conversations for SFT
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
else
    source .venv/bin/activate
fi

# Series name: from arg, env var, or default to today's date (e.g., jan11)
SERIES_NAME="${1:-${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
shift || true
# Optional flags: --from (reuse pretrained models), --sft-datamix (custom SFT data)
FROM_SERIES=""
SFT_DATAMIX=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --from) FROM_SERIES="$2"; shift 2 ;;
        --sft-datamix) SFT_DATAMIX="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
# Depths to sweep (6 points, well-spaced in log(params) since params ~ d^3)
DEPTHS=(${DEPTHS:-12 14 16 18 20 24})
# Hardware: auto-detect GPUs, override with NPROC_PER_NODE env var
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
# Wandb
WANDB_RUN="${WANDB_RUN:-${SERIES_NAME}}"

RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_fullsweep_results"
mkdir -p "$RESULTS_DIR"

# Group all runs in this sweep together in wandb dashboard
export WANDB_RUN_GROUP="${WANDB_RUN}_fullsweep"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# Stage-level resume via marker files
stage_done() { [ -f "$RESULTS_DIR/${TAG}_${1}.done" ]; }
mark_done()  { touch "$RESULTS_DIR/${TAG}_${1}.done"; }

# Run a stage if not already done. Usage:
#   run_stage <step_label> <stage_name> [--timed] -- <command...>
# Logs, checks .done marker, runs command with tee, marks done on success.
run_stage() {
    local label="$1"; shift
    local stage="$1"; shift
    local timed=false
    if [ "$1" = "--timed" ]; then timed=true; shift; fi
    [ "$1" = "--" ] && shift

    if stage_done "$stage"; then
        log "$label d=$d... skipped (done)"
        return 0
    fi
    log "$label d=$d..."

    local start=$(date +%s)
    "$@" 2>&1 | tee "$RESULTS_DIR/${TAG}_${stage}.log" || { log "ERROR: $stage failed for d=$d"; exit 1; }
    if $timed; then
        echo $(( $(date +%s) - start )) > "$RESULTS_DIR/${TAG}_${stage}.time"
    fi
    mark_done "$stage"
}

# Write/update CSVs from log files (all analysis done in Python)
update_csvs() { python runs/analyze_fullsweep.py "$SERIES_NAME" --write-csv; }

# Device batch size for a given depth
get_batch_size() {
    if [ $1 -ge 28 ]; then echo 8
    elif [ $1 -ge 20 ]; then echo 16
    else echo 32; fi
}

# =============================================================================
log "=============================================="
log "${SERIES_NAME} Full Pipeline Sweep"
log "Depths: ${DEPTHS[*]}"
log "GPUs: ${NPROC_PER_NODE}"
[ -n "$FROM_SERIES" ] && log "From: ${FROM_SERIES} (post-training only)"
[ -n "$SFT_DATAMIX" ] && log "SFT datamix: ${SFT_DATAMIX}"
log "=============================================="

# =============================================================================
# Phase 1: Pretrain + SFT for all depths (scaling law data available early)
# =============================================================================
log "Phase 1: Pretrain + SFT for all depths"

for d in "${DEPTHS[@]}"; do
    TAG="${SERIES_NAME}_fullsweep_d${d}"
    BS=$(get_batch_size $d)

    log "--- Phase 1: d=$d (batch_size=$BS, tag=$TAG) ---"

    if [ -z "$FROM_SERIES" ]; then
        run_stage "[1/8]" pretrain --timed -- \
            torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
                --depth=$d --run="${WANDB_RUN}_d${d}_pretrain" --model-tag="${TAG}" \
                --device-batch-size=$BS --fp8

        run_stage "[2/8]" base_eval -- \
            torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- \
                --model-tag="${TAG}" --device-batch-size=$BS
    else
        FROM_TAG="${FROM_SERIES}_fullsweep_d${d}"
        log "[1/8] pretrain d=$d... skipped (--from $FROM_SERIES)"
        log "[2/8] base_eval d=$d... skipped (--from $FROM_SERIES)"
    fi

    # Build SFT command with optional --from-tag and --datamix
    SFT_EXTRA_ARGS=""
    [ -n "$FROM_SERIES" ] && SFT_EXTRA_ARGS="$SFT_EXTRA_ARGS --from-tag=${FROM_TAG}"
    [ -n "$SFT_DATAMIX" ] && SFT_EXTRA_ARGS="$SFT_EXTRA_ARGS --datamix=${SFT_DATAMIX}"

    run_stage "[3/8]" sft --timed -- \
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
            --run="${WANDB_RUN}_d${d}_sft" --model-tag="${TAG}" --device-batch-size=$BS \
            --assistant-only $SFT_EXTRA_ARGS

    run_stage "[4/8]" chat_eval_sft -- \
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
            -i sft --model-tag="${TAG}"

    run_stage "[5/8]" sft_holdout_bpb -- \
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- \
            --source=sft --model-tag="${TAG}" --device-batch-size=$BS --eval=bpb,core
done

log "Phase 1 complete. Updating CSVs..."
update_csvs

# =============================================================================
# Phase 2: RL for all depths (pretrain+SFT scaling data already available)
# =============================================================================
log "Phase 2: RL for all depths"

for d in "${DEPTHS[@]}"; do
    TAG="${SERIES_NAME}_fullsweep_d${d}"
    BS=$(get_batch_size $d)

    # Skip if all RL stages already done
    if stage_done rl && stage_done chat_eval_rl && stage_done rl_holdout_bpb; then
        log "Skipping d=$d (Phase 2 complete)"
        continue
    fi

    log "--- Phase 2: d=$d (tag=$TAG) ---"

    run_stage "[6/8]" rl --timed -- \
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- \
            --run="${WANDB_RUN}_d${d}_rl" --model-tag="${TAG}" --device-batch-size=8

    run_stage "[7/8]" chat_eval_rl -- \
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
            -i rl --model-tag="${TAG}"

    run_stage "[8/8]" rl_holdout_bpb -- \
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- \
            --source=rl --model-tag="${TAG}" --device-batch-size=$BS --eval=bpb,core
done

# Final CSV update and summary
update_csvs
log "${SERIES_NAME} Full Pipeline Sweep Complete!"
python runs/analyze_fullsweep.py "$SERIES_NAME"
