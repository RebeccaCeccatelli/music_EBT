#!/bin/bash
### Baseline HF GPT2 Transformer Music - Pretraining Script
### Hugging Face GPT2 baseline using transformers library for comparison
### This is a separate baseline to validate the benefits of EBT

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --array=0

### LOG INFO ###
export MODEL_SIZE="small"

lr=(0.0008)

### Project Root Discovery ###
find_project_root() {
    local dir="$1"
    for ((i=0; i<10; i++)); do
        if [[ -f "${dir}/train_model.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    echo ""
    return 1
}

PROJECT_ROOT="$(find_project_root "$(pwd)")"
if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "❌ Error: Could not find project root."
    exit 1
fi

SCRATCH_LOGS_DIR="${HOME}/orcd/scratch/rebcecca/music_EBT_logs"

export PYTHONPATH="${PROJECT_ROOT}:${HOME}/music-EBT/data/mus/symbolic:$PYTHONPATH"
export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}" || exit 1

# Parse command-line arguments
# Supported tokenizer types:
#   REMI                      - miditok REMI (default, giga-midi/tokens/miditok/)
#   Anticipation-Arrival-Time - AMT paper format, full vocab 55028 (giga-midi/tokens/anticipation/)
#   Anticipation-Vanilla      - arrival-time, no control block, smaller vocab (giga-midi/tokens/anticipation-vanilla/)
DATASET_NAME="giga_midi"
TOKENIZER_TYPE="REMI"
RESUME_CKPT=""
BATCH_SIZE=""
ACCUM_STEPS=""
VAL_CHECK_INTERVAL=""
LIMIT_VAL_BATCHES=""
# expandable_segments avoids fragmentation when cached-but-unallocated memory exists.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --tokenizer_type) TOKENIZER_TYPE="$2"; shift 2 ;;
        --resume_training_ckpt) RESUME_CKPT="$2"; shift 2 ;;
        --batch_size_per_device) BATCH_SIZE="$2"; shift 2 ;;
        --accumulate_grad_batches) ACCUM_STEPS="$2"; shift 2 ;;
        --val_check_interval) VAL_CHECK_INTERVAL="$2"; shift 2 ;;
        --limit_val_batches) LIMIT_VAL_BATCHES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Tokenizer-aware defaults (overridden by explicit CLI args above).
# Anticipation vocab (55028) produces ~120x larger logit tensors than REMI (~452);
# halve batch size to reduce per-step memory/compute, and checkpoint more often.
case "${TOKENIZER_TYPE}" in
    Anticipation-*)
        BATCH_SIZE="${BATCH_SIZE:-8}"
        ACCUM_STEPS="${ACCUM_STEPS:-16}"
        VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-100}"
        LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1072}"
        ;;
    *)
        BATCH_SIZE="${BATCH_SIZE:-16}"
        ACCUM_STEPS="${ACCUM_STEPS:-16}"
        VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
        LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
        ;;
esac

# Build tokenizer slug for run naming
case "${TOKENIZER_TYPE}" in
    REMI)                       TOK_SLUG="remi" ;;
    Anticipation-Arrival-Time)  TOK_SLUG="ant-at-full" ;;
    Anticipation-Vanilla)       TOK_SLUG="ant-at-ar" ;;
    Anticipation-Interarrival)  TOK_SLUG="ant-ia-full" ;;
    *)                          TOK_SLUG=$(echo "${TOKENIZER_TYPE}" | tr '[:upper:]' '[:lower:]') ;;
esac
PEAK_LR="${lr[${SLURM_ARRAY_TASK_ID}]}"
BASE_RUN_NAME="baseline-hf-gpt2-small-${TOK_SLUG}"
FULL_RUN_NAME="${BASE_RUN_NAME}-job${SLURM_JOB_ID:-local}"
scontrol update JobId="${SLURM_JOB_ID}" Name="${FULL_RUN_NAME}" 2>/dev/null || true
MAX_STEPS=100000

# Auto-detect last checkpoint from a previous run window.
# Excludes current job ID to avoid picking the newly created empty directory.
if [[ -z "${RESUME_CKPT}" ]]; then
    PREV_CKPT_DIR=$(ls -td "${SCRATCH_LOGS_DIR}/checkpoints/${BASE_RUN_NAME}"* 2>/dev/null | grep -v "job${SLURM_JOB_ID}" | head -1)
    if [[ -n "${PREV_CKPT_DIR}" && -f "${PREV_CKPT_DIR}/last.ckpt" ]]; then
        RESUME_CKPT="${PREV_CKPT_DIR}/last.ckpt"
        echo "Auto-resuming from: ${RESUME_CKPT}"
    fi
fi

PYTHON_PID=0
SIGTERM_RECEIVED=0
_handle_sigterm() {
    SIGTERM_RECEIVED=1
    echo "[$(date -Iseconds)] SIGTERM received — forwarding to Python (PID ${PYTHON_PID})..."
    [[ ${PYTHON_PID} -ne 0 ]] && kill -USR1 "${PYTHON_PID}" 2>/dev/null
}
trap '_handle_sigterm' TERM

python train_model.py \
--run_name "${FULL_RUN_NAME}" \
--modality "MUS_SYMB" \
--model_name "baseline_hf_gpt2_transformer" \
--model_size "${MODEL_SIZE}" \
--tokenizer_type "${TOKENIZER_TYPE}" \
--normalize_initial_condition \
--context_length 1024 \
--gpus "1" \
--peak_learning_rate "${PEAK_LR}" \
--batch_size_per_device "${BATCH_SIZE}" \
--accumulate_grad_batches "${ACCUM_STEPS}" \
--gradient_clip_val 1.0 \
--weight_decay 0.05 \
--min_lr_scale 10 \
--max_steps 100000 \
--max_scheduling_steps 100000 \
--warm_up_steps 10000 \
--dataset_name "${DATASET_NAME}" \
--tokenizer_config_path "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json" \
--num_workers 12 \
--validation_split_pct 0.05 \
--limit_train_batches 1.0 \
--limit_val_batches "${LIMIT_VAL_BATCHES}" \
--limit_test_batches 1.0 \
--wandb_project 'mus_symb_baseline_pretrain' \
--log_model_archi \
--log_gradients \
--log_every_n_steps 100 \
--set_matmul_precision "medium" \
--wandb_watch \
--val_check_interval "${VAL_CHECK_INTERVAL}" \
${RESUME_CKPT:+--resume_training_ckpt "${RESUME_CKPT}"} \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run} &
PYTHON_PID=$!
wait "${PYTHON_PID}"
TRAIN_EXIT_CODE=$?
if [[ ${SIGTERM_RECEIVED} -eq 1 ]]; then
    wait "${PYTHON_PID}" 2>/dev/null
    TRAIN_EXIT_CODE=$?
fi

# Exit 0:   clean exit — either training finished or PL saved a checkpoint on SIGTERM.
#           PL does NOT auto-requeue on this cluster, so we check the highest saved
#           step to decide whether to resubmit.
# Exit 143: SIGTERM reached bash before PL could handle it — resubmit unconditionally.
# Anything else: crash (OOM, Python error, etc.) — do not resubmit.
_do_resubmit() {
    sbatch "${BASH_SOURCE[0]}" \
        --tokenizer_type "${TOKENIZER_TYPE}" \
        --dataset_name "${DATASET_NAME}" \
        --batch_size_per_device "${BATCH_SIZE}" \
        --accumulate_grad_batches "${ACCUM_STEPS}" \
        --val_check_interval "${VAL_CHECK_INTERVAL}" \
        --limit_val_batches "${LIMIT_VAL_BATCHES}"
}

if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    LAST_STEP=$(ls "${SCRATCH_LOGS_DIR}/checkpoints/${BASE_RUN_NAME}"*/epoch=*.ckpt 2>/dev/null \
        | grep -oE "step=step=[0-9]+" | grep -oE "[0-9]+$" | sort -n | tail -1)
    if [[ -n "${LAST_STEP}" && "${LAST_STEP}" -ge "${MAX_STEPS}" ]]; then
        echo "Training complete at step ${LAST_STEP}. Not resubmitting."
    else
        echo "Clean exit but training incomplete (last step: ${LAST_STEP:-none}). Resubmitting..."
        _do_resubmit
    fi
elif [[ ${TRAIN_EXIT_CODE} -eq 143 ]]; then
    echo "Job killed by SIGTERM before PL could handle it. Resubmitting..."
    _do_resubmit
else
    echo "Training failed with exit code ${TRAIN_EXIT_CODE}. Not resubmitting."
    exit ${TRAIN_EXIT_CODE}
fi

# NOTES:
# - Baseline HF GPT2 transformer for comparison with EBT
# - Same training configuration as baseline_llama_transformer.sh for fair comparison
# - 100k steps matching AMT paper, 1024 context length (tokens already tokenized at 1024)
# - 24h time limit; resume with --resume_training_ckpt if wall time is hit
# - Standard AdamW optimizer with cosine annealing
# - Warmup 10k steps (10% of total, matching AMT paper ratio)
