#!/bin/bash
### EBT Symbolic Music - Inference Script
### Generates MIDI samples from a trained EBT checkpoint, then synthesizes WAV.
###
### Run from project root:
###   sbatch job_scripts/mus/inference/ebt.sh
###
### Overridable via environment or CLI flags:
###   CHECKPOINT=<path>  sbatch job_scripts/mus/inference/ebt.sh
###   sbatch job_scripts/mus/inference/ebt.sh --num_samples 10 --ebt_num_steps 3

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=./logs/slurm_%j.out

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

export PYTHONPATH="${PROJECT_ROOT}:${HOME}/music-EBT/data/mus/symbolic:$PYTHONPATH"
export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}" || exit 1

SCRATCH_LOGS_DIR="${HOME}/orcd/scratch/rebcecca/music_EBT_logs"

# ── Config (all overridable) ─────────────────────────────────────────────────
MODEL_SIZE="small"
TOKENIZER_TYPE="REMI"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
GENERATION_LENGTH="${GENERATION_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MCMC_NUM_STEPS="${MCMC_NUM_STEPS:-}"
EBT_ADVANCED="${EBT_ADVANCED:-}"
USE_TEST_SPLIT="${USE_TEST_SPLIT:-}"
SEED="${SEED:-}"
CHECKPOINT="${CHECKPOINT:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --tokenizer_type)    TOKENIZER_TYPE="$2";         shift 2 ;;
        --model_size)        MODEL_SIZE="$2";             shift 2 ;;
        --checkpoint)        CHECKPOINT="$2";             shift 2 ;;
        --num_samples)       NUM_SAMPLES="$2";            shift 2 ;;
        --generation_length) GENERATION_LENGTH="$2";      shift 2 ;;
        --temperature)       TEMPERATURE="$2";            shift 2 ;;
        --top_p)             TOP_P="$2";                  shift 2 ;;
        --seed)              SEED="$2";                   shift 2 ;;
        --mcmc_num_steps)    MCMC_NUM_STEPS="$2";            shift 2 ;;
        --ebt_advanced)      EBT_ADVANCED="--ebt_advanced"; shift ;;
        --use_test_split)    USE_TEST_SPLIT="--use_test_split"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Tokenizer slug (matches ebt_s1.sh naming convention).
case "${TOKENIZER_TYPE}" in
    REMI)                       TOK_SLUG="remi" ;;
    Anticipation-Arrival-Time)  TOK_SLUG="ant-at-full" ;;
    Anticipation-Vanilla)       TOK_SLUG="ant-at-ar" ;;
    Anticipation-Interarrival)  TOK_SLUG="ant-ia-full" ;;
    *)                          TOK_SLUG=$(echo "${TOKENIZER_TYPE}" | tr '[:upper:]' '[:lower:]') ;;
esac

BASE_RUN_NAME="ebt-symb-${MODEL_SIZE}-${TOK_SLUG}-s1"

# ── Checkpoint auto-detection ─────────────────────────────────────────────────
# If CHECKPOINT is not set, find the checkpoint with the lowest valid_loss
# across all training runs matching this model/tokenizer combination.
if [[ -z "${CHECKPOINT}" ]]; then
    BEST_CKPT=""
    BEST_LOSS="9999"
    for dir in "${SCRATCH_LOGS_DIR}/checkpoints/${BASE_RUN_NAME}"*/; do
        [[ -d "${dir}" ]] || continue
        for ckpt in "${dir}"epoch=*.ckpt; do
            [[ -f "${ckpt}" ]] || continue
            loss=$(echo "${ckpt}" | grep -oE "valid_loss=[0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+$")
            [[ -z "${loss}" ]] && continue
            if (( $(echo "${loss} < ${BEST_LOSS}" | bc -l) )); then
                BEST_LOSS="${loss}"
                BEST_CKPT="${ckpt}"
            fi
        done
    done
    if [[ -z "${BEST_CKPT}" ]]; then
        echo "❌ No checkpoint found for '${BASE_RUN_NAME}'."
        echo "   Set CHECKPOINT=<path> or train a model first."
        exit 1
    fi
    CHECKPOINT="${BEST_CKPT}"
    echo "Auto-selected checkpoint (val_loss=${BEST_LOSS}): ${CHECKPOINT}"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRATCH_LOGS_DIR}/inference/ebt_${TOK_SLUG}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

SLURM_LOG="./logs/slurm_ebt_${TIMESTAMP}.log"
mkdir -p "$(dirname "${SLURM_LOG}")"

echo "=========================================="
echo "EBT Symbolic Music — Inference"
echo "=========================================="
echo "Checkpoint:   ${CHECKPOINT}"
echo "Tokenizer:    ${TOKENIZER_TYPE}"
echo "Generate:     ${GENERATION_LENGTH} tokens × ${NUM_SAMPLES} samples"
echo "Sampling:     temp=${TEMPERATURE}, top_p=${TOP_P}"
echo "EBT advanced: ${EBT_ADVANCED:-off}"
echo "Output:       ${OUTPUT_DIR}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null)"
echo "=========================================="

# ── Run inference ─────────────────────────────────────────────────────────────
python "${PROJECT_ROOT}/inference/mus/infer_ebt.py" \
    --checkpoint "${CHECKPOINT}" \
    --num_samples "${NUM_SAMPLES}" \
    --generation_length "${GENERATION_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    ${MCMC_NUM_STEPS:+--mcmc_num_steps "${MCMC_NUM_STEPS}"} \
    ${EBT_ADVANCED} \
    ${USE_TEST_SPLIT} \
    ${SEED:+--seed "${SEED}"} \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda

INFER_EXIT=$?
if [[ ${INFER_EXIT} -ne 0 ]]; then
    echo "❌ Inference failed (exit ${INFER_EXIT})"
    exit ${INFER_EXIT}
fi

# ── Upload to wandb ───────────────────────────────────────────────────────────
echo ""
echo "Uploading to wandb..."
python "${PROJECT_ROOT}/inference/mus/upload_to_wandb.py" \
    "${OUTPUT_DIR}" \
    --wandb_project "music_inference_ebt" \
    --wandb_entity "rceccatelli-eth-z-rich"

echo ""
echo "=========================================="
echo "✅ Inference complete"
echo "   Output: ${OUTPUT_DIR}"
echo "=========================================="
