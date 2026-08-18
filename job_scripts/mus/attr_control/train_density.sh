#!/bin/bash
### Train the NoteDensityRegressor on frozen EBT embeddings.
###
### The embedding table is extracted from the best available EBT checkpoint
### (auto-selected by lowest validation loss) and kept frozen.  The MLP is
### trained with MSE loss to predict note density from the mean token embedding.
###
### Run from project root:
###   sbatch job_scripts/mus/attr_control/train_density.sh
###
### Override tokenizer:
###   TOKENIZER_TYPE=Anticipation-Arrival-Time sbatch job_scripts/mus/attr_control/train_density.sh

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=60GB
#SBATCH --partition=mit_preemptable
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

# ── Config (all overridable via environment) ────────────────────────────────
TOKENIZER_TYPE="${TOKENIZER_TYPE:-REMI}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-500000}"
CHECKPOINT="${CHECKPOINT:-}"

# ── Derive model slug from tokenizer (matches pretrain script naming) ────────
case "${TOKENIZER_TYPE}" in
    REMI)                       TOK_SLUG="remi";        MODEL_SLUG="remi" ;;
    Anticipation-Arrival-Time)  TOK_SLUG="ant-at-full"; MODEL_SLUG="ant-at-full" ;;
    *)                          TOK_SLUG="$(echo "${TOKENIZER_TYPE}" | tr '[:upper:]' '[:lower:]')"
                                MODEL_SLUG="${TOK_SLUG}" ;;
esac

BASE_RUN_NAME="ebt-symb-small-${TOK_SLUG}-s1"

# ── Auto-select best EBT checkpoint if not explicitly provided ───────────────
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
        echo "   Set CHECKPOINT=<path> or train an EBT model first."
        exit 1
    fi
    CHECKPOINT="${BEST_CKPT}"
    echo "Auto-selected EBT checkpoint (val_loss=${BEST_LOSS}): ${CHECKPOINT}"
fi

scontrol update JobId="${SLURM_JOB_ID}" JobName="attr-density-${TOK_SLUG}" 2>/dev/null || true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}
OUTPUT_DIR="${SCRATCH_LOGS_DIR}/attr_control/density_regressor_${MODEL_SLUG}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Density Regressor Training"
echo "=========================================="
echo "Tokenizer:    ${TOKENIZER_TYPE}"
echo "EBT ckpt:     ${CHECKPOINT}"
echo "Output:       ${OUTPUT_DIR}"
echo "Epochs:       ${EPOCHS}   Batch: ${BATCH_SIZE}   LR: ${LR}"
echo "Max train:    ${MAX_TRAIN_SAMPLES}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null)"
echo "=========================================="

python "${PROJECT_ROOT}/attribute_control/train_density_regressor.py" \
    --checkpoint "${CHECKPOINT}" \
    --tokenizer_type "${TOKENIZER_TYPE}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --num_workers 8 \
    --device cuda

EXIT_CODE=$?
if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "❌ Training failed (exit ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi

echo ""
echo "=========================================="
echo "✅ Density regressor training complete"
echo "   Best checkpoint: ${OUTPUT_DIR}/best.pt"
echo "=========================================="
