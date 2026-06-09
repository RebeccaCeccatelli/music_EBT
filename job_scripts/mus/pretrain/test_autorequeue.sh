#!/bin/bash
### Test auto-requeue logic with minimal overhead
### This script simulates training but exits quickly so we can test the requeue logic

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:05:00
#SBATCH --mem=4GB
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --output=./logs/slurm_test_requeue_%j.out

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH_LOGS_DIR="${HOME}/orcd/scratch/rebcecca/music_EBT_logs"

export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
cd "${PROJECT_ROOT}" || exit 1

BASE_RUN_NAME="test-autorequeue"
FULL_RUN_NAME="${BASE_RUN_NAME}-job${SLURM_JOB_ID:-local}"
MAX_STEPS=1000

echo "=========================================="
echo "Test Auto-Requeue Logic"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:  ${FULL_RUN_NAME}"
echo "Time limit: 5 minutes (will timeout and trigger requeue)"
echo "=========================================="

# Auto-resume from last checkpoint
PREV_CKPT_DIR=$(ls -td "${SCRATCH_LOGS_DIR}/checkpoints/${BASE_RUN_NAME}"* 2>/dev/null | head -1)
RESUME_CKPT=""
if [[ -n "${PREV_CKPT_DIR}" && -f "${PREV_CKPT_DIR}/last.ckpt" ]]; then
    RESUME_CKPT="${PREV_CKPT_DIR}/last.ckpt"
    echo "Auto-resuming from: ${RESUME_CKPT}"
else
    echo "No previous checkpoint found, starting fresh"
    mkdir -p "${SCRATCH_LOGS_DIR}/checkpoints/${FULL_RUN_NAME}"
fi

# Simulate training: create dummy checkpoint every 30 seconds
# Script will run for ~4.5 minutes, then timeout at 5 minutes
STEP=0
for i in {1..9}; do
    echo "Simulating training step $((i * 100))..."
    STEP=$((i * 100))

    # Create a dummy checkpoint
    CKPT_DIR="${SCRATCH_LOGS_DIR}/checkpoints/${FULL_RUN_NAME}"
    mkdir -p "${CKPT_DIR}"
    echo "dummy" > "${CKPT_DIR}/last.ckpt"
    echo "step=${STEP}" >> "${CKPT_DIR}/last.ckpt"

    sleep 30
done

echo "Training simulation complete at step ${STEP}"
TRAIN_EXIT_CODE=0

# Auto-resubmit logic (same as baseline scripts)
_do_resubmit() {
    echo "Resubmitting..."
    sbatch "${BASH_SOURCE[0]}"
}

if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    LAST_STEP=$(ls "${SCRATCH_LOGS_DIR}/checkpoints/${BASE_RUN_NAME}"*/last.ckpt 2>/dev/null | head -1 | xargs grep -o "step=[0-9]*" | grep -o "[0-9]*" | tail -1)
    if [[ -n "${LAST_STEP}" && "${LAST_STEP}" -ge "${MAX_STEPS}" ]]; then
        echo "Training complete at step ${LAST_STEP}. Not resubmitting."
    else
        echo "Clean exit but incomplete (last step: ${LAST_STEP:-none}). Resubmitting..."
        _do_resubmit
    fi
else
    echo "Training failed (exit ${TRAIN_EXIT_CODE}). Not resubmitting."
    exit ${TRAIN_EXIT_CODE}
fi
