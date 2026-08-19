#!/bin/bash
### Test auto-requeue: simulates PL's SIGTERM → save → exit 0 → resubmit cycle.
###
### How to verify it works:
###   1. sbatch this script
###   2. Watch logs/slurm_test_requeue_<jobid>.out for each run — each should print
###      "Resuming from step N" with N increasing across runs
###   3. After ~3-4 runs (MAX_STEPS=25, ~8 steps per 2-min job), the last log
###      should print "Training complete" and NOT resubmit

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:02:00
#SBATCH --mem=1GB
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --signal=TERM@30
#SBATCH --output=./logs/slurm_test_requeue_%j.out

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH_LOGS_DIR="${HOME}/orcd/scratch/rebcecca/music_EBT_logs"

BASE_RUN_NAME="test-autorequeue"
CKPT_DIR="${SCRATCH_LOGS_DIR}/checkpoints/${BASE_RUN_NAME}"
CKPT_FILE="${CKPT_DIR}/last.ckpt"
MAX_STEPS=25

mkdir -p "${CKPT_DIR}"

echo "=========================================="
echo "Test Auto-Requeue"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Time limit: 2 min | TERM signal at T-30s"
echo "Started at: $(date)"
echo "=========================================="

# Load step from previous checkpoint (simulates PL resume).
CURRENT_STEP=0
if [[ -f "${CKPT_FILE}" ]]; then
    CURRENT_STEP=$(grep -oP "step=\K[0-9]+" "${CKPT_FILE}" | tail -1)
    CURRENT_STEP="${CURRENT_STEP:-0}"
    echo "Resuming from step ${CURRENT_STEP}"
else
    echo "No checkpoint found — starting from step 0"
fi

# SIGTERM handler: set a flag so the main loop exits cleanly.
# This mirrors PL's behavior: it catches SIGTERM, saves a checkpoint,
# then exits 0 — so bash sees a clean exit and runs the resubmit logic.
SIGTERM_RECEIVED=0
_handle_sigterm() {
    SIGTERM_RECEIVED=1
}
trap '_handle_sigterm' TERM

# Training simulation: 1 step every 10 seconds.
# With a 2-min job and TERM at T-30s, each run completes ~8 steps before signal.
echo "Training steps (10s each):"
while [[ ${CURRENT_STEP} -lt ${MAX_STEPS} && ${SIGTERM_RECEIVED} -eq 0 ]]; do
    # Backgrounded sleep so the SIGTERM trap fires immediately (not deferred).
    sleep 10 & wait $!
    [[ ${SIGTERM_RECEIVED} -eq 1 ]] && break
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo "  step ${CURRENT_STEP}/${MAX_STEPS}  [$(date +%H:%M:%S)]"
    echo "step=${CURRENT_STEP}" > "${CKPT_FILE}"
done

# Determine exit code for the resubmit logic below.
if [[ ${SIGTERM_RECEIVED} -eq 1 ]]; then
    echo "SIGTERM received at $(date) — saving checkpoint at step ${CURRENT_STEP}"
    echo "step=${CURRENT_STEP}" > "${CKPT_FILE}"
    TRAIN_EXIT_CODE=0
else
    TRAIN_EXIT_CODE=0
fi

# ------------------------------------------------------------------
# Resubmit logic — mirrors production scripts exactly.
#
# Exit 0   + incomplete → resubmit (PL saved on SIGTERM, job ran short).
# Exit 0   + complete   → done, no resubmit.
# Exit 143 + any        → SIGTERM hit bash directly (rare) → resubmit.
# Other                 → crash, do not resubmit.
# ------------------------------------------------------------------
_do_resubmit() {
    echo "Resubmitting with sbatch..."
    sbatch "${BASH_SOURCE[0]}"
}

if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    LAST_STEP=$(grep -oP "step=\K[0-9]+" "${CKPT_FILE}" 2>/dev/null | tail -1)
    LAST_STEP="${LAST_STEP:-0}"
    if [[ ${LAST_STEP} -ge ${MAX_STEPS} ]]; then
        echo "Training complete at step ${LAST_STEP}/${MAX_STEPS}. Done — not resubmitting."
    else
        echo "Clean exit but incomplete (step ${LAST_STEP}/${MAX_STEPS}). Resubmitting..."
        _do_resubmit
    fi
elif [[ ${TRAIN_EXIT_CODE} -eq 143 ]]; then
    echo "SIGTERM hit bash before handler fired. Resubmitting..."
    _do_resubmit
else
    echo "Failed with exit code ${TRAIN_EXIT_CODE}. Not resubmitting."
    exit ${TRAIN_EXIT_CODE}
fi
