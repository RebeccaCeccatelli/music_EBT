#!/bin/bash
### End-to-end test: bash SIGTERM trap → forwards as SIGUSR1 to Python → PL-style requeue.
###
### Simulates exactly what the real training scripts do:
###   bash wraps Python; --signal=TERM@60 fires SIGTERM 60s before wall time;
###   bash catches SIGTERM and forwards it as SIGUSR1 to Python (PL bypasses SIGTERM
###   in SLURM auto-requeue mode; USR1 triggers its checkpoint-and-requeue handler);
###   Python saves checkpoint and exits 0; bash resubmits.
###
### Expected flow:
###   Job 1: ~15 steps (8s each = 2 min), SIGTERM at ~2 min → USR1 → checkpoint → resubmit
###   Job 2: resumes from step 15, completes to step 25
###
### Submit:
###   sbatch job_scripts/debug/test_autorequeue_python.sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
#SBATCH --mem=1GB
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --signal=TERM@60
#SBATCH --output=./logs/slurm_test_pl_%j.out
#SBATCH --job-name=test-pl-sigterm

find_project_root() {
    local dir="$1"
    for ((i=0; i<10; i++)); do
        if [[ -f "${dir}/train_model.py" ]]; then
            echo "${dir}"; return 0
        fi
        dir="$(dirname "${dir}")"
    done
    echo ""; return 1
}

PROJECT_ROOT="$(find_project_root "$(pwd)")"
if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "❌ Could not find project root."; exit 1
fi

export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}" || exit 1

MAX_STEPS=25
CKPT_FILE="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/test_pl_sigterm_ckpt.txt"

echo "=========================================="
echo "test_autorequeue_python — job ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "CKPT_FILE: ${CKPT_FILE}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "=========================================="

_do_resubmit() {
    echo "[bash] Resubmitting..."
    sbatch "${BASH_SOURCE[0]}"
}

# ── SIGTERM trap: catch signal and forward it to Python ───────────────────────
PYTHON_PID=0
SIGTERM_RECEIVED=0

_handle_sigterm() {
    SIGTERM_RECEIVED=1
    echo "[bash] $(date -Iseconds) SIGTERM received — forwarding as SIGUSR1 to Python (PID ${PYTHON_PID})..."
    # PL bypasses SIGTERM in SLURM auto-requeue mode; USR1 triggers its checkpoint-and-requeue handler.
    [[ ${PYTHON_PID} -ne 0 ]] && kill -USR1 "${PYTHON_PID}" 2>/dev/null
}
trap '_handle_sigterm' TERM

# ── Run Python "training" in background so bash can handle signals ─────────────
CKPT_FILE="${CKPT_FILE}" MAX_STEPS="${MAX_STEPS}" STEP_SECS=8 \
    python job_scripts/debug/test_pl_sigterm.py &
PYTHON_PID=$!
wait "${PYTHON_PID}"
TRAIN_EXIT_CODE=$?
# wait returns early when a signal arrives; re-wait for the real exit code
if [[ ${SIGTERM_RECEIVED} -eq 1 ]]; then
    wait "${PYTHON_PID}" 2>/dev/null
    TRAIN_EXIT_CODE=$?
fi

echo "[bash] Python exited with code ${TRAIN_EXIT_CODE} (SIGTERM_RECEIVED=${SIGTERM_RECEIVED})"

# ── Resubmit logic (mirrors real training scripts) ────────────────────────────
if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    LAST_STEP=$(cat "${CKPT_FILE}" 2>/dev/null || echo "0")
    if [[ "${LAST_STEP}" -ge "${MAX_STEPS}" ]]; then
        echo "[bash] ✅ Training complete at step ${LAST_STEP}. Not resubmitting."
        rm -f "${CKPT_FILE}"
    else
        echo "[bash] Clean exit but incomplete (step ${LAST_STEP}/${MAX_STEPS}). Resubmitting..."
        _do_resubmit
    fi
elif [[ ${TRAIN_EXIT_CODE} -eq 143 ]]; then
    echo "[bash] SIGTERM reached bash before Python could handle it. Resubmitting..."
    _do_resubmit
else
    echo "[bash] ❌ Python crashed (exit ${TRAIN_EXIT_CODE}). Not resubmitting."
    exit ${TRAIN_EXIT_CODE}
fi
