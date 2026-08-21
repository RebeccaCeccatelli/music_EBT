#!/bin/bash
#
# Launch the EBT Gradio demo on a GPU node with auto-shutdown.
# Run from the project root on the login node:
#
#   bash demo/launch_demo.sh           # SSH-tunnel access
#   bash demo/launch_demo.sh --share   # public gradio.live URL
#
# Auto-shuts down after IDLE_MINUTES of no user activity (generate/preview calls).

IDLE_MINUTES=20

# ── Compute-node half ─────────────────────────────────────────────────────────
if [[ -n "$SLURM_JOB_ID" ]]; then
    PROJECT_ROOT="${EBT_PROJECT_ROOT}"
    cd "$PROJECT_ROOT" || { echo "❌ Cannot cd to $PROJECT_ROOT"; exit 1; }

    export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/data/mus/symbolic:$PYTHONPATH"
    export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
    export PYTHONUNBUFFERED=1

    PORT=7860
    NODE=$(hostname)
    IDLE_SECS=$(( IDLE_MINUTES * 60 ))

    echo ""
    echo "=========================================="
    echo " EBT Demo  ·  node: ${NODE}"
    if [[ -n "${SHARE_FLAG}" ]]; then
        echo " Mode: public (gradio.live URL will appear below)"
    else
        echo " On your laptop, run:"
        echo "   ssh -L ${PORT}:localhost:${PORT} -J rebcecca@eosloan.mit.edu rebcecca@${NODE}"
        echo " Then open: http://localhost:${PORT}"
    fi
    echo " Auto-shutdown after ${IDLE_MINUTES} min of inactivity."
    echo "=========================================="

    LOG=$(mktemp /tmp/ebt_demo_XXXXXX.log)
    python demo/app.py --port "${PORT}" ${SHARE_FLAG} > "${LOG}" 2>&1 &
    DEMO_PID=$!

    tail -f "${LOG}" &
    TAIL_PID=$!

    # Watchdog: count generate/preview HTTP requests; shutdown when idle too long
    PREV_COUNT=0
    LAST_ACTIVE=$(date +%s)
    while kill -0 "${DEMO_PID}" 2>/dev/null; do
        sleep 30
        COUNT=$(grep -Ec 'queue/join|run/predict' "${LOG}" 2>/dev/null) || COUNT=0
        if [[ "${COUNT}" -ne "${PREV_COUNT}" ]]; then
            LAST_ACTIVE=$(date +%s)
            PREV_COUNT=${COUNT}
        fi
        NOW=$(date +%s)
        if (( NOW - LAST_ACTIVE >= IDLE_SECS )); then
            echo ""
            echo "⏰ ${IDLE_MINUTES} min of inactivity — shutting down demo."
            kill "${DEMO_PID}" 2>/dev/null
            break
        fi
    done

    kill "${TAIL_PID}" 2>/dev/null
    rm -f "${LOG}"
    exit 0
fi

# ── Login-node half ───────────────────────────────────────────────────────────
export SHARE_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --share) export SHARE_FLAG="--share" ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT="$(realpath "$0")"
export EBT_PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Requesting GPU node (partition: mit_preemptable)…"
srun --gpus=1 --mem=40GB --time=3:00:00 \
     --partition=mit_preemptable --account=mit_general \
     bash "${SCRIPT}"
