#!/bin/bash
### Launcher: submit one inference job per baseline model.
### Run from the project root:  bash job_scripts/mus/inference/baselines_interactive.sh
###
### Override any variable at call time:
###   CHECKPOINT_GPT2=<path> CHECKPOINT_LLAMA=<path> bash baselines_interactive.sh

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
cd "${PROJECT_ROOT}" || exit 1

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

SINGLE_SCRIPT="${PROJECT_ROOT}/job_scripts/mus/inference/baselines_interactive_single.sh"

# ── Checkpoints ────────────────────────────────────────────────────────────────
# REMI (validation loss ~0.66 / 0.65)
# CHECKPOINT_GPT2="${CHECKPOINT_GPT2:-logs/checkpoints/baseline-hf-gpt2-small-remi-job13735276_2026-05-11_11-29-14_/epoch=epoch=8-step=step=4788-valid_loss=valid_loss=0.6630.ckpt}"
# CHECKPOINT_LLAMA="${CHECKPOINT_LLAMA:-logs/checkpoints/baseline-llama-small-remi-job13735284_2026-05-11_11-29-25_/epoch=epoch=15-step=step=8540-valid_loss=valid_loss=0.6461.ckpt}"

# Anticipation-Arrival-Time (validation loss ~2.13 / 2.03)
CHECKPOINT_GPT2="${CHECKPOINT_GPT2:-logs/checkpoints/baseline-hf-gpt2-small-ant-at-full-job13696815_2026-05-10_14-26-02_/epoch=epoch=0-step=step=1700-valid_loss=valid_loss=2.1304.ckpt}"
CHECKPOINT_LLAMA="${CHECKPOINT_LLAMA:-logs/checkpoints/baseline-llama-small-ant-at-full-job13696816_2026-05-10_14-26-17_/epoch=epoch=0-step=step=2600-valid_loss=valid_loss=2.0254.ckpt}"

# ── Shared generation config ───────────────────────────────────────────────────
PROMPT_LENGTH="${PROMPT_LENGTH:-128}"
GENERATION_LENGTH="${GENERATION_LENGTH:-384}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
USE_TEST_SPLIT="${USE_TEST_SPLIT:-}"
WANDB_PROJECT="${WANDB_PROJECT:-music_inference_baselines}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p logs/slurm/mus/

SHARED_EXPORTS="PROMPT_LENGTH=${PROMPT_LENGTH},GENERATION_LENGTH=${GENERATION_LENGTH},NUM_SAMPLES=${NUM_SAMPLES},USE_TEST_SPLIT=${USE_TEST_SPLIT},WANDB_PROJECT=${WANDB_PROJECT},WANDB_ENTITY=${WANDB_ENTITY}"

echo "Submitting GPT2 inference job..."
GPT2_JOB=$(sbatch \
    --job-name=inf-gpt2 \
    --export=ALL,MODEL_NAME=baseline_hf_gpt2_transformer,MODEL_SHORT=gpt2,CHECKPOINT=${CHECKPOINT_GPT2},${SHARED_EXPORTS} \
    "${SINGLE_SCRIPT}" | awk '{print $4}')
echo "  → Job ID: ${GPT2_JOB}"

echo "Submitting Llama inference job..."
LLAMA_JOB=$(sbatch \
    --job-name=inf-llama \
    --export=ALL,MODEL_NAME=baseline_llama_transformer,MODEL_SHORT=llama,CHECKPOINT=${CHECKPOINT_LLAMA},${SHARED_EXPORTS} \
    "${SINGLE_SCRIPT}" | awk '{print $4}')
echo "  → Job ID: ${LLAMA_JOB}"

echo ""
echo "Both jobs submitted. Monitor with:"
echo "  squeue -u ${USER}"
echo "  tail -f logs/slurm/mus/baselines_interactive_inf-gpt2_${GPT2_JOB}.log"
echo "  tail -f logs/slurm/mus/baselines_interactive_inf-llama_${LLAMA_JOB}.log"
