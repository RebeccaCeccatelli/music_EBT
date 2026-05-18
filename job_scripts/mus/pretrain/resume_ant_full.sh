#!/bin/bash
### Resume ant-full (Anticipation-Arrival-Time) baseline training for GPT2 and Llama.
###
### Submits both models explicitly from the last stable checkpoints (step 1700/2600,
### val_loss ~2.13/2.03). Passes --resume_training_ckpt to bypass auto-detection,
### which would otherwise pick newer but unstable run directories.
###
### Run from the project root:
###   bash job_scripts/mus/pretrain/resume_ant_full.sh

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

SCRATCH_LOGS_DIR="${HOME}/orcd/scratch/rebcecca/music_EBT_logs"

# Last stable checkpoints before training became unstable (LR scheduler reset bug).
# The LR scheduler fix is now committed; these are clean starting points.
GPT2_CKPT="${SCRATCH_LOGS_DIR}/checkpoints/baseline-hf-gpt2-small-ant-at-full-job13696815_2026-05-10_14-26-02_/last.ckpt"
LLAMA_CKPT="${SCRATCH_LOGS_DIR}/checkpoints/baseline-llama-small-ant-at-full-job13696816_2026-05-10_14-26-17_/last.ckpt"

for ckpt in "${GPT2_CKPT}" "${LLAMA_CKPT}"; do
    if [[ ! -f "${ckpt}" ]]; then
        echo "❌ Checkpoint not found: ${ckpt}"
        exit 1
    fi
done

mkdir -p "${SCRATCH_LOGS_DIR}/slurm/mus/"

echo "Submitting GPT2 ant-full resume..."
GPT2_JOB=$(sbatch \
    --job-name=gpt2-ant-full \
    --output="${SCRATCH_LOGS_DIR}/slurm/mus/pretrain_gpt2_ant_full_%j.log" \
    job_scripts/mus/pretrain/baseline_hf_gpt2_transformer.sh \
    --tokenizer_type Anticipation-Arrival-Time \
    --resume_training_ckpt "${GPT2_CKPT}" \
    | awk '{print $4}')
echo "  → Job ID: ${GPT2_JOB}"

echo "Submitting Llama ant-full resume..."
LLAMA_JOB=$(sbatch \
    --job-name=llama-ant-full \
    --output="${SCRATCH_LOGS_DIR}/slurm/mus/pretrain_llama_ant_full_%j.log" \
    job_scripts/mus/pretrain/baseline_llama_transformer.sh \
    --tokenizer_type Anticipation-Arrival-Time \
    --resume_training_ckpt "${LLAMA_CKPT}" \
    | awk '{print $4}')
echo "  → Job ID: ${LLAMA_JOB}"

echo ""
echo "Both jobs submitted. Monitor with:"
echo "  squeue -u ${USER}"
echo "  tail -f ${SCRATCH_LOGS_DIR}/slurm/mus/pretrain_gpt2_ant_full_${GPT2_JOB}.log"
echo "  tail -f ${SCRATCH_LOGS_DIR}/slurm/mus/pretrain_llama_ant_full_${LLAMA_JOB}.log"
