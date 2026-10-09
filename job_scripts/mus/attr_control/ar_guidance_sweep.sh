#!/bin/bash
### Plug-and-play attribute-guidance sweep for a frozen AR baseline (GPT-2 /
### Llama) — the head-to-head counterpart to listen_density_sweep.sh's EBT
### sweeps. See attribute_control/ar_guidance_sweep.py for the methods.
###
### Run from project root:
###   METHOD=tilt      MODEL=llama sbatch job_scripts/mus/attr_control/ar_guidance_sweep.sh
###   METHOD=best_of_n MODEL=gpt2  sbatch job_scripts/mus/attr_control/ar_guidance_sweep.sh
###   METHOD=pplm MODEL=llama ATTRIBUTES=velocity \
###       REGRESSOR_CKPTS=<llama-space best.pt> sbatch job_scripts/mus/attr_control/ar_guidance_sweep.sh
###
### Optional: ATTRIBUTES (default velocity,duration,pitch_register), STRENGTHS,
### MODEL_CKPT (overrides MODEL), PROMPT_INDICES, OUT_DIR, WANDB_PROJECT.

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1  # L40S only: this torch build (2.4+cu121) has no kernels for the RTX Pro 6000 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
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
CKPT_DIR="${SCRATCH_LOGS_DIR}/checkpoints"

METHOD="${METHOD:-tilt}"
MODEL="${MODEL:-llama}"
# Final REMI baseline checkpoints (docs/STATUS.md) — picked explicitly, never
# by lowest valid_loss in the filename.
case "${MODEL}" in
    llama) DEFAULT_CKPT="${CKPT_DIR}/baseline-llama-small-remi-job21725352_2026-09-01_03-14-23_/epoch=epoch=185-step=step=99660-valid_loss=valid_loss=0.5239.ckpt" ;;
    gpt2)  DEFAULT_CKPT="${CKPT_DIR}/baseline-hf-gpt2-small-remi-job21688861_2026-09-01_00-24-25_/epoch=epoch=185-step=step=99660-valid_loss=valid_loss=0.4756.ckpt" ;;
    *)     echo "❌ MODEL must be llama or gpt2 (or set MODEL_CKPT)"; exit 1 ;;
esac
MODEL_CKPT="${MODEL_CKPT:-${DEFAULT_CKPT}}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "❌ Checkpoint not found: ${MODEL_CKPT}"
    exit 1
fi

# The same 16 prompts as the EBT REMI sweeps (attribute_control/sweep_tables/),
# so every system is scored on identical prompts.
PROMPT_INDICES="${PROMPT_INDICES:-1326,3107,4563,4579,7157,8484,9235,9938,11732,12623,13268,13781,15617,15922,16537,16753}"
OUT_DIR="${OUT_DIR:-${SCRATCH_LOGS_DIR}/attr_control/ar_guidance/${MODEL}_${METHOD}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}}"

EXTRA_ARGS=()
[[ -n "${STRENGTHS}" ]] && EXTRA_ARGS+=(--strengths "${STRENGTHS}")
[[ -n "${REGRESSOR_CKPTS}" ]] && EXTRA_ARGS+=(--regressor_checkpoints "${REGRESSOR_CKPTS}")
[[ -n "${WANDB_PROJECT}" ]] && EXTRA_ARGS+=(--wandb_project "${WANDB_PROJECT}")

scontrol update JobId="${SLURM_JOB_ID}" JobName="ar-${MODEL}-${METHOD}" 2>/dev/null || true

echo "AR guidance sweep: model=${MODEL} method=${METHOD}"
echo "Checkpoint: ${MODEL_CKPT}"
echo "Output:     ${OUT_DIR}"

python "${PROJECT_ROOT}/attribute_control/ar_guidance_sweep.py" \
    --model_checkpoint "${MODEL_CKPT}" \
    --method "${METHOD}" \
    --attributes "${ATTRIBUTES:-velocity,duration,pitch_register}" \
    --prompt_indices "${PROMPT_INDICES}" \
    --target_deltas_std="${TARGET_DELTAS_STD:--2,-1,-0.5,0.5,1,2}" \
    --baseline_repeats "${BASELINE_REPEATS:-3}" \
    --gen_len "${GEN_LEN:-256}" \
    --temperature "${TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.9}" \
    --seed "${SEED:-0}" \
    --bigram_table "${SCRATCH_LOGS_DIR}/attr_control/_bigram_table_REMI.npy" \
    --out_dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"
