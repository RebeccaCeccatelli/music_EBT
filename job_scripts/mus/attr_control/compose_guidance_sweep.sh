#!/bin/bash
### Multi-attribute (compositional) guidance sweep: attribute_control/compose_guidance_sweep.py.
###   MODEL_CKPT=<ckpt> METHOD=r3|pplm|tilt|best_of_n [REGRESSOR_CKPTS=<a>,<b>] \
###       sbatch job_scripts/mus/attr_control/compose_guidance_sweep.sh
### Optional: ATTRIBUTES (default velocity,duration), STRENGTHS, TARGET_DELTAS_STD_MAG
### (default 1,2), PROMPT_INDICES (default: the 16 REMI sweep prompts), OUT_DIR.

#SBATCH --nodes=1
#SBATCH --gpus=l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --output=./logs/slurm_%j.out

find_project_root() {
    local dir="$1"
    for ((i=0; i<10; i++)); do
        if [[ -f "${dir}/train_model.py" ]]; then echo "${dir}"; return 0; fi
        dir="$(dirname "${dir}")"
    done
    echo ""; return 1
}
PROJECT_ROOT="$(find_project_root "$(pwd)")"
[[ -z "${PROJECT_ROOT}" ]] && { echo "❌ Error: Could not find project root."; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${HOME}/music-EBT/data/mus/symbolic:$PYTHONPATH"
export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}" || exit 1

SCRATCH_LOGS_DIR="${HOME}/orcd/scratch/rebcecca/music_EBT_logs"
: "${MODEL_CKPT:?MODEL_CKPT=<checkpoint>}" "${METHOD:?METHOD=r3|pplm|tilt|best_of_n}"
[[ -f "${MODEL_CKPT}" ]] || { echo "❌ Checkpoint not found: ${MODEL_CKPT}"; exit 1; }
case "${MODEL_CKPT}" in
    *ebt-symb*) MODEL=ebt ;; *baseline-llama*) MODEL=llama ;; *baseline-hf-gpt2*) MODEL=gpt2 ;; *) MODEL=model ;;
esac
# Default: the 16 prompts of the single-attribute REMI sweeps.
PROMPT_INDICES="${PROMPT_INDICES:-1326,3107,4563,4579,7157,8484,9235,9938,11732,12623,13268,13781,15617,15922,16537,16753}"
OUT_DIR="${OUT_DIR:-${SCRATCH_LOGS_DIR}/attr_control/compose_guidance/${MODEL}_${METHOD}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}}"

EXTRA_ARGS=()
[[ -n "${STRENGTHS}" ]] && EXTRA_ARGS+=(--strengths "${STRENGTHS}")
[[ -n "${REGRESSOR_CKPTS}" ]] && EXTRA_ARGS+=(--regressor_checkpoints "${REGRESSOR_CKPTS}")

scontrol update JobId="${SLURM_JOB_ID}" JobName="compose-${MODEL}-${METHOD}" 2>/dev/null || true
echo "Compose sweep: model=${MODEL} method=${METHOD}"
echo "Checkpoint: ${MODEL_CKPT}"
echo "Output:     ${OUT_DIR}"

python "${PROJECT_ROOT}/attribute_control/compose_guidance_sweep.py" \
    --model_checkpoint "${MODEL_CKPT}" \
    --method "${METHOD}" \
    --attributes "${ATTRIBUTES:-velocity,duration}" \
    --target_deltas_std_mag "${TARGET_DELTAS_STD_MAG:-1,2}" \
    --prompt_indices "${PROMPT_INDICES}" \
    --baseline_repeats "${BASELINE_REPEATS:-3}" \
    --gen_len "${GEN_LEN:-256}" \
    --temperature "${TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.9}" \
    --seed "${SEED:-0}" \
    --bigram_table "${SCRATCH_LOGS_DIR}/attr_control/_bigram_table_REMI.npy" \
    --out_dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"
