#!/bin/bash
### Test R3-guided post-hoc note removal (attribute_control/energy_guided_edit.py)
### against real-time R3 guidance's negative-direction weak spot.
###
### Run from project root:
###   REGRESSOR_CKPT=<path/to/best.pt> PROMPT_INDICES=8484,9938 sbatch job_scripts/mus/attr_control/energy_guided_edit.sh

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
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

if [[ -z "${REGRESSOR_CKPT}" ]]; then
    echo "❌ Set REGRESSOR_CKPT=<path/to/best.pt>"
    exit 1
fi
if [[ -z "${PROMPT_INDICES}" ]]; then
    echo "❌ Set PROMPT_INDICES=<comma list>"
    exit 1
fi

python "${PROJECT_ROOT}/attribute_control/energy_guided_edit.py" \
    --regressor_checkpoint "${REGRESSOR_CKPT}" \
    --prompt_indices "${PROMPT_INDICES}" \
    --target_delta_std "${TARGET_DELTA_STD:--2.0}" \
    --note_type "${NOTE_TYPE:-drum}" \
    --lam "${LAM:-1.0}" \
    --max_removals "${MAX_REMOVALS:-30}" \
    --tolerance "${TOLERANCE:-0.02}" \
    --score_window "${SCORE_WINDOW:-64}" \
    --gen_len "${GEN_LEN:-256}" \
    --seed "${SEED:-0}" \
    --wandb_project "${WANDB_PROJECT:-mus_symb_attr_control}" \
    --wandb_run_name "${WANDB_RUN_NAME:-}"
