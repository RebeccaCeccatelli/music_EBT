#!/bin/bash
### Generate a listenable grid of density-guided samples across (prompt,
### lambda, target) and log them to wandb as playable audio, to judge by ear
### where guidance strength starts corrupting musical quality.
###
### Run from project root:
###   REGRESSOR_CKPT=<path/to/best.pt> sbatch job_scripts/mus/attr_control/listen_density_sweep.sh

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
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

python "${PROJECT_ROOT}/attribute_control/listen_density_sweep.py" \
    --regressor_checkpoint "${REGRESSOR_CKPT}" \
    --n_prompts "${N_PROMPTS:-5}" \
    --targets "${TARGETS:-0.05,0.10,0.15,0.20,0.25}" \
    --lambdas "${LAMBDAS:-0.25,0.5,1,2,4}" \
    --gen_len "${GEN_LEN:-256}" \
    --seed "${SEED:-0}" \
    --wandb_project "${WANDB_PROJECT:-mus_symb_attr_control}" \
    --wandb_run_name "${WANDB_RUN_NAME:-}"
