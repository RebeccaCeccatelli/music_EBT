#!/bin/bash
### Benchmark a NoteDensityRegressor's actual steering effect on generation.
###
### Sweeps a grid of density targets against a sample of validation prompts,
### measures achieved density, and reports correlation/direction metrics.
### Reusable across regressor checkpoints — the EBT checkpoint is read from
### the regressor's own metadata unless overridden.
###
### Run from project root:
###   REGRESSOR_CKPT=<path/to/best.pt> sbatch job_scripts/mus/attr_control/benchmark_density.sh

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

LAMBDA_ARGS=(--lambda_density "${LAMBDA_DENSITY:-1.0}")
if [[ -n "${LAMBDAS}" ]]; then
    LAMBDA_ARGS=(--lambdas "${LAMBDAS}")
fi

# Defaults to the curated non-drum melodic prompt pool (built after finding
# ~73% of this dataset's songs contain drums, which compute_density/etc. are
# partly blind to — see attribute_control/note_density.py). Override with
# PROMPT_INDICES=<comma list> for specific songs, or PROMPT_INDICES_FILE=""
# to fall back to unrestricted random sampling.
DEFAULT_PROMPT_POOL="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/attr_control/clean_melodic_prompts.json"
PROMPT_ARGS=()
if [[ -n "${PROMPT_INDICES}" ]]; then
    PROMPT_ARGS=(--prompt_indices "${PROMPT_INDICES}")
elif [[ -n "${PROMPT_INDICES_FILE-${DEFAULT_PROMPT_POOL}}" ]]; then
    PROMPT_ARGS=(--prompt_indices_file "${PROMPT_INDICES_FILE:-${DEFAULT_PROMPT_POOL}}")
fi

python "${PROJECT_ROOT}/attribute_control/benchmark_density_control.py" \
    --regressor_checkpoint "${REGRESSOR_CKPT}" \
    --targets "${TARGETS:-0.05,0.10,0.15,0.20,0.25}" \
    --n_prompts "${N_PROMPTS:-8}" \
    --repeats "${REPEATS:-3}" \
    --gen_len "${GEN_LEN:-128}" \
    "${LAMBDA_ARGS[@]}" \
    --seed "${SEED:-0}" \
    "${PROMPT_ARGS[@]}" \
    --wandb_project "${WANDB_PROJECT:-mus_symb_attr_control}" \
    --wandb_run_name "${WANDB_RUN_NAME:-}"
