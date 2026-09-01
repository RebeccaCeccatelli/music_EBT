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

# TARGET_DELTAS_STD (offsets in corpus stdev units) takes priority over
# TARGET_DELTAS (raw-unit offsets), which takes priority over TARGETS (shared
# absolute values across prompts), when set.
TARGET_ARGS=(--targets "${TARGETS:-0.05,0.10,0.15,0.20,0.25}")
if [[ -n "${TARGET_DELTAS}" ]]; then
    # =value (not a separate argv token) since deltas can start with "-",
    # which argparse would otherwise misread as a new flag.
    TARGET_ARGS=(--target_deltas="${TARGET_DELTAS}")
fi
if [[ -n "${TARGET_DELTAS_STD}" ]]; then
    TARGET_ARGS=(--target_deltas_std="${TARGET_DELTAS_STD}")
fi

# STEP_GATING=0 reproduces the pre-gating behavior (guidance fires on every
# generation step) for A/B comparison against the default gated mechanism.
GATING_ARGS=()
if [[ "${STEP_GATING:-1}" == "0" ]]; then
    GATING_ARGS=(--no_step_gating)
fi

# LAMBDA_TAPER=1 holds lambda at full strength for LAMBDA_TAPER_HOLD_FRAC of
# generation, then decays it to LAMBDA_TAPER_FLOOR (a fraction of lambda) —
# see --lambda_taper in listen_density_sweep.py for the motivation.
TAPER_ARGS=()
if [[ "${LAMBDA_TAPER:-0}" == "1" ]]; then
    TAPER_ARGS=(--lambda_taper
                --lambda_taper_hold_frac "${LAMBDA_TAPER_HOLD_FRAC:-0.4}"
                --lambda_taper_floor "${LAMBDA_TAPER_FLOOR:-0.2}")
fi

python "${PROJECT_ROOT}/attribute_control/listen_density_sweep.py" \
    --regressor_checkpoint "${REGRESSOR_CKPT}" \
    --n_prompts "${N_PROMPTS:-5}" \
    "${TARGET_ARGS[@]}" \
    --baseline_repeats "${BASELINE_REPEATS:-5}" \
    --lambdas "${LAMBDAS:-0.25,0.5,1,2,4}" \
    --gen_len "${GEN_LEN:-256}" \
    --temperature "${TEMPERATURE:-0.7}" \
    --top_p "${TOP_P:-0.9}" \
    --seed "${SEED:-0}" \
    "${PROMPT_ARGS[@]}" \
    "${GATING_ARGS[@]}" \
    "${TAPER_ARGS[@]}" \
    --wandb_project "${WANDB_PROJECT:-mus_symb_attr_control}" \
    --wandb_run_name "${WANDB_RUN_NAME:-}"
