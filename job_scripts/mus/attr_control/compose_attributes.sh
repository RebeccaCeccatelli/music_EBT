#!/bin/bash
### Test R3's compositional "Reduce" step: guide generation toward multiple
### attribute targets at once (see attribute_control/compose_attributes.py).
###
### Run from project root:
###   REGRESSOR_CKPTS=<ckpt1>,<ckpt2> TARGET_DELTAS_STD=-1.5,1.5 \
###   PROMPT_INDICES=8484,9938,12623 sbatch job_scripts/mus/attr_control/compose_attributes.sh

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:45:00
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

if [[ -z "${REGRESSOR_CKPTS}" ]]; then
    echo "❌ Set REGRESSOR_CKPTS=<ckpt1>,<ckpt2>[,...]"
    exit 1
fi
if [[ -z "${PROMPT_INDICES}" ]]; then
    echo "❌ Set PROMPT_INDICES=<comma list>"
    exit 1
fi

WEIGHTS_ARGS=()
if [[ -n "${WEIGHTS}" ]]; then
    WEIGHTS_ARGS=(--weights "${WEIGHTS}")
fi

# ALL_SIGN_COMBOS=1 with TARGET_DELTAS_STD_MAG=<mag1>,<mag2>[,...] tests every
# +/- combination against each baseline (e.g. 2 attributes -> 4 combos), instead
# of the single fixed TARGET_DELTAS_STD.
COMBO_ARGS=()
if [[ "${ALL_SIGN_COMBOS:-0}" == "1" ]]; then
    if [[ -z "${TARGET_DELTAS_STD_MAG}" ]]; then
        echo "❌ ALL_SIGN_COMBOS=1 requires TARGET_DELTAS_STD_MAG=<mag1>,<mag2>[,...]"
        exit 1
    fi
    COMBO_ARGS=(--all_sign_combos --target_deltas_std_mag "${TARGET_DELTAS_STD_MAG}")
elif [[ -n "${TARGET_DELTAS_STD_LIST}" ]]; then
    # Explicit curated combo list, e.g. an intensity-level sweep:
    # "d0_0,d0_1;d1_0,d1_1;..." — =value since it can start with "-".
    COMBO_ARGS=(--target_deltas_std_list="${TARGET_DELTAS_STD_LIST}")
elif [[ -z "${TARGET_DELTAS_STD}" ]]; then
    echo "❌ Set TARGET_DELTAS_STD, TARGET_DELTAS_STD_LIST, or ALL_SIGN_COMBOS=1 with TARGET_DELTAS_STD_MAG"
    exit 1
else
    # =value (not a separate argv token) since deltas can start with "-",
    # which argparse would otherwise misread as a new flag.
    COMBO_ARGS=(--target_deltas_std="${TARGET_DELTAS_STD}")
fi

# LAMS=<l1>,<l2>[,...] sweeps multiple lambdas (each combo x each lambda,
# against the same baseline) instead of the single fixed LAM.
LAM_ARGS=(--lam "${LAM:-1.0}")
if [[ -n "${LAMS}" ]]; then
    LAM_ARGS=(--lams "${LAMS}")
fi

python "${PROJECT_ROOT}/attribute_control/compose_attributes.py" \
    --regressor_checkpoints "${REGRESSOR_CKPTS}" \
    "${COMBO_ARGS[@]}" \
    --prompt_indices "${PROMPT_INDICES}" \
    "${LAM_ARGS[@]}" \
    "${WEIGHTS_ARGS[@]}" \
    --gen_len "${GEN_LEN:-256}" \
    --seed "${SEED:-0}" \
    --wandb_project "${WANDB_PROJECT:-mus_symb_attr_control}" \
    --wandb_run_name "${WANDB_RUN_NAME:-}"
