#!/bin/bash
### Baseline Llama Transformer Music - Inference Script
### Generate MIDI music using a pretrained baseline llama transformer

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=logs/slurm/mus/baseline-llama-mus-inference_%A-%a.log

### ADDITIONAL RUN INFO ###
#SBATCH --array=0

### LOG INFO ###
#SBATCH --job-name=baseline-llama-mus-inference
export RUN_NAME="baseline-llama-mus-inference"
export MODEL_SIZE="small"

### Get the project root directory
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
    echo "❌ Error: Could not find project root. train_model.py not found in parent directories."
    echo "   Make sure this script is in <project_root>/job_scripts/mus/inference/"
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
export PYTHONUNBUFFERED=1

# Load environment variables from .env file
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Get wandb entity from environment or use default
WANDB_ENTITY="${WANDB_ENTITY:-rceccatelli}"

python "${PROJECT_ROOT}/train_model.py" \
--run_name ${RUN_NAME} \
--modality "MUS_SYMB" \
--model_name "baseline_llama_transformer" \
--model_size ${MODEL_SIZE} \
\
--tokenizer_type "REMI" \
--dataset_name "giga-midi" \
\
--context_length 1024 \
\
--gpus "-1" \
\
--batch_size_per_device 4 \
--num_workers 8 \
\
--wandb_project "music_inference" \
\
--log_model_archi \
\
--execution_mode "inference" \
--only_test \
--only_test_model_ckpt "logs/checkpoints/baseline-llama-symb-small-prod0.0006_2026-04-22_18-17-39_/last.ckpt" \
--infer_max_gen_len 256 \
--infer_topp 0.9 \
--infer_temp 1.0 \
--infer_logprobs True \
--infer_echo True \
\
--override_slurm_checks \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}
