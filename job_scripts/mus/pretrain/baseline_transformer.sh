#!/bin/bash
### Baseline Transformer Music - Pretraining Script
### Standard transformer baseline for comparison with EBT models
### This is a quick 2-hour test run to validate training pipeline

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00  # Extended for 50k steps
#SBATCH --mem=80GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=/dev/null

### ADDITIONAL RUN INFO ###
#SBATCH --array=0

### LOG INFO ###
#SBATCH --job-name=baseline-symb-xxs-prod
# SLURM logs sent to wandb (see --wandb_project below)
export RUN_NAME="baseline-symb-xxs-prod"
export MODEL_SIZE="xxs"

### Get the project root directory
### Try multiple methods since $0 can be unreliable in SLURM
### Method 1: Search up directory tree for train_model.py (most reliable in SLURM)
find_project_root() {
    local dir="$1"
    for ((i=0; i<10; i++)); do
        if [[ -f "${dir}/train_model.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    echo "" # Not found
    return 1
}

PROJECT_ROOT="$(find_project_root "$(pwd)")"
if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "❌ Error: Could not find project root. train_model.py not found in parent directories."
    echo "   Make sure this script is in <project_root>/job_scripts/mus/pretrain/"
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}" || exit 1

# Parse command-line arguments
DATASET_NAME="giga_midi"
TOKENIZER_TYPE="REMI"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --tokenizer_type)
            TOKENIZER_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

lr=(0.0008)  # Slightly higher LR for better signal

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "MUS_SYMB" \
--model_name "baseline_transformer" \
--model_size ${MODEL_SIZE} \
--tokenizer_type "${TOKENIZER_TYPE}" \
--normalize_initial_condition \
--context_length 512 \
--gpus "1" \
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 32 \
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
--weight_decay 0.05 \
--min_lr_scale 10 \
--max_steps 50000 \
--max_scheduling_steps 50000 \
--warm_up_steps 5000 \
--dataset_name "${DATASET_NAME}" \
--num_workers 12 \
--validation_split_pct 0.1 \
--limit_train_batches 1.0 \
--limit_val_batches 1.0 \
--limit_test_batches 1.0 \
--wandb_project 'mus_symb_baseline_pretrain' \
--log_model_archi \
--log_gradients \
--log_every_n_steps 200 \
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}

# NOTES:
# - Production run with 50k steps (24 hour timeout)
# - Standard transformer baseline (no MCMC/energy-based training)
# - 10% validation split for better overfitting detection
# - Higher regularization (weight_decay 0.05) to reduce overfitting
# - 5k step warm-up (~10% of total steps)
# - Logging every 200 steps for cleaner graphs
#
# USAGE:
# Default (giga_midi + REMI):
#   sbatch baseline_transformer.sh
# 
# With custom dataset and tokenizer:
#   sbatch baseline_transformer.sh --dataset_name giga-midi --tokenizer_type anticipation
#   sbatch baseline_transformer.sh --dataset_name jordan-progrock-dataset --tokenizer_type anticipation
