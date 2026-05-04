#!/bin/bash
### Baseline Llama Transformer Music - Pretraining Script
### Custom Llama2-inspired baseline transformer for comparison with EBT models

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=80GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=./logs/slurm_%j.out

### ADDITIONAL RUN INFO ###
#SBATCH --array=0

### LOG INFO ###
export RUN_NAME="baseline-llama-symb-small-prod"
export MODEL_SIZE="small"

# Recommended LR for 'small' per your model_utils.py is 0.0006
lr=(0.0006)

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

export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}" || exit 1

# Parse command-line arguments (overridable via sbatch)
DATASET_NAME="giga_midi"
TOKENIZER_TYPE="REMI"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --tokenizer_type) TOKENIZER_TYPE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

python train_model.py \
--run_name "${RUN_NAME}-${lr[${SLURM_ARRAY_TASK_ID}]}" \
--modality "MUS_SYMB" \
--model_name "baseline_llama_transformer" \
--model_size "${MODEL_SIZE}" \
--tokenizer_type "${TOKENIZER_TYPE}" \
--normalize_initial_condition \
--context_length 512 \
--gpus "1" \
--peak_learning_rate 0.0008 \
--batch_size_per_device 64 \
--accumulate_grad_batches 4 \
--gradient_clip_val 1.0 \
--weight_decay 0.05 \
--min_lr_scale 10 \
--max_steps 25000 \
--max_scheduling_steps 25000 \
--warm_up_steps 2500 \
--dataset_name "${DATASET_NAME}" \
--tokenizer_config_path "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json" \
--num_workers 12 \
--validation_split_pct 0.05 \
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
