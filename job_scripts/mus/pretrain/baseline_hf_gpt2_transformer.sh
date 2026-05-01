#!/bin/bash
### Baseline HF GPT2 Transformer Music - Pretraining Script
### Hugging Face GPT2 baseline using transformers library for comparison
### This is a separate baseline to validate the benefits of EBT

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=80GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --array=0

### LOG INFO ###
export RUN_NAME="baseline-hf-gpt2-symb-small"
export MODEL_SIZE="small"

# Learning rate
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

# Parse command-line arguments
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
--model_name "baseline_hf_gpt2_transformer" \
--model_size "${MODEL_SIZE}" \
--tokenizer_type "${TOKENIZER_TYPE}" \
--normalize_initial_condition \
--context_length 1024 \
--gpus "1" \
--peak_learning_rate "${lr[${SLURM_ARRAY_TASK_ID}]}" \
--batch_size_per_device 32 \
--accumulate_grad_batches 8 \
--gradient_clip_val 1.0 \
--weight_decay 0.05 \
--min_lr_scale 10 \
--max_steps 60000 \
--max_scheduling_steps 60000 \
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
# - Baseline HF GPT2 transformer for comparison with EBT
# - Same training configuration as baseline_llama_transformer.sh for fair comparison
# - 60k steps, 1024 context length
# - Standard AdamW optimizer with cosine annealing
# - bf16 mixed precision for efficiency
