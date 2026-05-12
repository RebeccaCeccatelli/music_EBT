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
export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

# Load environment variables from .env file
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Get wandb entity from environment or use default
WANDB_ENTITY="${WANDB_ENTITY:-rceccatelli}"

"${HOME}/.conda/envs/music_EBT/bin/python" "${PROJECT_ROOT}/inference/mus/infer_baselines_interactive.py" \
--checkpoint "/home/rebcecca/music-EBT/logs/checkpoints/baseline-llama-small-remi-job13735284_2026-05-11_11-29-25_/epoch=epoch=15-step=step=8540-valid_loss=valid_loss=0.6461.ckpt" \
--model_name baseline_llama_transformer \
--prompt_length 128 \
--generation_length 256 \
--num_samples 10 \
--output_dir "${PROJECT_ROOT}/logs/inference_outputs/llama_remi_0.646" \
--device cuda
