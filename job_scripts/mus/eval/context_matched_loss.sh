#!/bin/bash
### Validation loss / perplexity of the final EBT and baseline checkpoints on
### identical validation windows at context 512 (+ baselines at 1024).
### See eval/context_matched_loss.py.
###
###   sbatch job_scripts/mus/eval/context_matched_loss.sh
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=60GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --job-name=context-matched-loss
#SBATCH --output=./logs/slurm_%j.out

set -euo pipefail

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
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${HOME}/music-EBT/data/mus/symbolic:${PYTHONPATH:-}"
export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1

N_WINDOWS="${N_WINDOWS:-1000}"
B="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/checkpoints"
OUT="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/music_quality/context_matched_loss"
mkdir -p "${OUT}"

# Final checkpoints, as listed in docs/STATUS.md.
one() { ls $1 | head -1; }
python eval/context_matched_loss.py --tokenizer_type REMI --n_windows "${N_WINDOWS}" \
    --model "EBT=$(one "${B}/ebt-symb-small-remi-s1-*/*step=33732-*.ckpt")" \
    --model "GPT-2=$(one "${B}/baseline-hf-gpt2-small-remi-*/*step=99660-valid_loss=valid_loss=0.4756.ckpt")" \
    --model "Llama=$(one "${B}/baseline-llama-small-remi-*/*step=99660-valid_loss=valid_loss=0.5239.ckpt")" \
    --out "${OUT}/remi.json"

python eval/context_matched_loss.py --tokenizer_type Anticipation-Arrival-Time --n_windows "${N_WINDOWS}" \
    --model "EBT=$(one "${B}/ebt-symb-small-ant-at-full-s1-job*/*step=88800-*.ckpt")" \
    --model "GPT-2=$(one "${B}/baseline-hf-gpt2-small-ant-at-full-*/*step=98900-*.ckpt")" \
    --model "Llama=$(one "${B}/baseline-llama-small-ant-at-full-*/*step=100000-*.ckpt")" \
    --out "${OUT}/ant-at-full.json"
echo "✅ done"
