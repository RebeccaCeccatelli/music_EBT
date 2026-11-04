#!/bin/bash
### Validation loss of both EBTs after each of 1..MAX_STEPS MCMC refinement
### steps (trained with 2; extra steps repeat the final landscape).
### See eval/mcmc_steps_loss.py.
###
###   sbatch job_scripts/mus/eval/mcmc_steps_loss.sh
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1  # L40S only: this torch build (2.4+cu121) has no kernels for the RTX Pro 6000 nodes
#SBATCH --exclude=node3006  # bad GPU: "CUDA unknown error" at init (2026-10-12); in the header so auto-resubmits keep it
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rebeccaceccatelli@gmail.com
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=60GB
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --account=mit_general
#SBATCH --job-name=mcmc-steps-loss
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
MAX_STEPS="${MAX_STEPS:-16}"
N_WINDOWS="${N_WINDOWS:-1000}"
B="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/checkpoints"
OUT="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/music_quality/mcmc_steps"
mkdir -p "${OUT}"
one() { ls $1 | head -1; }
# SET=stab: the regularised Anticipation lineage (Langevin noise, replay buffer, random step size;
# still 2 steps) at its best val, vs the original lineage at its nearest kept step.
if [[ "${SET:-main}" == stab ]]; then
    python eval/mcmc_steps_loss.py --max_steps "${MAX_STEPS}" --n_windows "${N_WINDOWS}" --batch_size 2 \
        --model "EBT-Ant-stab-42900=$(one "${B}/ebt-symb-small-ant-at-full-s1-stab+-job*/*step=42900-*.ckpt")" \
        --model "EBT-Ant-55500=$(one "${B}/ebt-symb-small-ant-at-full-s1-job*/*step=55500-*.ckpt")" \
        --out "${OUT}/ant-at-full-stab.json"
    echo "✅ done"; exit 0
fi

python eval/mcmc_steps_loss.py --max_steps "${MAX_STEPS}" --n_windows "${N_WINDOWS}" --batch_size 8 \
    --model "EBT-REMI=$(one "${B}/ebt-symb-small-remi-s1-*/*step=33732-*.ckpt")" \
    --out "${OUT}/remi.json"
# Anticipation's 55k vocab: keep the batch small (all steps' logits are held).
python eval/mcmc_steps_loss.py --max_steps "${MAX_STEPS}" --n_windows "${N_WINDOWS}" --batch_size 2 \
    --model "EBT-Ant=$(one "${B}/ebt-symb-small-ant-at-full-s1-job*/*step=88800-*.ckpt")" \
    --out "${OUT}/ant-at-full.json"
echo "✅ done"
