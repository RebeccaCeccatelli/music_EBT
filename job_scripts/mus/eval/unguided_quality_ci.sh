#!/bin/bash
### Paired bootstrap CIs + permutation tests for the unguided music-quality
### comparison (EBT vs GPT-2 vs Llama). CPU only. See eval/unguided_quality_ci.py.
###
###   sbatch job_scripts/mus/eval/unguided_quality_ci.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --mem=8GB
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --job-name=unguided-quality-ci
#SBATCH --output=./logs/slurm_%j.out

set -euo pipefail
cd "${HOME}/music-EBT"
export PYTHONPATH="${HOME}/music-EBT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
PY="${HOME}/.conda/envs/music_EBT/bin/python"
R="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/music_quality"
OUT="docs/thesis_findings/figures/unguided_quality_ci"
mkdir -p "${OUT}"
# One process per tokenizer, in parallel (2 CPUs).
for t in remi ant-at-full-ar; do
    "${PY}" eval/unguided_quality_ci.py --root "${R}" --tok "${t}" --n_boot "${N_BOOT:-1000}" \
        --n_perm "${N_PERM:-1000}" --models "${MODELS:-ebt,gpt2,llama}" --out "${OUT}/${t}${SUFFIX:-}.json" > "logs/unguided_quality_ci_${SLURM_JOB_ID}_${t}.log" 2>&1 &
done
wait
cat logs/unguided_quality_ci_${SLURM_JOB_ID}_*.log | grep -v Warning
echo "✅ done"
