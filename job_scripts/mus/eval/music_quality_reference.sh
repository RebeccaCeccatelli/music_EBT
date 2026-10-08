#!/bin/bash
### Build the real-data reference set for eval/music_quality.py: decode N random
### GigaMIDI validation windows per tokenizer to MIDI, then score them.
###
###   sbatch job_scripts/mus/eval/music_quality_reference.sh
###   N=1000 N_TOKENS=512 sbatch job_scripts/mus/eval/music_quality_reference.sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --job-name=music-quality-ref
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

N="${N:-500}"
N_TOKENS="${N_TOKENS:-256}"
OUT_ROOT="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/music_quality"

# TOKS: which tokenizers to build (default both). AR_ONLY=1: Anticipation
# windows from AUTOREGRESS-mode sequences only (written to a separate *-ar dir).
for TOK in ${TOKS:-REMI Anticipation-Arrival-Time}; do
    SLUG=$([[ "${TOK}" == REMI ]] && echo remi || echo ant-at-full)
    EXTRA=()
    if [[ "${TOK}" != REMI && "${AR_ONLY:-0}" == 1 ]]; then
        SLUG="${SLUG}-ar"; EXTRA=(--autoregress_only)
    fi
    DIR="${OUT_ROOT}/reference_${SLUG}_${N_TOKENS}tok"
    echo "=== ${TOK} → ${DIR}"
    python eval/dump_reference_midi.py --tokenizer_type "${TOK}" --n "${N}" \
        --n_tokens "${N_TOKENS}" --out_dir "${DIR}/midi" "${EXTRA[@]}"
    python eval/music_quality.py score --midi_dir "${DIR}/midi" --out "${DIR}/scores.csv"
done
echo "✅ reference sets done"
