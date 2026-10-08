#!/bin/bash
### Generate unguided continuations from one model for eval/music_quality.py,
### with settings identical across EBT and the baselines: same seed → same
### validation songs, prompt = song start, same prompt/generation length, T=0.7,
### top-p 0.9. Then score generated and ground-truth continuations.
###
###   MODEL=ebt   TOK=REMI CHECKPOINT=<ckpt> sbatch job_scripts/mus/eval/music_quality_generate.sh
###   MODEL=llama TOK=Anticipation-Arrival-Time CHECKPOINT=<ckpt> sbatch ...
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=60GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
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
# demo/ holds convert_midi_simple, which infer_ebt.py imports for the WAV step.
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/demo:${HOME}/music-EBT/data/mus/symbolic:${PYTHONPATH:-}"
export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1

: "${MODEL:?MODEL=ebt|gpt2|llama}" "${TOK:?TOK=REMI|Anticipation-Arrival-Time}" "${CHECKPOINT:?CHECKPOINT=<path>}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
PROMPT_LENGTH="${PROMPT_LENGTH:-64}"
GENERATION_LENGTH="${GENERATION_LENGTH:-256}"
SEED="${SEED:-0}"

SLUG=$([[ "${TOK}" == REMI ]] && echo remi || echo ant-at-full)
OUT="${OUT:-${HOME}/orcd/scratch/rebcecca/music_EBT_logs/music_quality/gen_${MODEL}_${SLUG}}"
scontrol update JobId="${SLURM_JOB_ID}" JobName="mq-gen-${MODEL}-${SLUG}" 2>/dev/null || true
echo "Model: ${MODEL}  Tokenizer: ${TOK}  N=${NUM_SAMPLES}  prompt=${PROMPT_LENGTH}  gen=${GENERATION_LENGTH}  seed=${SEED}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT}"

if [[ "${MODEL}" == ebt ]]; then
    python inference/mus/infer_ebt.py --checkpoint "${CHECKPOINT}" \
        --num_samples "${NUM_SAMPLES}" --prompt_length "${PROMPT_LENGTH}" \
        --generation_length "${GENERATION_LENGTH}" --seed "${SEED}" \
        --output_dir "${OUT}"
    MIDI_DIR="${OUT}/midi"; GEN_GLOB='*_generated.mid'
else
    MODEL_NAME=$([[ "${MODEL}" == gpt2 ]] && echo baseline_hf_gpt2_transformer || echo baseline_llama_transformer)
    python inference/mus/infer_baselines_interactive.py --checkpoint "${CHECKPOINT}" \
        --model_name "${MODEL_NAME}" --num_samples "${NUM_SAMPLES}" \
        --prompt_length "${PROMPT_LENGTH}" --generation_length "${GENERATION_LENGTH}" \
        --seed "${SEED}" --output_dir "${OUT}/midi" --device cuda
    MIDI_DIR="${OUT}/midi"; GEN_GLOB='*_prompt_with_generated_continuation.mid'
fi

# Both scripts save prompt+continuation and the real prompt+continuation
# (ground truth) for the same songs; score both so each model has a matched
# real reference as well as the random-window one.
python eval/music_quality.py score --midi_dir "${MIDI_DIR}" --glob "${GEN_GLOB}" --out "${OUT}/scores_generated.csv"
python eval/music_quality.py score --midi_dir "${MIDI_DIR}" --glob '*_ground_truth.mid' --out "${OUT}/scores_ground_truth.csv"
echo "✅ done"
