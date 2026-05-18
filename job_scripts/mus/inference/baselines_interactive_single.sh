#!/bin/bash
### Single-model interactive inference for symbolic music baselines
### Submitted by baselines_interactive.sh; do not call directly.
###
### Required env vars (set by launcher via --export):
###   MODEL_NAME   - e.g. baseline_hf_gpt2_transformer
###   MODEL_SHORT  - e.g. gpt2
###   CHECKPOINT   - path to .ckpt file

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=logs/slurm/mus/baselines_interactive_%x_%j.log

mkdir -p logs/slurm/mus/

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

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Validate required vars
if [[ -z "${MODEL_NAME}" || -z "${MODEL_SHORT}" || -z "${CHECKPOINT}" ]]; then
    echo "❌ MODEL_NAME, MODEL_SHORT, and CHECKPOINT must be set."
    exit 1
fi

# Shared config (overridable from environment)
PROMPT_LENGTH="${PROMPT_LENGTH:-128}"
GENERATION_LENGTH="${GENERATION_LENGTH:-384}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
USE_TEST_SPLIT="${USE_TEST_SPLIT:-}"
WANDB_PROJECT="${WANDB_PROJECT:-music_inference_baselines}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="/home/rebcecca/orcd/pool/music_outputs/inference/baselines/run_${TIMESTAMP}_${MODEL_SHORT}"
mkdir -p "${RUN_DIR}/${MODEL_SHORT}/midi" "${RUN_DIR}/${MODEL_SHORT}/wav" "${RUN_DIR}/${MODEL_SHORT}/tokens"

SLURM_LOG="logs/slurm/mus/baselines_interactive_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"

echo "=========================================="
echo "Symbolic Music Baselines - Inference"
echo "=========================================="
echo "Run:    ${RUN_DIR}"
echo "Model:  ${MODEL_SHORT} (${MODEL_NAME})"
echo "Ckpt:   ${CHECKPOINT}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Prompt: ${PROMPT_LENGTH} tokens → Generate: ${GENERATION_LENGTH} tokens × ${NUM_SAMPLES} samples"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

{
    /home/rebcecca/.conda/envs/music_EBT/bin/python "${PROJECT_ROOT}/inference/mus/infer_baselines_interactive.py" \
        --checkpoint "${CHECKPOINT}" \
        --model_name "${MODEL_NAME}" \
        --prompt_length "${PROMPT_LENGTH}" \
        --generation_length "${GENERATION_LENGTH}" \
        --num_samples "${NUM_SAMPLES}" \
        --output_dir "${RUN_DIR}/${MODEL_SHORT}/midi" \
        ${USE_TEST_SPLIT} \
        --device cuda

    echo ""
    echo "Synthesizing WAV files..."
    /home/rebcecca/.conda/envs/music_EBT/bin/python << PYTHON_EOF
import sys
sys.path.insert(0, "${PROJECT_ROOT}")
from convert_midi_simple import simple_synth
from pathlib import Path

midi_dir = Path("${RUN_DIR}/${MODEL_SHORT}/midi")
wav_dir  = Path("${RUN_DIR}/${MODEL_SHORT}/wav")
wav_dir.mkdir(parents=True, exist_ok=True)

for midi_file in sorted(midi_dir.glob("*.mid")):
    wav_file = wav_dir / midi_file.with_suffix(".wav").name
    if wav_file.exists():
        print(f"↪ {wav_file.name} already exists")
        continue
    try:
        simple_synth(str(midi_file), str(wav_file))
        print(f"✅ {wav_file.name}")
    except Exception as e:
        print(f"❌ {wav_file.name}: {e}")
PYTHON_EOF

    echo ""
    echo "Generating piano rolls..."
    /home/rebcecca/.conda/envs/music_EBT/bin/python "${PROJECT_ROOT}/inference/mus/tokens_to_piano_roll.py" "${RUN_DIR}"

    echo ""
    echo "=========================================="
    echo "✅ Inference Complete — ${MODEL_SHORT}"
    echo "=========================================="
    echo "Outputs: ${RUN_DIR}/${MODEL_SHORT}/"
}

EXIT_CODE=$?

echo ""
echo "Uploading to wandb..."
/home/rebcecca/.conda/envs/music_EBT/bin/python "${PROJECT_ROOT}/inference/mus/upload_to_wandb.py" \
    "${RUN_DIR}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --log_file "${SLURM_LOG}"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Job complete"
else
    echo "❌ Job failed with exit code ${EXIT_CODE}"
fi

exit $EXIT_CODE
