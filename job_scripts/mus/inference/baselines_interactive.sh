#!/bin/bash
### Interactive Inference for Symbolic Music Baselines
### Generates MIDI continuations and synthesizes to audio for listening

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=logs/slurm/mus/baselines_interactive_%j.log

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

# Load environment variables
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Configuration - EDIT THESE TO CUSTOMIZE
CHECKPOINT_GPT2="${CHECKPOINT_GPT2:-logs/checkpoints/baseline-hf-gpt2-symb-small-0.0006_2026-05-01_22-13-37_/epoch=epoch=11-step=step=6432-valid_loss=valid_loss=0.7746.ckpt}"
CHECKPOINT_LLAMA="${CHECKPOINT_LLAMA:-logs/checkpoints/baseline-llama-symb-small-prod-0.0006_2026-05-01_22-13-36_/epoch=epoch=12-step=step=6968-valid_loss=valid_loss=0.7793.ckpt}"
PROMPT_LENGTH="${PROMPT_LENGTH:-128}"
GENERATION_LENGTH="${GENERATION_LENGTH:-384}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
USE_TEST_SPLIT="${USE_TEST_SPLIT:-}" # Set to "--use_test_split" to use test split

# Generate timestamp for unique run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="./inference_outputs/run_${TIMESTAMP}"

# Create run directory with organized structure
mkdir -p "${RUN_DIR}/gpt2/midi" "${RUN_DIR}/gpt2/wav" "${RUN_DIR}/gpt2/tokens"
mkdir -p "${RUN_DIR}/llama/midi" "${RUN_DIR}/llama/wav" "${RUN_DIR}/llama/tokens"

echo "=========================================="
echo "Symbolic Music Baselines - Interactive Inference"
echo "=========================================="
echo "Run: $RUN_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Prompt length: ${PROMPT_LENGTH} tokens"
echo "Generation length: ${GENERATION_LENGTH} tokens"
echo "Number of samples: ${NUM_SAMPLES}"
echo "=========================================="

# Create inference log directory
mkdir -p inference_logs

# Generate log file
INFERENCE_LOG="inference_logs/slurm_inference_${SLURM_JOB_ID}_${TIMESTAMP}.log"

{
    echo "=========================================="
    echo "GPT2 Baseline Inference (5 samples)"
    echo "=========================================="
    /home/rebcecca/.conda/envs/music_EBT/bin/python "${PROJECT_ROOT}/inference/mus/infer_baselines_interactive.py" \
        --checkpoint "${CHECKPOINT_GPT2}" \
        --model_name baseline_hf_gpt2_transformer \
        --prompt_length "${PROMPT_LENGTH}" \
        --generation_length "${GENERATION_LENGTH}" \
        --num_samples "${NUM_SAMPLES}" \
        --output_dir "${RUN_DIR}/gpt2/midi" \
        ${USE_TEST_SPLIT} \
        --device cuda

    echo ""
    echo "=========================================="
    echo "Llama Baseline Inference (5 samples)"
    echo "=========================================="
    /home/rebcecca/.conda/envs/music_EBT/bin/python "${PROJECT_ROOT}/inference/mus/infer_baselines_interactive.py" \
        --checkpoint "${CHECKPOINT_LLAMA}" \
        --model_name baseline_llama_transformer \
        --prompt_length "${PROMPT_LENGTH}" \
        --generation_length "${GENERATION_LENGTH}" \
        --num_samples "${NUM_SAMPLES}" \
        --output_dir "${RUN_DIR}/llama/midi" \
        ${USE_TEST_SPLIT} \
        --device cuda

    echo ""
    echo "=========================================="
    echo "✅ Inference Complete!"
    echo "=========================================="
    echo "Converting MIDI to WAV..."
    /home/rebcecca/.conda/envs/music_EBT/bin/python << 'PYTHON_EOF'
import sys
sys.path.insert(0, "${PROJECT_ROOT}")
from convert_midi_simple import simple_synth
from pathlib import Path
import glob

run_dir = Path("${RUN_DIR}")
for model_dir in ["gpt2", "llama"]:
    midi_dir = run_dir / model_dir / "midi"
    wav_dir = run_dir / model_dir / "wav"

    midi_files = sorted(midi_dir.glob("*.mid"))
    for midi_file in midi_files:
        wav_file = wav_dir / midi_file.name.replace(".mid", ".wav")
        try:
            simple_synth(str(midi_file), str(wav_file))
            print(f"✅ {wav_file.name}")
        except Exception as e:
            print(f"❌ {wav_file.name}: {e}")
PYTHON_EOF

    echo ""
    echo "=========================================="
    echo "✅ All Complete!"
    echo "=========================================="
    echo "Outputs saved to:"
    echo "  Run: $RUN_DIR"
    echo "  GPT2:  ${RUN_DIR}/gpt2/"
    echo "  Llama: ${RUN_DIR}/llama/"
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Log file: $INFERENCE_LOG"

} 2>&1 | tee "$INFERENCE_LOG"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Inference completed successfully"
else
    echo "❌ Inference failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
