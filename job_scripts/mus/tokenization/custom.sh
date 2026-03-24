#!/bin/bash
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=custom_tok_%j.out

# --- 1. Capture Arguments ---
DATASET_NAME=$1
TOKENIZER_TYPE=$2

if [ -z "$DATASET_NAME" ] || [ -z "$TOKENIZER_TYPE" ]; then
    echo "❌ Error: Missing required arguments."
    echo "Usage: sbatch $0 [dataset-name] [tokenizer-type]"
    echo "Example: sbatch $0 jordan-progrock-dataset miditok"
    exit 1
fi

# --- 2. DYNAMIC NAMING (Self-Re-Submission) ---
# Construct a clean name like: tok_jordan-progrock_miditok
JOB_NAME="tok_${DATASET_NAME}_${TOKENIZER_TYPE}"

if [[ "$SLURM_JOB_NAME" == "custom_tok.sh" || -z "$SLURM_JOB_NAME" ]]; then
    echo "🔄 Re-submitting with personalized name: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" --output="${JOB_NAME}_%j.out" "$0" "$DATASET_NAME" "$TOKENIZER_TYPE"
    exit 0
fi

# --- 3. LOAD ENVIRONMENT VARIABLES ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ENV_PATH="../../../.env"

if [ -f "$ENV_PATH" ]; then
    export $(grep -v '^#' "$ENV_PATH" | xargs)
    echo "✅ Configuration loaded from .env"
else
    echo "⚠️ Warning: .env not found. Using shell environment or defaults."
fi

SYMBOLIC_ROOT="$PROJECT_ROOT/data/mus/symbolic"

cd "$PROJECT_ROOT" || { echo "❌ Could not enter $PROJECT_ROOT"; exit 1; }

# --- 5. DYNAMIC STORAGE LOGIC (Your Logic Intact) ---
if [ -n "$CUSTOM_STORAGE_PATH" ]; then
    export REMOTE_DATA_STORAGE="$CUSTOM_STORAGE_PATH"
elif [ -n "$REMOTE_DATA_STORAGE" ]; then
    export REMOTE_DATA_STORAGE
else
    unset REMOTE_DATA_STORAGE
fi

# --- 6. Environment Setup ---
export PYTHONPATH="$SYMBOLIC_ROOT:$SYMBOLIC_ROOT/tokenization/anticipation:$PYTHONPATH"
MODULE_NAME="tokenization.${TOKENIZER_TYPE}_tokenizer"
PYTHON_EXEC=$(which python3)

echo "--- Tokenization Metadata ---"
echo "Job Name:       $SLURM_JOB_NAME"
echo "Project Root:   $PROJECT_ROOT"
echo "Data Storage:   ${REMOTE_DATA_STORAGE:-Local Project Directory}"
echo "Dataset:        $DATASET_NAME"
echo "Module:         $MODULE_NAME"
echo "-----------------------------"

# --- 7. Execution ---
if "$PYTHON_EXEC" -m "$MODULE_NAME" CUSTOM "$DATASET_NAME"; then
    echo "✅ Tokenization Complete for $DATASET_NAME."
else
    echo "❌ Error: Tokenization failed for $DATASET_NAME."
    exit 1
fi