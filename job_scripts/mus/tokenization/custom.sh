#!/bin/bash
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=custom_tok_%j.out

# ==============================================================================
# USER CONFIGURATION
# ------------------------------------------------------------------------------
CUSTOM_STORAGE_PATH="/orcd/home/002/rebcecca/orcd/pool/music_datasets"
# ==============================================================================

# --- 1. Capture Arguments ---
# $1: Dataset Name (e.g., jordan-progrock-dataset)
# $2: Tokenizer Type (defaults to 'anticipation')
DATASET_NAME=$1
TOKENIZER_TYPE=${2:-anticipation} 

# Validation: Stop if no dataset name is provided
if [ -z "$DATASET_NAME" ]; then
    echo "❌ Error: No dataset name provided."
    echo "Usage: sbatch $0 [dataset-name] [tokenizer-type (optional)]"
    echo "Example: sbatch $0 jordan-progrock-dataset miditok"
    exit 1
fi

# --- 2. Resolve Paths ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(realpath "$SCRIPT_DIR/../../..")
cd "$ROOT_DIR"

# --- 3. Dynamic Storage Logic ---
if [ -n "$CUSTOM_STORAGE_PATH" ]; then
    export REMOTE_DATA_STORAGE="$CUSTOM_STORAGE_PATH"
elif [ -n "$REMOTE_DATA_STORAGE" ]; then
    export REMOTE_DATA_STORAGE
else
    unset REMOTE_DATA_STORAGE
fi

# --- 4. Environment Setup ---
# Since both tokenizers are in 'tokenization/', we just need the root in PYTHONPATH
export PYTHONPATH="$ROOT_DIR/data/mus/symbolic:$ROOT_DIR/data/mus/symbolic/tokenization/anticipation"

# Determine the module name based on the type
# This maps 'anticipation' -> 'tokenization.anticipation_tokenizer'
# and 'miditok' -> 'tokenization.miditok_tokenizer'
MODULE_NAME="tokenization.${TOKENIZER_TYPE}_tokenizer"

# --- 5. Execution ---
PYTHON_EXEC=$(which python3)

echo "--- Tokenization Metadata ---"
echo "User:           $USER"
echo "Project Root:   $ROOT_DIR"
echo "Data Storage:   ${REMOTE_DATA_STORAGE:-Local Project Directory}"
echo "Dataset:        $DATASET_NAME"
echo "Module:         $MODULE_NAME"
echo "-----------------------------"

# Run the selected tokenizer
"$PYTHON_EXEC" -m "$MODULE_NAME" CUSTOM "$DATASET_NAME"