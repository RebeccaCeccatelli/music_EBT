#!/bin/bash
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00              # Increased to allow for LAKH scale + Download
#SBATCH --output=lakh_full_pipe_%j.out

# ==============================================================================
# USER CONFIGURATION
# ------------------------------------------------------------------------------
CUSTOM_STORAGE_PATH="/orcd/home/002/rebcecca/orcd/pool/music_datasets"
# ==============================================================================

# --- 1. Capture Arguments ---
# $1: Tokenizer Type (Defaults to 'anticipation')
TOKENIZER_TYPE=${1:-anticipation} 

# --- 2. Resolve Paths Agnostically ---
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
# CRITICAL: We add both the symbolic root AND the internal anticipation source folder
export PYTHONPATH="$ROOT_DIR/data/mus/symbolic:$ROOT_DIR/data/mus/symbolic/tokenization/anticipation"

# Construct the module name based on choice
MODULE_NAME="tokenization.${TOKENIZER_TYPE}_tokenizer"

# --- 5. Execution ---
PYTHON_EXEC=$(which python3)

echo "--- Lakh Full Pipeline Metadata ---"
echo "User:           $USER"
echo "Project Root:   $ROOT_DIR"
echo "Data Storage:   ${REMOTE_DATA_STORAGE:-Local Project Directory}"
echo "Tokenizer Mode: $TOKENIZER_TYPE"
echo "-----------------------------------"

# STEP 1: Download/Verify Data (Always runs first)
echo "🚀 [Step 1/2] Checking/Downloading Lakh Dataset..."
"$PYTHON_EXEC" -m dataloaders.lakh_dataloader

# STEP 2: Preprocess and Tokenize
echo "🎹 [Step 2/2] Starting Tokenization ($MODULE_NAME)..."
"$PYTHON_EXEC" -m "$MODULE_NAME" LAKH

echo "✅ Full Pipeline Complete."