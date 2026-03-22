#!/bin/bash
#SBATCH --job-name=giga_full_pipe
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00             
#SBATCH --output=giga_full_pipe_%j.out

# ==============================================================================
# USER CONFIGURATION
# ------------------------------------------------------------------------------
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    
    # Check if the variable is now set
    if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
        echo "❌ Error: .env file found, but HUGGING_FACE_HUB_TOKEN is empty."
        exit 1
    else
        # Print only the first 4 characters to confirm it loaded the right thing safely
        echo "✅ Token loaded successfully (Starts with: ${HUGGING_FACE_HUB_TOKEN:0:4}...)"
    fi
else
    echo "❌ Error: .env file not found. Please create it with your token."
    exit 1
fi

# Path where your datasets are stored
CUSTOM_STORAGE_PATH="/orcd/home/002/rebcecca/orcd/pool/music_datasets"
# ==============================================================================

# --- 1. Capture Arguments ---
TOKENIZER_TYPE=${1:-anticipation} 

# --- 2. Environment & Path Setup ---
# We need to point PYTHONPATH to the directory that CONTAINS the 'dataloaders' folder
# Based on your path: /orcd/home/002/rebcecca/music_EBT/data/mus/symbolic
export PROJECT_ROOT="/orcd/home/002/rebcecca/music_EBT/data/mus/symbolic"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Move to the project root so Python can resolve 'dataloaders' and 'tokenization' modules
cd "$PROJECT_ROOT"

# --- 3. Dynamic Storage Logic ---
if [ -n "$CUSTOM_STORAGE_PATH" ]; then
    export REMOTE_DATA_STORAGE="$CUSTOM_STORAGE_PATH"
fi

# Construct the module name
MODULE_NAME="tokenization.${TOKENIZER_TYPE}_tokenizer"

# --- 4. Execution ---
PYTHON_EXEC=$(which python3)

echo "--- GigaMIDI Full Pipeline Metadata ---"
echo "User:           $USER"
echo "Project Root:   $PROJECT_ROOT"
echo "Data Storage:   ${REMOTE_DATA_STORAGE}"
echo "Tokenizer Mode: $TOKENIZER_TYPE"
echo "---------------------------------------"

# STEP 1: Download/Extract GigaMIDI
echo "🚀 [Step 1/2] Downloading & Unzipping GigaMIDI..."
# Running as a module requires the parent dir to be in PYTHONPATH
"$PYTHON_EXEC" -m dataloaders.giga_midi_dataloader

# STEP 2: Tokenize
echo "🎹 [Step 2/2] Starting Tokenization ($MODULE_NAME)..."
"$PYTHON_EXEC" -m "$MODULE_NAME" giga-midi

echo "✅ GigaMIDI Pipeline Complete."