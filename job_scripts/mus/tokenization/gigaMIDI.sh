#!/bin/bash
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00            
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# --- 1. Capture Arguments ---
TOKENIZER_TYPE=$1

if [ -z "$TOKENIZER_TYPE" ]; then
    echo "❌ Error: No tokenizer type provided."
    echo "Usage: sbatch $0 [tokenizer-type]"
    echo "Example: sbatch $0 anticipation_tokenizer"
    exit 1
fi

# --- 2. DYNAMIC NAMING ---
if [[ "$SLURM_JOB_NAME" == "gigaMIDI.sh" || -z "$SLURM_JOB_NAME" ]]; then
    # We EXPLICITLY set the --output here for the real job. 
    # The current 'manager' job will follow the header and send its output to /dev/null.
    sbatch --job-name="gigaMIDI_${TOKENIZER_TYPE}" \
           "$0" "$TOKENIZER_TYPE"
    exit 0
fi

# --- 3. LOAD ENVIRONMENT VARIABLES ---
# Locates .env relative to the script's directory (matches lakh.sh logic)
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ENV_PATH="../../../.env"

if [ -f "$ENV_PATH" ]; then
    export $(grep -v '^#' "$ENV_PATH" | xargs)
    echo "✅ Environment variables loaded from .env"
    
    if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
        echo "✅ HF Token loaded (Starts with: ${HUGGING_FACE_HUB_TOKEN:0:4}...)"
    fi

    # Ensure WandB can log in automatically on the compute node
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY="$WANDB_API_KEY"
        echo "✅ WandB API Key exported."
    fi
else
    echo "❌ Error: .env file not found at $ENV_PATH."
    exit 1
fi

# --- 4. DYNAMIC STORAGE LOGIC ---
# Priority: 1. .env CUSTOM_STORAGE_PATH, 2. Existing REMOTE_DATA_STORAGE
if [ -n "$CUSTOM_STORAGE_PATH" ]; then
    export REMOTE_DATA_STORAGE="$CUSTOM_STORAGE_PATH"
elif [ -n "$REMOTE_DATA_STORAGE" ]; then
    export REMOTE_DATA_STORAGE
fi

# --- 5. Resolve Paths & Environment ---
# Standardize the Root to match the location of your modules
SYMBOLIC_ROOT="$PROJECT_ROOT/data/mus/symbolic"

cd "$PROJECT_ROOT" || { echo "❌ Failed to enter $PROJECT_ROOT"; exit 1; }

# Add both the symbolic root and the anticipation sub-dir to PYTHONPATH
export PYTHONPATH="$SYMBOLIC_ROOT:$SYMBOLIC_ROOT/tokenization/anticipation:$PYTHONPATH"
export PYTHONUNBUFFERED=1 # Force real-time logging
export TQDM_ISATTY=1
export TQDM_MININTERVAL=0.1
export WANDB_CONSOLE=wrap_raw

LOADER_MODULE="dataloaders.giga_midi_dataloader"
TOKEN_MODULE="tokenization.${TOKENIZER_TYPE}_tokenizer"
PYTHON_EXEC=$(which python3)

echo "--- GigaMIDI Full Pipeline Metadata ---"
echo "Project Root:   $PROJECT_ROOT"
echo "Data Storage:   ${REMOTE_DATA_STORAGE:-Local Project Directory}"
echo "Module:         $TOKEN_MODULE"
echo "---------------------------------------"

# --- 6. Execution ---

# STEP 1: Download/Extract GigaMIDI
echo "🚀 [Step 1/2] Checking/Downloading GigaMIDI..."
"$PYTHON_EXEC" -m "$LOADER_MODULE"

# STEP 2: Tokenize
echo "🎹 [Step 2/2] Starting Tokenization ($TOKEN_MODULE)..."
if "$PYTHON_EXEC" -m "$TOKEN_MODULE" gigamidi; then
    echo "✅ GigaMIDI Pipeline Complete."
else
    echo "❌ Error: Tokenization failed."
    exit 1
fi