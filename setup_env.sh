#!/bin/bash
# 'set -e' ensures the script stops immediately if any command fails.
set -e

echo "--------------------------------------------------------"
echo "🛠️  Music_EBT: Initializing Research Environment"
echo "--------------------------------------------------------"

# --- ⚙️ GLOBAL CONFIGURATION ---

# 1. STORAGE CONFIG: Redirects heavy datasets to high-capacity storage (Pool).
# 💡 USER TIP: Set to "" to use the local 'datasets' folder within the repo.
export REMOTE_DATA_STORAGE="/orcd/home/002/rebcecca/orcd/pool/music_datasets"

if [ -n "$REMOTE_DATA_STORAGE" ]; then
    echo "📂 Storage: [REMOTE] Creating directory at $REMOTE_DATA_STORAGE..."
    mkdir -p "$REMOTE_DATA_STORAGE"
    echo "✅ Storage: Linked to Pool storage."
else
    echo "📂 Storage: [LOCAL] No remote path set. Using repository defaults."
fi

# 2. PYTHONPATH: Tells Python to look inside our symbolic music directory for modules.
# We use $(pwd) to ensure absolute paths regardless of where the script is called from.
export PYTHONPATH=$PYTHONPATH:$(pwd)/data/mus/symbolic
echo "🐍 Python: PYTHONPATH updated to include $(pwd)/data/mus/symbolic"

echo -e "\n📦 STEP 1: Syncing Submodules..."
# Fetch the 'anticipation' logic which lives in a separate repository.
git submodule update --init --recursive

# Submodule Integrity Check
if [ ! -f "data/mus/symbolic/tokenization/anticipation/setup.py" ]; then
    echo "⚠️  Alert: Submodule folder appears empty or missing files."
    echo "🔄  Attempting force-sync of submodules..."
    git submodule update --init --recursive --force
else
    echo "✅ Submodules: Successfully synced and verified."
fi

echo -e "\n📚 STEP 2: Installing Dependencies..."
# Install libraries from requirements.txt (miditok, transformers, etc.)
if [ -f "requirements.txt" ]; then
    echo "📥 Running pip install -r requirements.txt..."
    pip install -q -r requirements.txt
    echo "✅ Dependencies: Installation complete."
else
    echo "⚠️  Warning: requirements.txt not found. Skipping pip install."
fi

echo -e "\n🏗️  STEP 3: Configuring Package Structure..."
# Creates __init__.py files to turn directories into importable Python packages.
# This prevents 'ModuleNotFoundError' when running scripts from different folders.
INIT_FILE="data/mus/symbolic/tokenization/__init__.py"
if [ ! -f "$INIT_FILE" ]; then
    touch "$INIT_FILE"
    echo "🪄  Created: $INIT_FILE"
fi
echo "✅ Structure: Package tree is valid."

echo -e "\n--------------------------------------------------------"
echo "✅ Environment Ready for Research."
echo "📍 Data Storage: ${REMOTE_DATA_STORAGE:-'Local Repo'}"
echo "--------------------------------------------------------"

# --- 🧪 DIAGNOSTIC ---
echo "🔍 Running final environment verification..."
python3 utils/verify_env.py