#!/bin/bash
set -e

echo "🚀 Syncing Music_EBT Submodules..."
# Download the anticipation-musicEBT submodule
git submodule update --init --recursive

# Safety check for the submodule
if [ ! -f "data/mus/symbolic/tokenization/anticipation/setup.py" ]; then
    echo "⚠️ Submodule folder is empty. Trying a force update..."
    git submodule update --init --recursive --force
fi

echo "🐍 Installing Python Dependencies..."
pip install -r requirements.txt

echo "📂 Configuring Package Structure..."
# Ensure Python sees the tokenization folder as a package
touch data/mus/symbolic/tokenization/__init__.py

echo "✅ Environment Refreshed."

# Force the path for this specific execution, to be able to run verify_env.py
export PYTHONPATH=$PYTHONPATH:$(pwd)/data/mus/symbolic

# Run the diagnostic
python3 utils/verify_env.py