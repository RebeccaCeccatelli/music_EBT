#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting Music_EBT Environment Setup..."

# 1. Ensure the target directory exists
TARGET_DIR="data/mus/symbolic/tokenization"
mkdir -p "$TARGET_DIR"

# 2. Install requirements and clone anticipation into the specific folder
# Using --src forces the editable git repo into our target directory
pip install --src "$TARGET_DIR" -r requirements.txt

echo "✅ Installation Complete!"
echo "📍 Anticipation is now at: $TARGET_DIR/anticipation"

# 3. Run the diagnostic to confirm everything is linked
python utils/verify_env.py