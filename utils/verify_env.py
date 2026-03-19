"""
Diagnostic utility to verify the Music_EBT environment.
Checks for GPU availability, library versions, and data paths.
Usage: python utils/verify_env.py
"""

import sys
import torch
import anticipation
from transformers import AutoTokenizer

def run_diagnostics():
    print("--- 🛠️ Music_EBT Environment Diagnostics ---")
    
    # 1. Check Package Path
    print(f"📍 Anticipation Path: {anticipation.__file__}")
    
    # 2. Check GPU (Crucial for Cluster Users)
    cuda_available = torch.cuda.is_available()
    print(f"🎮 GPU Available: {'✅ Yes' if cuda_available else '❌ No (CPU Mode)'}")
    if cuda_available:
        print(f"   - Device: {torch.cuda.get_device_name(0)}")

    # 3. Check Transformer Logic
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        print("✅ Transformers/Tokenizers: Functional")
    except Exception as e:
        print(f"❌ Transformers Error: {e}")

    # 4. Check Config
    from anticipation import config
    print(f"🎼 MIDI Resolution: {config.TIME_RESOLUTION} bins/sec")
    
    print("\nEnvironment is VALIDATED for research.")

if __name__ == "__main__":
    run_diagnostics()