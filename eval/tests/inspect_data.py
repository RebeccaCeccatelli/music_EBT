#!/usr/bin/env python3
"""
Inspect actual data being used for training
"""

import os
import sys

# Check where the tokenized data actually is
base_data_path = '/home/rebcecca/orcd/pool/music_datasets'
dataset_name = 'giga-midi'
tokenizer_type = 'miditok'

tokens_dir = os.path.join(base_data_path, dataset_name, "tokens", tokenizer_type)
token_file = os.path.join(tokens_dir, f"tokenized-events-train.txt")

print(f"Looking for tokens at: {token_file}")
print(f"File exists: {os.path.exists(token_file)}")

if not os.path.exists(tokens_dir):
    print(f"\n❌ Directory doesn't exist: {tokens_dir}")
    print(f"\nTrying alternative path...")
    
    # Try checking environment variable
    alt_path = os.getenv('CUSTOM_STORAGE_PATH', '/home/rebcecca/orcd/pool/music_datasets')
    print(f"CUSTOM_STORAGE_PATH: {alt_path}")
    
    # List what's actually there
    for root, dirs, files in os.walk('/home/rebcecca/orcd/pool/music_datasets'):
        level = root.replace('/home/rebcecca/orcd/pool/music_datasets', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}{file}')
        if level > 2:  # Limit depth
            break
else:
    print(f"\n✅ Token file found!")
    
    # Read first few lines
    print(f"\nFirst 10 sequences from training set:\n")
    with open(token_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            tokens = list(map(int, line.split()))
            print(f"Seq {i}: {len(tokens)} tokens, range [{min(tokens)}, {max(tokens)}], first 20: {tokens[:20]}")
    
    # Check statistics
    print(f"\nDataset statistics:")
    total_seqs = 0
    total_tokens = 0
    min_len = float('inf')
    max_len = 0
    vocab_min = float('inf')
    vocab_max = 0
    
    with open(token_file, 'r') as f:
        for line in f:
            tokens = list(map(int, line.split()))
            total_seqs += 1
            total_tokens += len(tokens)
            min_len = min(min_len, len(tokens))
            max_len = max(max_len, len(tokens))
            vocab_min = min(vocab_min, min(tokens))
            vocab_max = max(vocab_max, max(tokens))
            
            if total_seqs >= 1000:  # Sample first 1000
                break
    
    print(f"  Sequences sampled: {total_seqs}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg tokens/seq: {total_tokens/total_seqs:.1f}")
    print(f"  Min seq length: {min_len}")
    print(f"  Max seq length: {max_len}")
    print(f"  Token range: [{vocab_min}, {vocab_max}]")
    print(f"  Vocab size should be: 284")
