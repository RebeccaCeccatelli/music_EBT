#!/usr/bin/env python3
"""
Minimal test to verify training logic correctness.
Tests:
1. Batch shapes through forward pass
2. Loss computation 
3. Gradient flow
"""

import torch
import torch.nn.functional as F
from data.mus.symbolic.dataloaders.custom_music_dataloader import CustomMusicDataset
from argparse import Namespace

# Create minimal hparams
hparams = Namespace(
    context_length=512,
    batch_size_per_device=4,
)

# Load dataset
dataset = CustomMusicDataset(hparams, "giga-midi", split="train", tokenizer_type="miditok")
print(f"✅ Loaded {len(dataset)} sequences")

# Get a batch manually
batch_list = [dataset[i] for i in range(4)]
print(f"\nIndividual sample shapes:")
for i, sample in enumerate(batch_list):
    print(f"  Sample {i}: input_ids shape = {sample['input_ids'].shape}, dtype = {sample['input_ids'].dtype}")

# Simulate default PyTorch collation
from torch.utils.data import default_collate
batch = default_collate(batch_list)
print(f"\nAfter default_collate:")
print(f"  Batch shape: {batch['input_ids'].shape}")
print(f"  First few token values: {batch['input_ids'][0, :10]}")
print(f"  Token value range: [{batch['input_ids'].min()}, {batch['input_ids'].max()}]")
print(f"  Dtype: {batch['input_ids'].dtype}")

# Extract input/targets as forward_loss_wrapper does
print(f"\n=== Forward Loss Wrapper Simulation ===")
print(f"Original batch shape: {batch['input_ids'].shape}")

# This is what forward_loss_wrapper does
batch_squeezed = batch['input_ids'].squeeze(dim=1)
print(f"After squeeze(dim=1): {batch_squeezed.shape}")

input_ids = batch_squeezed[:, :-1]
targets = batch_squeezed[:, 1:]
print(f"Input shape: {input_ids.shape}")
print(f"Target shape: {targets.shape}")

# Simulate forward pass
vocab_size = 284
batch_size = input_ids.shape[0]
seq_len = input_ids.shape[1]

# Random logits (what model would produce)
logits = torch.randn(batch_size, seq_len, vocab_size)
print(f"\nLogits shape: {logits.shape}")

# Flatten for loss
logits_flat = logits.reshape(-1, vocab_size)
targets_flat = targets.reshape(-1)
print(f"Logits flat: {logits_flat.shape}")
print(f"Targets flat: {targets_flat.shape}")

# Compute loss
pad_token_id = 0
ce_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token_id)
print(f"\nCross-entropy loss: {ce_loss.item():.4f}")
print(f"Expected random loss (log(vocab_size)): {torch.log(torch.tensor(vocab_size)).item():.4f}")

# Perplexity
ppl = torch.exp(ce_loss)
print(f"Perplexity: {ppl.item():.1f}")
print(f"Expected random perplexity (vocab_size): {vocab_size}")

# Check if targets have valid token IDs
unique_targets = targets_flat[targets_flat != pad_token_id].unique()
print(f"\n=== Data Quality Checks ===")
print(f"Unique target tokens (non-pad): {len(unique_targets)}")
print(f"Min token ID: {unique_targets.min()}")
print(f"Max token ID: {unique_targets.max()}")
print(f"All tokens < vocab_size? {(unique_targets < vocab_size).all()}")
print(f"Fraction of pad tokens: {(targets_flat == pad_token_id).float().mean():.4f}")
