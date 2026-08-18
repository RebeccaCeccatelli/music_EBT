"""
Train the NoteDensityRegressor on frozen EBT token embeddings.

The EBT embedding table is extracted from a checkpoint and frozen.
For each sequence in the training set:
  1. Compute note density label (on-the-fly, no storage needed)
  2. Look up mean embedding of the token sequence
  3. Train MLP with MSE loss

Usage (see also job_scripts/mus/attr_control/train_density.sh):
    python attribute_control/train_density_regressor.py \
        --checkpoint <ebt_ckpt_path> \
        --tokenizer_type REMI \
        --output_dir ./attr_checkpoints/density_remi
"""

import os
import sys
import argparse
import random
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from argparse import Namespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from attribute_control.note_density import NoteDensityRegressor, compute_density


# ── Dataset wrapper ─────────────────────────────────────────────────────────

class DensityDataset(Dataset):
    """
    Wraps a music dataset to yield (mean_embedding, density_label) pairs.

    Embeddings are pre-fetched from the frozen EBT embedding table so the
    training loop never touches the full model.
    """

    def __init__(self, music_dataset, emb_weight: torch.Tensor, tokenizer_type: str):
        self.ds           = music_dataset
        self.emb_weight   = emb_weight          # (vocab_size, emb_dim) on CPU
        self.tokenizer_type = tokenizer_type

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        tokens = self.ds.get_full_tokens(idx)   # full unclipped sequence
        density = compute_density(tokens, self.tokenizer_type)

        # mean of hard token embeddings (no grad needed)
        with torch.no_grad():
            idx_t  = torch.tensor(tokens, dtype=torch.long)
            e_mean = self.emb_weight[idx_t].mean(0)   # (emb_dim,)

        return e_mean, torch.tensor(density, dtype=torch.float32)


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_embedding_weight(ckpt_path: str) -> torch.Tensor:
    """Load the embedding weight matrix from an EBT checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']

    key = 'model.embeddings.weight'
    if key not in state:
        raise KeyError(
            f"'{key}' not found in checkpoint. Available keys starting with "
            f"'model.embed': {[k for k in state if 'embed' in k][:10]}"
        )
    w = state[key].float()
    print(f"  Embedding shape: {tuple(w.shape)}  (vocab_size={w.shape[0]}, emb_dim={w.shape[1]})")
    return w


def load_music_dataset(tokenizer_type: str, split: str, hparams: Namespace):
    if tokenizer_type == 'REMI':
        from data.mus.symbolic.dataloaders.giga_midi_miditok_dataset import GigaMIDIMiditokDataset
        return GigaMIDIMiditokDataset(hparams, split=split)
    else:
        ant_dir = {
            'Anticipation-Arrival-Time': 'anticipation',
            'Anticipation-Vanilla':      'anticipation-vanilla',
            'Anticipation-Interarrival': 'anticipation-interarrival',
        }[tokenizer_type]
        from data.mus.symbolic.dataloaders.custom_music_dataloader import CustomMusicDataset
        return CustomMusicDataset(hparams, 'giga-midi', split, ant_dir)


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    # ── Extract frozen embeddings ─────────────────────────────────────────
    emb_weight = extract_embedding_weight(args.checkpoint)
    emb_dim    = emb_weight.shape[1]

    # ── Load datasets ─────────────────────────────────────────────────────
    hparams = Namespace(
        context_length=512,
        dataset_name='giga-midi',
        data_dir=None,
        tokenizer_type=args.tokenizer_type,
    )

    print(f"\nLoading datasets ({args.tokenizer_type})...")
    train_music = load_music_dataset(args.tokenizer_type, 'train',      hparams)
    val_music   = load_music_dataset(args.tokenizer_type, 'validation', hparams)

    # Sub-sample large Anticipation train split to keep training time manageable
    if args.max_train_samples and len(train_music) > args.max_train_samples:
        indices = random.sample(range(len(train_music)), args.max_train_samples)
        from torch.utils.data import Subset
        train_music = Subset(train_music, indices)
        # Patch get_full_tokens through the Subset
        train_music.get_full_tokens = lambda i: train_music.dataset.get_full_tokens(train_music.indices[i])

    train_ds = DensityDataset(train_music, emb_weight, args.tokenizer_type)
    val_ds   = DensityDataset(val_music,   emb_weight, args.tokenizer_type)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )
    print(f"  Train: {len(train_ds):,}   Val: {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = NoteDensityRegressor(emb_dim=emb_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    criterion = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    print(f"\nTraining NoteDensityRegressor  emb_dim={emb_dim}  hidden={args.hidden_dim}")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for e_mean, density in train_loader:
            e_mean  = e_mean.to(device)
            density = density.to(device)

            pred = model(e_mean)
            loss = criterion(pred, density)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for e_mean, density in val_loader:
                e_mean  = e_mean.to(device)
                density = density.to(device)
                pred     = model(e_mean)
                val_loss += criterion(pred, density).item()
        val_loss /= len(val_loader)

        print(f"  Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                'epoch':          epoch,
                'val_loss':       val_loss,
                'emb_dim':        emb_dim,
                'hidden_dim':     args.hidden_dim,
                'tokenizer_type': args.tokenizer_type,
                'ebt_checkpoint': args.checkpoint,
                'model_state':    model.state_dict(),
            }
            save_path = output_dir / 'best.pt'
            torch.save(ckpt, save_path)
            print(f"  ✅ Saved best checkpoint → {save_path}")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',       required=True,  help='EBT checkpoint to extract embeddings from')
    p.add_argument('--tokenizer_type',   default='REMI', choices=['REMI', 'Anticipation-Arrival-Time'])
    p.add_argument('--output_dir',       required=True)
    p.add_argument('--epochs',           type=int,   default=10)
    p.add_argument('--batch_size',       type=int,   default=512)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--hidden_dim',       type=int,   default=256)
    p.add_argument('--num_workers',      type=int,   default=8)
    p.add_argument('--max_train_samples', type=int,  default=500_000,
                   help='Cap on training sequences (use None for full dataset)')
    p.add_argument('--device',           default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
