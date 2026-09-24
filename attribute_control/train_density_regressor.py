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

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from attribute_control.note_density import NoteDensityRegressor
from attribute_control.attributes import ATTRIBUTES, _anticipation_triplets


# ── Dataset wrapper ─────────────────────────────────────────────────────────

class DensityDataset(Dataset):
    """
    Wraps a music dataset to yield (mean_embedding, attribute_label) pairs.
    Despite the name (density was the first attribute this was built for),
    the label is computed by whatever compute_fn is passed in — this is
    reused as-is for velocity/duration/etc via attribute_control.attributes.

    Each sample simulates the mixed hard/soft input the regressor will
    actually receive at inference time — a *local*, bounded window, not the
    whole song: `hard_window` hard (committed) token embeddings immediately
    followed by exactly one soft (MCMC-noisy) next-token distribution,
    matching generate_music.py's per-step attribute energy (which only ever
    steers the single position about to be sampled). The label is the
    attribute value of that same local window (hard tokens + the true next
    token), not of the full song — the regressor can't see the rest of the
    song, so it shouldn't be asked to predict it. The window's start offset
    and the soft token's noise level are both randomised so the regressor is
    robust to where in a generation it's invoked and to different MCMC
    convergence stages.
    """

    def __init__(self, music_dataset, emb_weight: torch.Tensor,
                 tokenizer_type: str, vocab_size: int, compute_fn,
                 hard_window: int = 48, note_only_window: bool = False):
        self.ds             = music_dataset
        self.emb_weight     = emb_weight          # (vocab_size, emb_dim) on CPU
        self.tokenizer_type = tokenizer_type
        self.vocab_size     = vocab_size
        self.compute_fn     = compute_fn
        self.hard_window    = hard_window
        # Anticipation triplets are (time, duration, note) — a raw last-N-tokens
        # window is only 1-in-3 note tokens, diluting attributes (like
        # pitch_register) that are ONLY ever encoded in the note token, and
        # further muddied since note = instrument*128 + pitch. When True,
        # samples hard_window+1 NOTE tokens specifically (pulling enough raw
        # triplets to cover them) instead of hard_window+1 raw tokens — same
        # window SIZE convention, just undiluted. Must match generate_music.py's
        # Anticipation energy closure at inference (this flag is saved into the
        # checkpoint so inference can read it back rather than needing to be
        # told separately). duration's regressor does NOT use this — duration
        # information is redundantly present across token types, so the
        # dilution that breaks pitch_register barely affects it.
        self.note_only_window = note_only_window

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        tokens = self.ds.get_full_tokens(idx)   # full unclipped sequence
        if self.tokenizer_type.startswith('Anticipation'):
            # get_full_tokens() is the raw stored sequence: a leading
            # AUTOREGRESS/ANTICIPATE mode marker, and (for ANTICIPATE-mode
            # sequences, ~90% of training data) real event triplets
            # interleaved with anticipated-control triplets. Every stride-3
            # index below (hard/future tokens, note_only_window's [2::3])
            # assumes position 0 IS a real triplet boundary — true only
            # after this same cleanup generate_anticipation() already does
            # at inference (mode token stripped, ops.split() applied) before
            # it ever windows. Skipping it here meant training windows were
            # off-phase almost everywhere a raw slice didn't happen to start
            # right after a mode token, silently training on wrong-field
            # "note" embeddings and wrong attribute labels alike — confirmed
            # via a before/after eval showing both label and prediction
            # ranges had collapsed toward 0 for a pitch_register regressor
            # trained this way (see docs/diary/2026-09-24.md).
            tokens = _anticipation_triplets(tokens)
        N = len(tokens)
        n_needed = self.hard_window + 1         # hard tokens + 1 soft token

        if self.note_only_window:
            raw_window_size = n_needed * 3      # n_needed triplets
            if N < raw_window_size:
                # Too short for a full window — use whatever whole triplets exist.
                raw_start = 0
                raw_window_size = N - (N % 3)
            else:
                max_start = N - raw_window_size
                raw_start = random.randint(0, max_start - (max_start % 3)) if max_start > 0 else 0
                raw_start -= raw_start % 3
            window = tokens[raw_start:raw_start + raw_window_size]
            note_tokens = window[2::3]
            if len(note_tokens) < 2:
                note_tokens = (note_tokens * 2)[:2]  # degenerate ultra-short song fallback
            hard_tokens, future_tokens = note_tokens[:-1], note_tokens[-1:]
        elif N < n_needed:
            # Song too short for a full window — use the whole thing, hard only.
            window = tokens
            hard_tokens, future_tokens = window[:-1] or window, window[-1:]
        else:
            start = random.randint(0, N - n_needed)
            window = tokens[start:start + n_needed]
            hard_tokens, future_tokens = window[:-1], window[-1:]

        label = self.compute_fn(window, self.tokenizer_type)

        with torch.no_grad():
            hard_idx = torch.tensor(hard_tokens, dtype=torch.long)
            future_idx = torch.tensor(future_tokens, dtype=torch.long)

            hard_emb_sum = self.emb_weight[hard_idx].sum(0)    # (D,)

            # Simulate MCMC state for the one soft/future token: peaked but
            # noisy logits. alpha ~ LogNormal(1.5, 0.7) → mostly in [2, 15],
            # covering early (uniform-ish) to late (nearly one-hot) MCMC steps.
            alpha = math.exp(random.gauss(1.5, 0.7))
            one_hot = torch.zeros(len(future_idx), self.vocab_size)
            one_hot.scatter_(1, future_idx.unsqueeze(1), 1.0)
            logits  = alpha * one_hot + torch.randn_like(one_hot)
            probs   = torch.softmax(logits, dim=-1)             # (1, V)
            fut_soft_sum = (probs @ self.emb_weight).sum(0)     # (D,)

            # Normalize by the actual number of embedding "slots" summed
            # (hard + 1 soft) — NOT len(window), which in note_only_window
            # mode is the raw triplet span, not the note-token count (matches
            # generate_music.py's inference-time convention: total = n_hard + n_soft).
            e_mean = (hard_emb_sum + fut_soft_sum) / (len(hard_tokens) + len(future_tokens))

        return e_mean, torch.tensor(label, dtype=torch.float32)


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

    # Sub-sample large Anticipation validation split the same way
    if args.max_val_samples and len(val_music) > args.max_val_samples:
        indices = random.sample(range(len(val_music)), args.max_val_samples)
        from torch.utils.data import Subset
        val_music = Subset(val_music, indices)
        val_music.get_full_tokens = lambda i: val_music.dataset.get_full_tokens(val_music.indices[i])

    compute_fn = ATTRIBUTES[args.attribute]

    vocab_size = emb_weight.shape[0]
    train_ds = DensityDataset(train_music, emb_weight, args.tokenizer_type, vocab_size=vocab_size,
                               compute_fn=compute_fn, hard_window=args.density_hard_window,
                               note_only_window=args.note_only_window)
    val_ds   = DensityDataset(val_music,   emb_weight, args.tokenizer_type, vocab_size=vocab_size,
                               compute_fn=compute_fn, hard_window=args.density_hard_window,
                               note_only_window=args.note_only_window)

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

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE and getattr(args, 'wandb_project', None)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=getattr(args, 'wandb_run_name', None) or Path(args.output_dir).name,
            config={
                'attribute':      args.attribute,
                'ebt_checkpoint': args.checkpoint,
                'tokenizer_type': args.tokenizer_type,
                'emb_dim':        emb_dim,
                'hidden_dim':     args.hidden_dim,
                'epochs':         args.epochs,
                'batch_size':     args.batch_size,
                'lr':             args.lr,
                'train_samples':  len(train_ds),
                'val_samples':    len(val_ds),
            },
        )
    # ──────────────────────────────────────────────────────────────────────────

    best_val_loss = float('inf')
    print(f"\nTraining {args.attribute} regressor  emb_dim={emb_dim}  hidden={args.hidden_dim}")

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

        if use_wandb:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'epoch': epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                'epoch':          epoch,
                'val_loss':       val_loss,
                'emb_dim':        emb_dim,
                'hidden_dim':     args.hidden_dim,
                'attribute':      args.attribute,
                'tokenizer_type': args.tokenizer_type,
                'ebt_checkpoint': args.checkpoint,
                'density_hard_window': args.density_hard_window,
                'note_only_window': args.note_only_window,
                'model_state':    model.state_dict(),
            }
            save_path = output_dir / 'best.pt'
            torch.save(ckpt, save_path)
            print(f"  ✅ Saved best checkpoint → {save_path}")

    if use_wandb:
        wandb.summary['best_val_loss'] = best_val_loss
        wandb.finish()

    print(f"\nTraining complete. Best val_loss={best_val_loss:.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',       required=True,  help='EBT checkpoint to extract embeddings from')
    p.add_argument('--attribute',        default='density', choices=list(ATTRIBUTES.keys()),
                   help='Which attribute to train a regressor for')
    p.add_argument('--tokenizer_type',   default='REMI', choices=['REMI', 'Anticipation-Arrival-Time'])
    p.add_argument('--output_dir',       required=True)
    p.add_argument('--epochs',           type=int,   default=10)
    p.add_argument('--batch_size',       type=int,   default=512)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--hidden_dim',       type=int,   default=256)
    p.add_argument('--density_hard_window', type=int, default=48,
                   help='Local hard-token window size (tokens) fed alongside the single '
                        'soft next-token; must match generate_music.py at inference')
    p.add_argument('--note_only_window', action='store_true',
                   help='Anticipation only: build the hard window from NOTE tokens only '
                        '(undiluted by time/duration tokens) instead of raw last-N tokens. '
                        'Use for attributes only encoded in the note token, e.g. pitch_register.')
    p.add_argument('--num_workers',      type=int,   default=8)
    p.add_argument('--max_val_samples',   type=int,  default=50_000,
                    help='Cap validation set size (Anticipation val splits can be 100x larger than REMI\'s)')
    p.add_argument('--max_train_samples', type=int,  default=500_000,
                   help='Cap on training sequences (use None for full dataset)')
    p.add_argument('--device',           default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--wandb_project',    default=None,  help='WandB project name (omit to disable)')
    p.add_argument('--wandb_run_name',   default=None,  help='WandB run name (defaults to output_dir basename)')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
