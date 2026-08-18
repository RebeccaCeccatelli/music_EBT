"""
Note density attribute: computation and regressor model.

Density is defined per tokenizer:
  REMI:         pitch_token_count / context_length
                Pitch token IDs: 5..93 (89 MIDI pitches, from miditok REMI vocab)
                Typical range: 0.05 (sparse melody) – 0.25 (dense polyphony)

  Anticipation: note_events_per_second
                Arrival times encoded directly in tokens (TIME_OFFSET=0, 100 bins/s).
                Typical range: 1 – 40 notes/second.

Both are output as raw floats from compute_density(). Training and inference use
the same units so targets set by the user are directly interpretable.
"""

import torch
import torch.nn as nn
from typing import List


# ── REMI constants ──────────────────────────────────────────────────────────
REMI_PITCH_MIN = 5
REMI_PITCH_MAX = 93


# ── Anticipation constants ──────────────────────────────────────────────────
ANT_TIME_OFFSET  = 0
ANT_NOTE_OFFSET  = 11000
ANT_REST         = 27512   # REST = NOTE_OFFSET + MAX_NOTE (pad token)
ANT_TIME_RES     = 100     # bins per second


# ── Density computation ─────────────────────────────────────────────────────

def compute_density_remi(tokens: List[int]) -> float:
    """Fraction of tokens that are Pitch events (0–1)."""
    if not tokens:
        return 0.0
    n_pitch = sum(REMI_PITCH_MIN <= t <= REMI_PITCH_MAX for t in tokens)
    return n_pitch / len(tokens)


def compute_density_anticipation(tokens: List[int]) -> float:
    """
    Notes per second, computed from arrival-time tokens.

    Anticipation sequences: tokens[0] is the mode token; tokens[1:] are
    341 triplets of (arrival_time, duration, note).  We use the note slot
    to distinguish event notes from rests/controls, and the time slot for
    the span.
    """
    if len(tokens) < 4:
        return 0.0
    triplet_tokens = tokens[1:]               # skip mode token
    time_toks = triplet_tokens[0::3]
    note_toks  = triplet_tokens[2::3]

    # arrival times in seconds (only event note positions, not controls)
    arrival_times = []
    n_notes = 0
    for t, n in zip(time_toks, note_toks):
        if ANT_NOTE_OFFSET <= n <= ANT_REST:  # event note (incl. REST)
            arrival_times.append((t - ANT_TIME_OFFSET) / ANT_TIME_RES)
            if n < ANT_REST:                  # REST doesn't count as a note
                n_notes += 1

    if n_notes == 0 or len(arrival_times) < 2:
        return 0.0
    time_span = max(arrival_times) - min(arrival_times)
    if time_span < 0.1:
        return 0.0
    return n_notes / time_span


def compute_density(tokens: List[int], tokenizer_type: str) -> float:
    if tokenizer_type == 'REMI':
        return compute_density_remi(tokens)
    return compute_density_anticipation(tokens)


# ── Regressor architecture ──────────────────────────────────────────────────

class NoteDensityRegressor(nn.Module):
    """
    Lightweight MLP that predicts note density from a mean token embedding.

    Input:  e_mean  (B, emb_dim)  — mean of frozen EBT token embeddings
    Output: density (B,)           — scalar, same units as compute_density()

    The EBT embedding table is NOT part of this module; it is passed at
    inference time so the model stays tokenizer-agnostic.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, e_mean: torch.Tensor) -> torch.Tensor:
        return self.net(e_mean).squeeze(-1)   # (B,)

    # ── convenience helpers ────────────────────────────────────────────────

    @staticmethod
    def mean_embedding_hard(
        tokens: List[int],
        emb_weight: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Mean of hard (discrete) token embeddings. Not differentiable."""
        idx = torch.tensor(tokens, dtype=torch.long, device=device)
        return emb_weight[idx].mean(0)          # (emb_dim,)

    @staticmethod
    def soft_embedding(
        logits: torch.Tensor,
        emb_weight: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Differentiable soft embedding: softmax(logits/T) @ W_emb.

        logits:     (vocab_size,)  — raw logits for the next token position
        emb_weight: (vocab_size, emb_dim)
        returns:    (emb_dim,)
        """
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs @ emb_weight               # (emb_dim,)
