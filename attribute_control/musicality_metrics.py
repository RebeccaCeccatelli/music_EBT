"""
Cheap, token-level proxies for "does this still sound like music" — meant to
catch the failure mode where attribute guidance (e.g. density steering in
generate_music.py) hits its numeric target but destroys musical coherence.
None of these require audio; they operate directly on REMI token sequences.

Three independent signals, since any one alone can be fooled:
  - ebt_self_energy:      the model's OWN learned energy for the sequence.
                           Rises if guidance pushes tokens the model wouldn't
                           naturally produce.
  - bigram_log_likelihood: plausibility under a transition table built from
                           REAL training data — independent of the model's
                           own (possibly-guided-into-miscalibration) opinion.
  - repetition_ratio:      fraction of repeated 4-grams — cheap detector for
                           the classic "guidance collapses into a loop" failure.

Usage: build_bigram_logprob_table() once (cached to disk), then call the
three scoring functions per generated sample.
"""

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import torch

from attribute_control.note_density import (
    REMI_PITCH_MIN, REMI_PITCH_MAX, REMI_PITCHDRUM_MIN, REMI_PITCHDRUM_MAX,
)
from attribute_control.attributes import (
    REMI_VELOCITY_MIN_ID, REMI_VELOCITY_MAX_ID,
    REMI_DURATION_MIN_ID, REMI_DURATION_MAX_ID,
    compute_melodic_interval,
)


MIN_EBT_INPUT = 2  # matches inference/mus/generate_music.py's EBT padding requirement


def build_bigram_logprob_table(
    dataset,
    vocab_size: int,
    n_songs: int = 3000,
    seed: int = 0,
    cache_path: str | None = None,
) -> np.ndarray:
    """
    (vocab_size, vocab_size) log P(next_token | token) table from real songs,
    with add-one (Laplace) smoothing. Cached to disk (as .npy) since building
    it requires scanning thousands of songs — expensive to redo per benchmark run.
    """
    if cache_path and Path(cache_path).exists():
        return np.load(cache_path)

    import random
    rng = random.Random(seed)
    n = min(n_songs, len(dataset))
    indices = rng.sample(range(len(dataset)), n)

    counts = np.ones((vocab_size, vocab_size), dtype=np.float64)  # add-one smoothing
    for idx in indices:
        tokens = dataset.get_full_tokens(idx)
        if len(tokens) < 2:
            continue
        t = np.asarray(tokens, dtype=np.int64)
        t = np.clip(t, 0, vocab_size - 1)
        np.add.at(counts, (t[:-1], t[1:]), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    logprobs = np.log(counts / row_sums)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, logprobs)

    return logprobs


def bigram_log_likelihood(tokens: List[int], logprob_table: np.ndarray) -> float:
    """Mean log P(next | current) over the sequence under the real-data table."""
    if len(tokens) < 2:
        return 0.0
    vocab_size = logprob_table.shape[0]
    t = np.clip(np.asarray(tokens, dtype=np.int64), 0, vocab_size - 1)
    return float(logprob_table[t[:-1], t[1:]].mean())


def repetition_ratio(tokens: List[int], n: int = 4) -> float:
    """Fraction of n-grams that are exact repeats of an earlier n-gram in the sequence."""
    if len(tokens) < 2 * n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    seen = set()
    repeats = 0
    for g in ngrams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(ngrams)


def ebt_self_energy(model, tokens: List[int], device) -> float:
    """
    The EBT's own final-MCMC-step energy for this sequence (lower = the model
    finds it more self-consistent). Runs the model's normal forward pass in
    scoring mode (learning=False) — no guidance, just "how does the model
    rate this sequence."

    Only meaningful for EBT: it's the one architecture whose forward pass
    returns a per-MCMC-step energy at all (baseline Llama/GPT2 have no such
    concept — there's no energy to report). Returns NaN for any other model
    rather than crashing, so score_sample() stays usable across architectures.
    """
    t = list(tokens)
    if len(t) < MIN_EBT_INPUT:
        t = t + [0] * (MIN_EBT_INPUT - len(t))
    x = torch.tensor(t, dtype=torch.long, device=device).unsqueeze(0)
    try:
        _, energies = model.forward(x, start_pos=0, learning=False, return_raw_logits=True)
    except TypeError:
        return float('nan')
    return float(energies[-1].mean().detach().item())


def grammar_violation_rate(tokens: List[int]) -> float:
    """
    Fraction of Pitch/PitchDrum-> and Velocity-> transitions that violate
    REMI's grammar, which is empirically 100% deterministic at these two
    points (checked over 681k real tokens: a Pitch/PitchDrum token is ALWAYS
    followed by Velocity, a Velocity token is ALWAYS followed by Duration —
    see generate_music.py's _remi_step_is_relevant). A well-formed sequence
    should score ~0 here regardless of how "usual" its content is, unlike
    bigram_ll — this catches literal format breakdown (the model emitting a
    token sequence that isn't even valid REMI), not just unusual-sounding
    but still well-formed output.
    """
    violations = 0
    checked = 0
    for a, b in zip(tokens, tokens[1:]):
        is_note = (REMI_PITCH_MIN <= a <= REMI_PITCH_MAX
                   or REMI_PITCHDRUM_MIN <= a <= REMI_PITCHDRUM_MAX)
        is_velocity = REMI_VELOCITY_MIN_ID <= a <= REMI_VELOCITY_MAX_ID
        if is_note:
            checked += 1
            if not (REMI_VELOCITY_MIN_ID <= b <= REMI_VELOCITY_MAX_ID):
                violations += 1
        elif is_velocity:
            checked += 1
            if not (REMI_DURATION_MIN_ID <= b <= REMI_DURATION_MAX_ID):
                violations += 1
    return violations / checked if checked else 0.0


def segment_scores(tokens: List[int], bigram_table: np.ndarray, n_segments: int = 4) -> List[dict]:
    """
    Split `tokens` into `n_segments` equal-length chunks (last chunk absorbs
    any remainder) and compute bigram_ll/repetition_ratio/grammar_violation_rate
    independently on each — a single whole-sequence average can hide a piece
    that's fine early on and degenerates later (or vice versa); this exposes
    the trend directly. Returns a list of per-segment dicts, ordered start to
    end. Doesn't include ebt_energy (would need one model forward pass per
    segment per sample — left out here to keep sweep cost down).
    """
    n = len(tokens)
    if n < n_segments:
        n_segments = max(1, n)
    chunk = n // n_segments
    segments = []
    for i in range(n_segments):
        start = i * chunk
        end = n if i == n_segments - 1 else (i + 1) * chunk
        seg = tokens[start:end]
        segments.append({
            "bigram_ll": bigram_log_likelihood(seg, bigram_table),
            "repetition_ratio": repetition_ratio(seg),
            "grammar_violation_rate": grammar_violation_rate(seg),
            "melodic_interval": compute_melodic_interval(seg, 'REMI'),
        })
    return segments


def score_sample(model, tokens: List[int], device, bigram_table: np.ndarray) -> dict:
    """Convenience wrapper: all three metrics for one generated sample."""
    return {
        "ebt_energy": ebt_self_energy(model, tokens, device),
        "bigram_ll": bigram_log_likelihood(tokens, bigram_table),
        "repetition_ratio": repetition_ratio(tokens),
        "grammar_violation_rate": grammar_violation_rate(tokens),
        "melodic_interval": compute_melodic_interval(tokens, 'REMI'),
    }
