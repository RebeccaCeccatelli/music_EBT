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
    """
    t = list(tokens)
    if len(t) < MIN_EBT_INPUT:
        t = t + [0] * (MIN_EBT_INPUT - len(t))
    x = torch.tensor(t, dtype=torch.long, device=device).unsqueeze(0)
    _, energies = model.forward(x, start_pos=0, learning=False, return_raw_logits=True)
    return float(energies[-1].mean().detach().item())


def score_sample(model, tokens: List[int], device, bigram_table: np.ndarray) -> dict:
    """Convenience wrapper: all three metrics for one generated sample."""
    return {
        "ebt_energy": ebt_self_energy(model, tokens, device),
        "bigram_ll": bigram_log_likelihood(tokens, bigram_table),
        "repetition_ratio": repetition_ratio(tokens),
    }
