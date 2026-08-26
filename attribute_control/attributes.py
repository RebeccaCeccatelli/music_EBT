"""
Additional attribute definitions for classifier-guided generation, alongside
note_density.py's compute_density(). Each compute_X() mirrors its contract:
takes a REMI token sequence, returns a single float. REMI-only for now.

Token ID ranges below are read directly off this project's REMI tokenizer
vocab (data/mus/symbolic/tokenization/... giga-midi/miditok/tokenizer.json).
Re-derive them if that tokenizer config ever changes:
    tok, _, _ = load_tokenizer(tokenizer_type='REMI', tokenizer_config_path=..., dataset_name='giga_midi')
    sorted((k, v) for k, v in tok.vocab.items() if k.startswith('Velocity'))
"""

from typing import List

# Velocity_3 .. Velocity_127: 32 tokens, MIDI velocity units, step 4.
REMI_VELOCITY_MIN_ID = 94
REMI_VELOCITY_MAX_ID = 125
REMI_VELOCITY_MIN_VAL = 3
REMI_VELOCITY_STEP = 4

# Duration_0.1.8 .. Duration_12.0.4: 64 tokens, bins coarsen short -> long.
REMI_DURATION_MIN_ID = 126
REMI_DURATION_MAX_ID = 189


def compute_velocity(tokens: List[int], tokenizer_type: str) -> float:
    """Mean MIDI velocity (0-1, normalized by 127) of Velocity tokens in the sequence."""
    if tokenizer_type != 'REMI':
        raise NotImplementedError("compute_velocity only supports REMI for now")
    vals = [
        (REMI_VELOCITY_MIN_VAL + REMI_VELOCITY_STEP * (t - REMI_VELOCITY_MIN_ID)) / 127.0
        for t in tokens if REMI_VELOCITY_MIN_ID <= t <= REMI_VELOCITY_MAX_ID
    ]
    return sum(vals) / len(vals) if vals else 0.0


def compute_duration_bias(tokens: List[int], tokenizer_type: str) -> float:
    """
    Mean normalized note length (0=shortest, 1=longest bin) of Duration tokens
    in the sequence — a staccato (near 0) vs legato/sustained (near 1) proxy.
    Uses each duration bin's rank in the vocab (bins are already ordered
    short -> long by construction) rather than parsing the
    beat.subdivision.resolution token name.
    """
    if tokenizer_type != 'REMI':
        raise NotImplementedError("compute_duration_bias only supports REMI for now")
    span = REMI_DURATION_MAX_ID - REMI_DURATION_MIN_ID
    vals = [
        (t - REMI_DURATION_MIN_ID) / span
        for t in tokens if REMI_DURATION_MIN_ID <= t <= REMI_DURATION_MAX_ID
    ]
    return sum(vals) / len(vals) if vals else 0.0


def _compute_density_remi(tokens, tokenizer_type):
    from attribute_control.note_density import compute_density
    return compute_density(tokens, tokenizer_type)


# Name -> compute_fn(tokens, tokenizer_type) -> float. Used by
# train_density_regressor.py (--attribute) and benchmark_density_control.py
# to pick the right label/scoring function generically.
ATTRIBUTES = {
    'density': _compute_density_remi,
    'velocity': compute_velocity,
    'duration': compute_duration_bias,
}
