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

from attribute_control.note_density import REMI_PITCHDRUM_MIN, REMI_PITCHDRUM_MAX

# Velocity_3 .. Velocity_127: 32 tokens, MIDI velocity units, step 4.
REMI_VELOCITY_MIN_ID = 94
REMI_VELOCITY_MAX_ID = 125
REMI_VELOCITY_MIN_VAL = 3
REMI_VELOCITY_STEP = 4

# Duration_0.1.8 .. Duration_12.0.4: 64 tokens, bins coarsen short -> long.
REMI_DURATION_MIN_ID = 126
REMI_DURATION_MAX_ID = 189

# Position_0 .. Position_31: 32 tokens, one per subdivision of a bar.
REMI_POSITION_MIN_ID = 190
REMI_POSITION_MAX_ID = 221

# Pitch_21 .. Pitch_109: 89 tokens, linear MIDI pitch (id + 16 = MIDI note).
# Excludes PitchDrum (222-283) deliberately — those encode WHICH drum-kit
# piece, not tonal height, so mixing them in would corrupt a register metric.
REMI_PITCH_REGISTER_MIN_ID = 5
REMI_PITCH_REGISTER_MAX_ID = 93
REMI_PITCH_MIDI_OFFSET = 16

# Program tokens: 143 tokens (ids 284-426, incl. 426=Program_-1 the drum-kit
# marker) sitting outside tokenizer.vocab's main dict (miditok's separate
# multi-vocab program handling) — empirically the ~90-100% deterministic
# lead-in to a Pitch/PitchDrum token, used to gate pitch-register guidance.
REMI_PROGRAM_MIN_ID = 284
REMI_PROGRAM_MAX_ID = 426

# Bar_None: the single bar-boundary token.
REMI_BAR_ID = 4


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


def compute_pitch_register(tokens: List[int], tokenizer_type: str) -> float:
    """
    Mean MIDI pitch (0-1, normalized by 127) of Pitch tokens in the sequence —
    a low/bass-register (near 0) vs high/treble-register (near 1) proxy.
    PitchDrum tokens are deliberately excluded: they identify a drum-kit
    piece, not a tonal height, so including them would mix two unrelated
    scales into one meaningless average.
    """
    if tokenizer_type != 'REMI':
        raise NotImplementedError("compute_pitch_register only supports REMI for now")
    vals = [
        (t + REMI_PITCH_MIDI_OFFSET) / 127.0
        for t in tokens if REMI_PITCH_REGISTER_MIN_ID <= t <= REMI_PITCH_REGISTER_MAX_ID
    ]
    return sum(vals) / len(vals) if vals else 0.0


def compute_polyphony(tokens: List[int], tokenizer_type: str) -> float:
    """
    Mean number of simultaneous notes per onset (chord thickness), distinct
    from density's note-onset-rate: a fast monophonic run and a slow chordal
    passage can have the same density but very different polyphony.

    REMI groups simultaneous notes under one Position/Bar token (chord tones
    share a timestamp, each prefixed by its own Program token but with no
    repeated Position in between) — so this counts Pitch/PitchDrum tokens
    between consecutive Position/Bar boundaries, giving one "chord size" per
    onset group, and averages over only the NON-EMPTY groups (rests/gaps are
    excluded, not counted as size-0 chords, so they don't dilute the average
    the way they do for density).
    """
    if tokenizer_type != 'REMI':
        raise NotImplementedError("compute_polyphony only supports REMI for now")
    is_boundary = lambda t: t == REMI_BAR_ID or REMI_POSITION_MIN_ID <= t <= REMI_POSITION_MAX_ID
    is_note = lambda t: (REMI_PITCH_REGISTER_MIN_ID <= t <= REMI_PITCH_REGISTER_MAX_ID
                          or REMI_PITCHDRUM_MIN <= t <= REMI_PITCHDRUM_MAX)
    group_sizes = []
    current = 0
    started = False
    for t in tokens:
        if is_boundary(t):
            if started and current > 0:
                group_sizes.append(current)
            current = 0
            started = True
        elif is_note(t):
            current += 1
    if started and current > 0:
        group_sizes.append(current)
    return sum(group_sizes) / len(group_sizes) if group_sizes else 0.0


def compute_rhythm_pace(tokens: List[int], tokenizer_type: str) -> float:
    """
    Mean gap (in Position units, 0-31 per bar) between consecutive note
    onsets — how closely packed or spread out notes are in time. Smaller =
    faster/denser pacing (onsets come frequently), larger = slower/sparser
    (onsets come rarely). Distinct from density (fraction of tokens that are
    notes, insensitive to actual time between them) and from polyphony
    (chord thickness at a single onset, not spacing between onsets).

    Only counts forward, same-bar gaps (bar-wrap and cross-bar jumps are
    skipped rather than guessed at, since Bar tokens don't carry an absolute
    time value in this vocab) — a conservative undercount of true pacing
    across bar lines, but avoids fabricating a wrong gap value at the seam.
    """
    if tokenizer_type != 'REMI':
        raise NotImplementedError("compute_rhythm_pace only supports REMI for now")
    position_vals = [
        t - REMI_POSITION_MIN_ID for t in tokens
        if REMI_POSITION_MIN_ID <= t <= REMI_POSITION_MAX_ID
    ]
    gaps = [b - a for a, b in zip(position_vals, position_vals[1:]) if b > a]
    return sum(gaps) / len(gaps) if gaps else 0.0


def compute_drum_density(tokens: List[int], tokenizer_type: str) -> float:
    """
    Fraction of note tokens that are PitchDrum rather than Pitch — 0 = purely
    tonal/melodic, 1 = purely percussive. ~73% of this corpus's songs contain
    drums (see note_density.py), so this is a well-populated axis rather than
    one that's mostly zeros.
    """
    if tokenizer_type != 'REMI':
        raise NotImplementedError("compute_drum_density only supports REMI for now")
    n_drum = sum(1 for t in tokens if REMI_PITCHDRUM_MIN <= t <= REMI_PITCHDRUM_MAX)
    n_pitch = sum(1 for t in tokens if REMI_PITCH_REGISTER_MIN_ID <= t <= REMI_PITCH_REGISTER_MAX_ID)
    total = n_drum + n_pitch
    return n_drum / total if total else 0.0


def compute_melodic_interval(tokens: List[int], tokenizer_type: str) -> float:
    """
    Mean absolute semitone distance between consecutive Pitch tokens (in
    token-sequence order) — smooth/stepwise motion (near 0) vs. large,
    disjunct leaps (higher values), a cheap proxy for "how jumpy/atonal-
    sounding" the pitch choices are. PitchDrum tokens are excluded (drum-kit
    piece IDs aren't a tonal pitch scale, so a "distance" between them is
    meaningless).

    Unlike polyphony/pitch_register this does NOT group by onset —
    simultaneous chord tones are treated as sequential steps, which inflates
    the value somewhat when polyphony is high. Deliberately kept this simple
    as a cheap diagnostic: most guided passages here are close to monophonic,
    and a chord-aware version would need the same onset-grouping machinery as
    polyphony for comparatively little benefit on this project's material.
    """
    if tokenizer_type != 'REMI':
        raise NotImplementedError("compute_melodic_interval only supports REMI for now")
    pitch_vals = [
        t for t in tokens
        if REMI_PITCH_REGISTER_MIN_ID <= t <= REMI_PITCH_REGISTER_MAX_ID
    ]
    intervals = [abs(b - a) for a, b in zip(pitch_vals, pitch_vals[1:])]
    return sum(intervals) / len(intervals) if intervals else 0.0


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
    'pitch_register': compute_pitch_register,
    'polyphony': compute_polyphony,
    'rhythm': compute_rhythm_pace,
    'drum_density': compute_drum_density,
    'melodic_interval': compute_melodic_interval,
}
