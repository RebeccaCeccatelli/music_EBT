"""
Wrapper for Anticipation tokenizer to provide decode() interface compatible with inference pipeline.

The decode() method converts anticipation event tokens to MIDI using the anticipation library's
events_to_midi() function, providing the same interface as miditok tokenizers.
"""

import sys
from pathlib import Path


class AnticipationTokenizerWrapper:
    """
    Wrapper providing decode() method for anticipation tokens.

    Makes anticipation tokenization compatible with the inference pipeline that expects
    tokenizer.decode(tokens) -> MIDI object interface.
    """

    def __init__(self, variant: str = 'Arrival-Time', use_vanilla: bool = False):
        """
        Initialize wrapper.

        Args:
            variant: 'Vanilla', 'Interarrival', or 'Arrival-Time'
            use_vanilla: Legacy flag for vanilla variant
        """
        self.variant = variant
        self.use_vanilla = use_vanilla or (variant == 'Vanilla')

        # Set environment flag for anticipation library
        import os
        if self.use_vanilla:
            os.environ['ANTICIPATION_VANILLA'] = 'true'
        else:
            os.environ['ANTICIPATION_VANILLA'] = 'false'

    def decode(self, tokens):
        """
        Convert anticipation tokens to MIDI.

        Args:
            tokens: List of token IDs in anticipation format

        Returns:
            mido.MidiFile object (compatible with tokens_to_midi.py)
        """
        import sys
        import os

        # Add anticipation submodule to path if not already there
        anticipation_root = os.path.join(
            os.path.dirname(__file__), 'anticipation'
        )
        if anticipation_root not in sys.path:
            sys.path.insert(0, anticipation_root)

        try:
            from anticipation import convert, ops
            from anticipation.vocab_selector import AUTOREGRESS, ANTICIPATE
        except ImportError as e:
            raise ImportError(
                f"Anticipation library not found: {e}. "
                f"Make sure anticipation is installed as a submodule in "
                f"data/mus/symbolic/tokenization/anticipation/"
            )

        # Strip mode token if present (first token is AUTOREGRESS or ANTICIPATE)
        tokens = list(tokens) if not isinstance(tokens, list) else tokens
        if len(tokens) > 0 and tokens[0] in [AUTOREGRESS, ANTICIPATE]:
            tokens = tokens[1:]

        # Trim to triplet boundary — events come in (time, dur, note) triplets
        remainder = len(tokens) % 3
        if remainder != 0:
            tokens = tokens[:-remainder]

        # Strip control tokens — ANTICIPATE-mode prompts have interleaved controls
        # that represent anticipated future events. removeControls in events_to_compound
        # maps them into the event vocab range, producing phantom notes. Keep events only.
        if len(tokens) > 0:
            events_only, _ = ops.split(tokens)
            tokens = list(events_only)

        # Normalize timestamps to start at 0 — dataset windows use absolute times, so a
        # window starting mid-song would otherwise produce leading silence in the MIDI.
        if len(tokens) > 0:
            min_t = ops.min_time(tokens, seconds=False)
            if min_t > 0:
                tokens[0::3] = [t - min_t for t in tokens[0::3]]

        # Convert based on variant
        if self.variant == 'Interarrival':
            # Interarrival format: time tokens and note tokens
            mid = convert.interarrival_to_midi(tokens)
        else:
            # Arrival-Time or Vanilla: event triplets (time, duration, note)
            mid = convert.events_to_midi(tokens)

        return mid
