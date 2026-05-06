#!/usr/bin/env python3
"""Generate piano roll visualizations from token sequences."""

import sys
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer


def tokens_to_piano_roll(token_ids: List[int], tokenizer) -> np.ndarray:
    """
    Convert token IDs to piano roll representation.

    Uses tokenizer.decode() to convert tokens to MIDI, then extracts notes directly.
    This works with all miditok tokenizers (REMI, Octuple, CPWord, MuMIDI).

    Returns: 2D array (time_steps, 128 pitches) with velocity values
    """
    try:
        # Decode tokens to MIDI object (symusic.Score)
        midi_obj = tokenizer.decode(token_ids)
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        return np.zeros((100, 128))

    # Extract notes from the MIDI object
    notes = []  # (pitch, start_time, end_time, velocity)

    try:
        # symusic.Score object has tracks attribute
        if hasattr(midi_obj, 'tracks'):
            for track in midi_obj.tracks:
                # symusic tracks have notes attribute
                if hasattr(track, 'notes'):
                    for note in track.notes:
                        # symusic notes have pitch, start, end, velocity attributes
                        pitch = note.pitch
                        start = note.start  # in ticks
                        end = note.end      # in ticks
                        velocity = note.velocity
                        notes.append((pitch, start, end, velocity))
    except Exception as e:
        print(f"Error extracting notes from MIDI object: {e}")
        return np.zeros((100, 128))

    if not notes:
        return np.zeros((100, 128))

    # Get ticks per beat from the MIDI object
    tpb = getattr(midi_obj, 'ticks_per_quarter', 480)

    # Convert ticks to quarter notes (standardized time)
    notes_quarters = [(pitch, start / tpb, end / tpb, velocity)
                      for pitch, start, end, velocity in notes]

    # Create piano roll with 4 time steps per quarter note
    max_time = max(end for _, _, end, _ in notes_quarters)
    time_steps = int(max_time * 4) + 1
    piano_roll = np.zeros((time_steps, 128), dtype=np.uint8)

    # Fill in notes with velocity
    for pitch, start, end, velocity in notes_quarters:
        if 0 <= pitch < 128:
            start_step = int(start * 4)
            end_step = int(end * 4)
            end_step = min(end_step, time_steps)
            if start_step < time_steps:
                piano_roll[start_step:end_step, pitch] = np.maximum(
                    piano_roll[start_step:end_step, pitch],
                    velocity
                )

    return piano_roll


def plot_piano_roll(piano_roll: np.ndarray, title: str = "") -> np.ndarray:
    """Create matplotlib figure of piano roll and return as image array."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)

    # Plot piano roll (velocity-weighted)
    im = ax.imshow(
        piano_roll.T,
        aspect='auto',
        origin='lower',
        cmap='YlOrRd',
        interpolation='nearest',
        vmin=0,
        vmax=127
    )

    ax.set_xlabel("Time (×250ms)", fontsize=11)
    ax.set_ylabel("MIDI Pitch", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add octave lines
    for octave in range(11):
        c_pitch = 12 * octave
        if c_pitch < 128:
            ax.axhline(y=c_pitch, color='gray', alpha=0.2, linewidth=0.8, linestyle='--')

    ax.set_yticks([12 * i for i in range(11)])
    ax.set_yticklabels([f"C{i-1}" for i in range(11)])

    plt.colorbar(im, ax=ax, label="Velocity")
    plt.tight_layout()

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return image


def save_piano_roll_image(piano_roll: np.ndarray, output_path: Path, title: str = ""):
    """Save piano roll as PNG image."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)

    # Plot piano roll (velocity-weighted)
    im = ax.imshow(
        piano_roll.T,
        aspect='auto',
        origin='lower',
        cmap='YlOrRd',
        interpolation='nearest',
        vmin=0,
        vmax=127
    )

    ax.set_xlabel("Time (×250ms)", fontsize=11)
    ax.set_ylabel("MIDI Pitch", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add octave lines
    for octave in range(11):
        c_pitch = 12 * octave
        if c_pitch < 128:
            ax.axhline(y=c_pitch, color='gray', alpha=0.2, linewidth=0.8, linestyle='--')

    ax.set_yticks([12 * i for i in range(11)])
    ax.set_yticklabels([f"C{i-1}" for i in range(11)])

    plt.colorbar(im, ax=ax, label="Velocity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def generate_piano_rolls_for_run(run_dir: Path, tokenizer_type: str = "REMI"):
    """Generate piano rolls for all token files in a run."""
    print(f"Loading {tokenizer_type} tokenizer...")
    tokenizer, _, _ = load_tokenizer(tokenizer_type=tokenizer_type, dataset_name="giga_midi")

    for model_dir in ["gpt2", "llama"]:
        tokens_dir = run_dir / model_dir / "tokens"
        piano_rolls_dir = run_dir / model_dir / "piano_rolls"

        if not tokens_dir.exists():
            print(f"⚠️  {tokens_dir} not found, skipping")
            continue

        piano_rolls_dir.mkdir(parents=True, exist_ok=True)
        token_files = sorted(tokens_dir.glob("*.json"))

        print(f"\n{model_dir.upper()}:")
        for token_file in token_files:
            with open(token_file, 'r') as f:
                token_ids = json.load(f)

            try:
                piano_roll = tokens_to_piano_roll(token_ids, tokenizer)
                output_path = piano_rolls_dir / token_file.name.replace(".json", ".png")
                save_piano_roll_image(
                    piano_roll,
                    output_path,
                    title=f"{model_dir} - {token_file.stem}"
                )
                print(f"  ✅ {output_path.name}")
            except Exception as e:
                print(f"  ❌ {token_file.name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tokens_to_piano_roll.py <run_dir> [--tokenizer_type REMI]")
        print("Example: python tokens_to_piano_roll.py inference_outputs/run_20260505_145305")
        print("         python tokens_to_piano_roll.py inference_outputs/run_20260505_145305 --tokenizer_type Octuple")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)

    tokenizer_type = "REMI"
    if "--tokenizer_type" in sys.argv:
        idx = sys.argv.index("--tokenizer_type")
        if idx + 1 < len(sys.argv):
            tokenizer_type = sys.argv[idx + 1]

    generate_piano_rolls_for_run(run_dir, tokenizer_type)
    print("\n✅ Piano rolls generated!")
