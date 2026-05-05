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

    Returns: 2D array (time_steps, 128 pitches)
    """
    # Get ID to token mapping
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    tokens = [id_to_token.get(t, f"UNK_{t}") for t in token_ids]

    # Parse tokens to extract note events
    notes = []  # (pitch, start_time, duration)
    current_time = 0
    current_pitch = None

    for token in tokens:
        if token.startswith("Pitch_"):
            # Extract pitch value
            pitch_str = token.split("_")[1]
            try:
                current_pitch = int(pitch_str)
            except:
                continue

        elif token.startswith("Duration_"):
            # Extract duration and add note
            duration_str = token.split("_")[1]
            try:
                duration = float(duration_str)
                if current_pitch is not None:
                    notes.append((current_pitch, current_time, duration))
                    current_time += duration
            except:
                continue

        elif token.startswith("Time_Shift_"):
            # Time shift
            shift_str = token.split("_")[-1]
            try:
                current_time += float(shift_str)
            except:
                continue

        elif token == "Bar_None":
            # Optional: could track bar boundaries
            pass

    if not notes:
        # Return empty piano roll if no notes
        return np.zeros((100, 128))

    # Create piano roll
    max_time = max(start + dur for pitch, start, dur in notes)
    time_steps = int(max_time * 4) + 1  # 4 time steps per beat
    piano_roll = np.zeros((time_steps, 128), dtype=np.float32)

    # Fill in notes
    for pitch, start, duration in notes:
        if 0 <= pitch < 128:
            start_step = int(start * 4)
            end_step = int((start + duration) * 4)
            end_step = min(end_step, time_steps)
            if start_step < time_steps:
                piano_roll[start_step:end_step, pitch] = 1.0

    return piano_roll


def plot_piano_roll(piano_roll: np.ndarray, title: str = "") -> np.ndarray:
    """Create matplotlib figure of piano roll and return as image array."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    # Plot piano roll
    ax.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='Blues', interpolation='nearest')

    ax.set_xlabel("Time")
    ax.set_ylabel("Pitch (MIDI note)")
    ax.set_title(title)

    # Add semitone lines for C notes
    for octave in range(11):
        c_pitch = 12 * octave
        if c_pitch < 128:
            ax.axhline(y=c_pitch, color='gray', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    # Convert figure to image array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return image


def save_piano_roll_image(piano_roll: np.ndarray, output_path: Path, title: str = ""):
    """Save piano roll as PNG image."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    # Plot piano roll
    im = ax.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='Blues', interpolation='nearest')

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title)

    # Add semitone lines for C notes
    for octave in range(11):
        c_pitch = 12 * octave
        if c_pitch < 128:
            ax.axhline(y=c_pitch, color='gray', alpha=0.2, linewidth=0.5)

    plt.colorbar(im, ax=ax, label="Note On")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def generate_piano_rolls_for_run(run_dir: Path):
    """Generate piano rolls for all token files in a run."""
    print(f"Loading tokenizer...")
    tokenizer, _, _ = load_tokenizer(tokenizer_type="REMI", dataset_name="giga_midi")

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
        print("Usage: python tokens_to_piano_roll.py <run_dir>")
        print("Example: python tokens_to_piano_roll.py inference_outputs/run_20260505_145305")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)

    generate_piano_rolls_for_run(run_dir)
    print("\n✅ Piano rolls generated!")
