#!/usr/bin/env python3
"""Generate piano roll visualizations from MIDI files."""

import sys
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def midi_to_piano_roll(midi_path: Path) -> np.ndarray:
    """
    Convert a MIDI file to a piano roll using symusic.
    Returns: 2D array (time_steps, 128 pitches)
    """
    import symusic
    score = symusic.Score(str(midi_path))

    tpb = score.ticks_per_quarter
    notes = []
    for track in score.tracks:
        for note in track.notes:
            notes.append((note.pitch, note.start / tpb, note.end / tpb))

    if not notes:
        return np.zeros((100, 128))

    max_time = max(end for _, _, end in notes)
    time_steps = int(max_time * 4) + 1
    piano_roll = np.zeros((time_steps, 128), dtype=np.float32)

    for pitch, start, end in notes:
        if 0 <= pitch < 128:
            start_step = int(start * 4)
            end_step = min(int(end * 4), time_steps)
            if start_step < time_steps:
                piano_roll[start_step:end_step, pitch] = 1.0

    return piano_roll


def plot_piano_roll(piano_roll: np.ndarray, title: str = "") -> np.ndarray:
    """Create matplotlib figure of piano roll and return as image array."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)

    ax.imshow(
        piano_roll.T,
        aspect='auto',
        origin='lower',
        cmap='Blues',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )

    ax.set_xlabel("Time (×250ms)", fontsize=11)
    ax.set_ylabel("MIDI Pitch", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')

    for octave in range(11):
        c_pitch = 12 * octave
        if c_pitch < 128:
            ax.axhline(y=c_pitch, color='gray', alpha=0.2, linewidth=0.8, linestyle='--')

    ax.set_yticks([12 * i for i in range(11)])
    ax.set_yticklabels([f"C{i-1}" for i in range(11)])

    plt.tight_layout()

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return image


def save_piano_roll_image(piano_roll: np.ndarray, output_path: Path, title: str = ""):
    """Save piano roll as PNG image."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)

    im = ax.imshow(
        piano_roll.T,
        aspect='auto',
        origin='lower',
        cmap='Blues',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )

    ax.set_xlabel("Time (×250ms)", fontsize=11)
    ax.set_ylabel("MIDI Pitch", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')

    for octave in range(11):
        c_pitch = 12 * octave
        if c_pitch < 128:
            ax.axhline(y=c_pitch, color='gray', alpha=0.2, linewidth=0.8, linestyle='--')

    ax.set_yticks([12 * i for i in range(11)])
    ax.set_yticklabels([f"C{i-1}" for i in range(11)])

    plt.colorbar(im, ax=ax, label="Note On")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def generate_piano_rolls_for_run(run_dir: Path, tokenizer_type: str = "REMI"):
    """Generate piano rolls for all MIDI files in a run."""
    for model_dir in ["gpt2", "llama"]:
        midi_dir = run_dir / model_dir / "midi"
        piano_rolls_dir = run_dir / model_dir / "piano_rolls"

        if not midi_dir.exists():
            print(f"⚠️  {midi_dir} not found, skipping")
            continue

        piano_rolls_dir.mkdir(parents=True, exist_ok=True)
        midi_files = sorted(midi_dir.glob("*.mid"))

        print(f"\n{model_dir.upper()}:")
        for midi_file in midi_files:
            try:
                piano_roll = midi_to_piano_roll(midi_file)
                output_path = piano_rolls_dir / midi_file.name.replace(".mid", ".png")
                save_piano_roll_image(
                    piano_roll,
                    output_path,
                    title=f"{model_dir} - {midi_file.stem}"
                )
                print(f"  ✅ {output_path.name}")
            except Exception as e:
                print(f"  ❌ {midi_file.name}: {e}")


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
