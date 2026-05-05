#!/usr/bin/env python3
"""Generate piano roll visualizations from MIDI files."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mido


def midi_to_piano_roll(midi_path: Path, ticks_per_beat: int = 480, tempo: int = 500000) -> np.ndarray:
    """
    Convert MIDI file to piano roll representation.

    Args:
        midi_path: Path to MIDI file
        ticks_per_beat: MIDI ticks per beat (from file)
        tempo: Default tempo in microseconds per beat

    Returns: 2D array (time_steps, 128 pitches) with velocity values
    """
    try:
        midi = mido.MidiFile(str(midi_path))

        # Collect all note events
        notes = []  # (pitch, start_time, end_time, velocity)
        note_ons = {}  # (channel, pitch) -> (start_time, velocity)

        current_time = 0
        tpb = midi.ticks_per_beat if midi.ticks_per_beat is not None else 480
        ticks_per_second = 1e6 / tempo / tpb

        for track in midi.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        note_ons[(msg.channel, msg.note)] = (current_time, msg.velocity)
                    else:
                        # note_on with velocity 0 = note_off
                        key = (msg.channel, msg.note)
                        if key in note_ons:
                            start_time, velocity = note_ons[key]
                            end_time = current_time
                            notes.append((msg.note, start_time, end_time, velocity))
                            del note_ons[key]

                elif msg.type == 'note_off':
                    key = (msg.channel, msg.note)
                    if key in note_ons:
                        start_time, velocity = note_ons[key]
                        end_time = current_time
                        notes.append((msg.note, start_time, end_time, velocity))
                        del note_ons[key]

                elif msg.type == 'set_tempo':
                    tempo = msg.tempo
                    ticks_per_second = 1e6 / tempo / tpb

        if len(notes) == 0:
            return np.zeros((100, 128))

        # Convert to seconds
        notes_sec = [(pitch, start / ticks_per_second, end / ticks_per_second, velocity)
                     for pitch, start, end, velocity in notes]

        # Create piano roll
        max_time = max(end for _, _, end, _ in notes_sec)
        time_steps = int(max_time * 100) + 10  # 100 samples per second
        piano_roll = np.zeros((time_steps, 128), dtype=np.uint8)

        # Fill in notes with velocity
        for pitch, start, end, velocity in notes_sec:
            if 0 <= pitch < 128:
                start_step = int(start * 100)
                end_step = int(end * 100)
                end_step = min(end_step, time_steps)
                if start_step < time_steps:
                    piano_roll[start_step:end_step, pitch] = np.maximum(
                        piano_roll[start_step:end_step, pitch],
                        velocity
                    )

        return piano_roll

    except Exception as e:
        print(f"    Error loading MIDI: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((100, 128))


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

    ax.set_xlabel("Time (×10ms)", fontsize=11)
    ax.set_ylabel("MIDI Pitch", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add octave lines
    for octave in range(11):
        c_pitch = 12 * octave
        if c_pitch < 128:
            ax.axhline(y=c_pitch, color='gray', alpha=0.2, linewidth=0.8, linestyle='--')

    # Add semitone labels for reference
    ax.set_yticks([12 * i for i in range(11)])
    ax.set_yticklabels([f"C{i-1}" for i in range(11)])

    plt.colorbar(im, ax=ax, label="Velocity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def generate_piano_rolls_for_run(run_dir: Path):
    """Generate piano rolls from MIDI files in a run."""
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
        print("Usage: python midi_to_piano_roll.py <run_dir>")
        print("Example: python midi_to_piano_roll.py inference_outputs/run_20260505_145305")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)

    generate_piano_rolls_for_run(run_dir)
    print("\n✅ Piano rolls generated!")
