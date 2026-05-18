#!/usr/bin/env python3
"""Convert MIDI files in a run to WAV files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from convert_midi_simple import simple_synth


def convert_run(run_dir):
    """Convert all MIDI files in a run to WAV."""
    run_path = Path(run_dir)

    for model_dir in ["gpt2", "llama"]:
        midi_dir = run_path / model_dir / "midi"
        wav_dir = run_path / model_dir / "wav"

        if not midi_dir.exists():
            print(f"⚠️  {midi_dir} not found, skipping")
            continue

        wav_dir.mkdir(parents=True, exist_ok=True)
        midi_files = sorted(midi_dir.glob("*.mid"))

        print(f"\n{model_dir.upper()}:")
        for midi_file in midi_files:
            wav_file = wav_dir / midi_file.name.replace(".mid", ".wav")
            try:
                simple_synth(str(midi_file), str(wav_file))
                print(f"  ✅ {wav_file.name}")
            except Exception as e:
                print(f"  ❌ {wav_file.name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python midi_to_wav.py <run_dir>")
        print("Example: python midi_to_wav.py inference_outputs/run_20260505_141217")
        sys.exit(1)

    convert_run(sys.argv[1])
