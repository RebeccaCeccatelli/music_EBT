#!/usr/bin/env python3
"""Convert MIDI to WAV using symusic and scipy synthesis."""

import sys
from pathlib import Path

try:
    import symusic
    import numpy as np
    from scipy.io import wavfile
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    sys.exit(1)


def simple_synth(midi_path, wav_path, sr=22050):
    """Simple MIDI synthesizer using basic sine waves."""
    try:
        # Load MIDI
        score = symusic.Score(midi_path)

        # Duration in samples
        tpq = score.tpq
        duration_sec = score.end() / tpq / 2  # Approximate: assumes 120 BPM
        n_samples = int(sr * duration_sec)
        audio = np.zeros(n_samples)

        # Simple synthesis: mix sine waves for each note
        for track in score.tracks:
            for note in track.notes:
                # Frequency from MIDI note number (A4=440Hz)
                freq = 440 * (2 ** ((note.pitch - 69) / 12))

                # Start/end samples (assuming 120 BPM: quarter note = 0.5s)
                start_s = note.start / tpq / 2
                end_s = (note.start + note.duration) / tpq / 2
                start_idx = int(start_s * sr)
                end_idx = int(end_s * sr)

                # Clamp to audio bounds
                start_idx = max(0, min(start_idx, n_samples - 1))
                end_idx = max(0, min(end_idx, n_samples))

                if start_idx < end_idx:
                    # Generate sine wave with amplitude from velocity
                    t = np.arange(start_idx, end_idx) / sr
                    amplitude = note.velocity / 127.0 * 0.3  # Normalize and reduce
                    wave = amplitude * np.sin(2 * np.pi * freq * t)

                    # Add fade in/out to reduce clicks
                    fade_len = min(100, (end_idx - start_idx) // 10)
                    if fade_len > 1:
                        wave[:fade_len] *= np.linspace(0, 1, fade_len)
                        wave[-fade_len:] *= np.linspace(1, 0, fade_len)

                    audio[start_idx:end_idx] += wave

        # Normalize and convert to int16
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        audio = np.int16(audio * 32767)

        # Save WAV
        wavfile.write(wav_path, sr, audio)
        return True
    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        return False


def main():
    base_dir = Path("./inference_outputs")

    success = 0
    for model_dir in sorted(base_dir.glob("*_comparison")):
        print(f"\nProcessing {model_dir.name}...")

        for midi_file in sorted(model_dir.glob("sample_*_generated.mid")):
            wav_file = midi_file.with_suffix(".wav")
            print(f"  {midi_file.name} → {wav_file.name}...", end=" ", flush=True)

            if simple_synth(str(midi_file), str(wav_file)):
                print("✅")
                success += 1
            else:
                print("❌")

    print(f"\n✅ Converted {success} files!")
    print(f"Listen with: aplay inference_outputs/gpt2_comparison/sample_0_generated.wav")


if __name__ == "__main__":
    main()
