#!/usr/bin/env python3
"""
Convert MIDI to WAV.

Primary path: FluidSynth + MuseScore General soundfont (high quality).
Fallback:     Pure sine-wave synthesis (robotic, kept for environments
              where FluidSynth is unavailable).

Soundfont search order:
  1. SOUNDFONT env var
  2. SOUNDFONT_PATH constant below
  3. ~/.fluidsynth/default_sound_font.sf2
"""

import os
import subprocess
import sys
from pathlib import Path

SOUNDFONT_PATH = os.environ.get(
    "SOUNDFONT",
    "/home/rebcecca/orcd/scratch/rebcecca/soundfonts/MuseScore_General.sf3",
)

FLUIDSYNTH_BIN = os.environ.get(
    "FLUIDSYNTH_BIN",
    str(Path(sys.executable).parent / "fluidsynth"),
)

SAMPLE_RATE = 44100


def _find_soundfont():
    candidates = [
        SOUNDFONT_PATH,
        os.path.expanduser("~/.fluidsynth/default_sound_font.sf2"),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            return p
    return None


def _fluidsynth_available():
    return os.path.isfile(FLUIDSYNTH_BIN) and os.access(FLUIDSYNTH_BIN, os.X_OK)


def simple_synth(midi_path, wav_path, sr=SAMPLE_RATE):
    """
    Render a MIDI file to WAV.

    Tries FluidSynth first; falls back to sine-wave synthesis if FluidSynth
    or the soundfont is unavailable.
    """
    sf = _find_soundfont()
    if _fluidsynth_available() and sf:
        return _fluidsynth_synth(midi_path, wav_path, sf, sr)
    return _sine_synth(midi_path, wav_path, sr)


_FLUIDSYNTH_TIMEOUT = 120    # seconds; runaway renders produce multi-GB files
_WAV_SIZE_LIMIT     = 50 * 1024 * 1024   # 50 MB; ~5 min at 44100 Hz stereo 16-bit


def _fluidsynth_synth(midi_path, wav_path, sf_path, sr):
    """Render via FluidSynth — instrument-accurate, high quality."""
    cmd = [
        FLUIDSYNTH_BIN,
        "-ni",           # non-interactive, no MIDI router
        "-r", str(sr),   # sample rate
        "-g", "0.8",     # gain (avoid clipping on dense passages)
        "-F", str(wav_path),
        str(sf_path),
        str(midi_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_FLUIDSYNTH_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"    [FluidSynth timeout after {_FLUIDSYNTH_TIMEOUT}s] falling back to sine synth", file=sys.stderr)
        Path(wav_path).unlink(missing_ok=True)
        return _sine_synth(midi_path, wav_path, sr)

    if result.returncode != 0 or not Path(wav_path).exists():
        print(f"    [FluidSynth error] {result.stderr.strip()}", file=sys.stderr)
        return _sine_synth(midi_path, wav_path, sr)

    wav_size = Path(wav_path).stat().st_size
    if wav_size > _WAV_SIZE_LIMIT:
        print(
            f"    [FluidSynth] WAV too large ({wav_size // 1024 // 1024} MB > "
            f"{_WAV_SIZE_LIMIT // 1024 // 1024} MB limit); falling back to sine synth",
            file=sys.stderr,
        )
        Path(wav_path).unlink(missing_ok=True)
        return _sine_synth(midi_path, wav_path, sr)

    return True


def _sine_synth(midi_path, wav_path, sr):
    """Fallback: mix pure sine waves — zero dependencies, robotic sound."""
    try:
        import symusic
        import numpy as np
        from scipy.io import wavfile

        score = symusic.Score(midi_path)
        tpq = score.tpq
        duration_sec = score.end() / tpq / 2
        n_samples = int(sr * duration_sec)
        audio = np.zeros(n_samples)

        for track in score.tracks:
            for note in track.notes:
                freq = 440.0 * (2 ** ((note.pitch - 69) / 12.0))
                start_s = note.start / tpq / 2
                end_s = (note.start + note.duration) / tpq / 2
                start_idx = max(0, min(int(start_s * sr), n_samples - 1))
                end_idx = max(0, min(int(end_s * sr), n_samples))
                if start_idx >= end_idx:
                    continue
                t = np.arange(start_idx, end_idx) / sr
                amp = note.velocity / 127.0 * 0.3
                wave = amp * np.sin(2 * np.pi * freq * t)
                fade = min(100, (end_idx - start_idx) // 10)
                if fade > 1:
                    wave[:fade] *= np.linspace(0, 1, fade)
                    wave[-fade:] *= np.linspace(1, 0, fade)
                audio[start_idx:end_idx] += wave

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        wavfile.write(str(wav_path), sr, np.int16(audio * 32767))
        return True
    except Exception as e:
        print(f"    [sine synth error] {e}", file=sys.stderr)
        return False


def main():
    base_dir = Path("./inference_outputs")
    success = 0
    for model_dir in sorted(base_dir.glob("*_comparison")):
        print(f"\nProcessing {model_dir.name}...")
        for midi_file in sorted(model_dir.glob("sample_*_generated.mid")):
            wav_file = midi_file.with_suffix(".wav")
            print(f"  {midi_file.name} → {wav_file.name}...", end=" ", flush=True)
            print("✅" if simple_synth(str(midi_file), str(wav_file)) else "❌")
            success += 1
    print(f"\n✅ Converted {success} files!")


if __name__ == "__main__":
    main()
