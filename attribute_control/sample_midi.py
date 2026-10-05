"""
Save sweep samples as MIDI in the layout eval/music_quality.py expects, so
guided and unguided generations can be scored with the MIDI-level metrics
after the fact, without regenerating:

    <dir>/<sample_id>_prompt.mid       prompt tokens alone
    <dir>/<sample_id>_generated.mid    prompt + continuation   (or _ground_truth)

music_quality.py pairs the two by name and scores only the notes after the
prompt's last onset, plus coherence with the prompt. sample_id is also
written into the sweep tables, so per-sample scores join back to their
(attribute, strength, target) row.
"""

from pathlib import Path

from inference.mus.tokens_to_midi import tokens_to_midi


def save_sample_midi(out_dir, sample_id: str, prompt, continuation, tokenizer,
                     kind: str = "generated") -> bool:
    """Write <sample_id>_prompt.mid and <sample_id>_<kind>.mid. Returns False
    (and writes nothing) if decoding fails, so one undecodable sample never
    kills a multi-hour sweep."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        prompt_bytes = tokens_to_midi(list(prompt), tokenizer)
        full_bytes = tokens_to_midi(list(prompt) + list(continuation), tokenizer)
    except Exception as e:
        print(f"⚠️ MIDI decode failed for {sample_id}: {e}")
        return False
    (out_dir / f"{sample_id}_prompt.mid").write_bytes(prompt_bytes)
    (out_dir / f"{sample_id}_{kind}.mid").write_bytes(full_bytes)
    return True
