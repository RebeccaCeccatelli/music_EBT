#!/usr/bin/env python3
"""Build the static (non-interactive) showcase page from saved sweep MIDI.

No generation happens here: every clip is an existing output of the final
guidance sweeps (same 16 REMI validation prompts for all models), so the
comparison is like-for-like.

    python demo/showcase/build_showcase.py candidates   # table to pick clips from
    python demo/showcase/build_showcase.py build        # render selection.json -> site/

`build` renders each selected MIDI to MP3 (FluidSynth + MuseScore General,
fixed gain, no loudness normalisation so velocity steering stays audible)
and writes site/data.json with the notes for the piano rolls.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import symusic

LOGS = Path.home() / "orcd/scratch/rebcecca/music_EBT_logs/attr_control"
HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
SELECTION = HERE / "selection.json"
SOUNDFONT = os.environ.get(
    "SOUNDFONT", str(Path.home() / "orcd/scratch/rebcecca/soundfonts/MuseScore_General.sf3"))
BIN = Path(sys.executable).parent

# Unguided continuations (r0..r2 = 3 baseline draws per prompt).
BASELINE_DIRS = {
    "ebt": LOGS / "listen_midi/25280946/baseline",
    "gpt2": LOGS / "ar_guidance/gpt2_tilt_20261005_153523_24940378/midi/baseline",
    "llama": LOGS / "ar_guidance/llama_tilt_20261005_150439_24940374/midi/baseline",
}
GROUND_TRUTH_DIR = LOGS / "listen_midi/25280946/ground_truth"

# Steered outputs at the thesis operating points
# (docs/thesis_findings/2026-10-09_guidance_ebt_vs_ar.md).
EBT_DIRS = {
    "velocity": LOGS / "listen_midi/25280946/guided",
    "duration": LOGS / "listen_midi/25280947/guided",
    "pitch_register": LOGS / "listen_midi/25280948/guided",
}
EBT_LAMBDA = {"velocity": "0.03", "duration": "0.02", "pitch_register": "0.02"}
PPLM_DIR = LOGS / "ar_guidance/llama_pplm_20261008_070421_25261493/midi"
PPLM_STRENGTH = {"velocity": "16", "duration": "16", "pitch_register": "32"}

PROMPTS = [1326, 3107, 4563, 4579, 7157, 8484, 9235, 9938,
           11732, 12623, 13268, 13781, 15617, 15922, 16537, 16753]


# --------------------------------------------------------------------- paths

def baseline_path(model, pid, draw=0):
    return BASELINE_DIRS[model] / f"p{pid}_r{draw}_generated.mid"


def ebt_path(attr, pid, sd, lam=None):
    """sd in {-2,-1,-0.5,0.5,1,2}; EBT files name the target delta in raw units."""
    lam = lam or EBT_LAMBDA[attr]
    hits = sorted(EBT_DIRS[attr].glob(f"p{pid}_{attr}_r3{lam}_d*_generated.mid"))
    deltas = sorted({float(h.name.split("_d")[-1].split("_")[0]) for h in hits})
    pos = [d for d in deltas if d > 0]  # 0.5, 1, 2 SD in raw units
    if len(pos) != 3:
        return None
    raw = dict(zip([0.5, 1, 2], pos))[abs(sd)] * (1 if sd > 0 else -1)
    for h in hits:
        if abs(float(h.name.split("_d")[-1].split("_")[0]) - raw) < 1e-6:
            return h
    return None


def pplm_path(attr, pid, sd):
    s = f"{sd:+g}"
    return PPLM_DIR / attr / f"p{pid}_{attr}_pplm{PPLM_STRENGTH[attr]}_d{s}_generated.mid"


# ------------------------------------------------------------------ analysis

def _notes(path):
    """All notes in seconds: (start, end, pitch, velocity, program, is_drum)."""
    score = symusic.Score(str(path)).to("second")
    out = []
    for t in score.tracks:
        for n in t.notes:
            out.append((round(n.time, 4), round(n.time + n.duration, 4), n.pitch,
                        n.velocity, t.program, t.is_drum))
    return sorted(out)


def _beat_lengths(path):
    score = symusic.Score(str(path))
    return [(round(n.time / score.tpq, 4), n.pitch, n.duration / score.tpq)
            for t in score.tracks for n in t.notes]


def prompt_of(path):
    path = str(path)
    for suffix in ("_generated.mid", "_ground_truth.mid"):
        if path.endswith(suffix):
            return Path(path[: -len(suffix)] + "_prompt.mid")
    return Path(path + ".noprompt")


def split_prompt(gen_path):
    """Notes of the continuation only (the saved generation includes the prompt)."""
    prompt_path = prompt_of(gen_path)
    notes = _notes(gen_path)
    if not prompt_path.exists():
        return notes, 0.0
    prompt = _notes(prompt_path)
    pset = {(n[0], n[2], n[4], n[5]) for n in prompt}
    cont = [n for n in notes if (n[0], n[2], n[4], n[5]) not in pset]
    # The continuation starts at its first new onset.
    prompt_end = min((n[0] for n in cont), default=max((n[0] for n in prompt), default=0.0))
    return cont, prompt_end


def metrics(gen_path):
    cont, prompt_end = split_prompt(gen_path)
    pitched = [n for n in cont if not n[5]]
    beats = _beat_lengths(gen_path)
    pp = prompt_of(gen_path)
    ponset = {(b[0], b[1]) for b in _beat_lengths(pp)} if pp.exists() else set()
    lens = [b[2] for b in beats if (b[0], b[1]) not in ponset]
    end = max((n[1] for n in cont), default=0.0)
    return {
        "n_notes": len(cont),
        "n_pitched": len(pitched),
        "seconds": round(end, 1),
        "prompt_end": round(prompt_end, 2),
        "velocity": round(sum(n[3] for n in cont) / len(cont), 1) if cont else None,
        "pitch": round(sum(n[2] for n in pitched) / len(pitched), 1) if pitched else None,
        "note_beats": round(sum(lens) / len(lens), 3) if lens else None,
    }


ATTR_KEY = {"velocity": "velocity", "duration": "note_beats", "pitch_register": "pitch"}


def cmd_candidates(_args):
    print("== Unguided (draw r0): continuation length / notes / pitched notes")
    for pid in PROMPTS:
        row = [f"p{pid:<6}"]
        for m in BASELINE_DIRS:
            p = baseline_path(m, pid)
            if p.exists():
                x = metrics(p)
                row.append(f"{m}:{x['seconds']:5.1f}s {x['n_notes']:3d}n {x['n_pitched']:3d}p")
        print("  ".join(row))

    for attr, key in ATTR_KEY.items():
        print(f"\n== {attr} ({key}); baseline = mean of EBT r0-r2; values at -2 / +2 SD")
        for pid in PROMPTS:
            base = [metrics(baseline_path("ebt", pid, r))[key] for r in range(3)
                    if baseline_path("ebt", pid, r).exists()]
            base = [b for b in base if b is not None]
            if not base:
                continue
            b = sum(base) / len(base)
            cells = []
            for name, fn in (("EBT", ebt_path), ("PPLM", pplm_path)):
                lo, hi = fn(attr, pid, -2), fn(attr, pid, 2)
                if lo and hi and Path(lo).exists() and Path(hi).exists():
                    vlo, vhi = metrics(lo)[key], metrics(hi)[key]
                    ok = (vlo is not None and vhi is not None and vlo < b < vhi)
                    cells.append(f"{name} {vlo} / {vhi} {'OK' if ok else '--'}")
                else:
                    cells.append(f"{name} missing")
            print(f"p{pid:<6} base {b:7.3f}   " + "   ".join(cells))


# ------------------------------------------------------------------- render

def render_mp3(midi, mp3):
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "x.wav"
        subprocess.run([str(BIN / "fluidsynth"), "-ni", "-q", "-r", "44100", "-g", "0.6",
                        "-F", str(wav), SOUNDFONT, str(midi)],
                       check=True, capture_output=True, timeout=120)
        subprocess.run([str(BIN / "lame"), "--quiet", "-V", "4", str(wav), str(mp3)],
                       check=True, timeout=120)


_QUALITY = {}


def dissonance(midi):
    """Harsh-dissonance score saved by the sweep's quality scoring, if any."""
    csv = Path(midi).parent.parent / "quality_scores.csv"
    if csv not in _QUALITY:
        rows = {}
        if csv.exists():
            lines = csv.read_text().splitlines()
            col = lines[0].split(",").index("sharp_dissonance")
            for line in lines[1:]:
                f = line.split(",")
                rows[f[0]] = round(float(f[col]), 3)
        _QUALITY[csv] = rows
    return _QUALITY[csv].get(Path(midi).name.replace("_generated.mid", ""))


def clip_entry(midi, clip_id, sources, force=False):
    midi = Path(midi)
    mp3 = SITE / "audio" / f"{clip_id}.mp3"
    # Re-render when the clip id now points at a different MIDI.
    src = str(midi.relative_to(LOGS)) if midi.is_relative_to(LOGS) else midi.name
    if force or not mp3.exists() or sources.get(clip_id) != src:
        render_mp3(midi, mp3)
    sources[clip_id] = src
    notes = _notes(midi)
    m = metrics(midi)
    m["dissonance"] = dissonance(midi)
    return {
        "audio": f"audio/{clip_id}.mp3",
        "prompt_end": m["prompt_end"],
        "notes": [[n[0], round(n[1] - n[0], 4), n[2], n[3], int(n[5])] for n in notes],
        "metrics": m,
    }


def resolve(spec):
    kind = spec["kind"]
    pid = spec["prompt"]
    if kind == "baseline":
        return baseline_path(spec["model"], pid, spec.get("draw", 0))
    if kind == "ground_truth":
        return GROUND_TRUTH_DIR / f"p{pid}_ground_truth.mid"
    if kind == "ebt":
        return ebt_path(spec["attribute"], pid, spec["sd"], spec.get("lambda"))
    if kind == "pplm":
        return pplm_path(spec["attribute"], pid, spec["sd"])
    raise ValueError(kind)


def cmd_build(args):
    sel = json.loads(SELECTION.read_text())
    (SITE / "audio").mkdir(parents=True, exist_ok=True)
    used = set()
    manifest = SITE / "audio" / "sources.json"
    sources = json.loads(manifest.read_text()) if manifest.exists() else {}

    def build_clips(clips):
        out = []
        for c in clips:
            path = resolve(c)
            if path is None or not Path(path).exists():
                sys.exit(f"missing clip: {c}")
            cid = c["id"]
            used.add(f"{cid}.mp3")
            print(f"  {cid:<40} {Path(path).name}")
            out.append({**{k: v for k, v in c.items() if k not in ("kind",)},
                        **clip_entry(path, cid, sources, args.force)})
        return out

    data = {"meta": sel["meta"], "sections": []}
    for sec in sel["sections"]:
        s = {k: v for k, v in sec.items() if k != "examples"}
        s["examples"] = []
        for ex in sec["examples"]:
            e = {k: v for k, v in ex.items() if k != "clips"}
            e["clips"] = build_clips(ex["clips"])
            s["examples"].append(e)
        data["sections"].append(s)

    for f in (SITE / "audio").glob("*.mp3"):
        if f.name not in used:
            f.unlink()
    manifest.write_text(json.dumps({k: v for k, v in sources.items() if f"{k}.mp3" in used},
                                   indent=1, sort_keys=True))
    (SITE / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    total = sum(f.stat().st_size for f in (SITE / "audio").glob("*.mp3"))
    print(f"wrote {SITE/'data.json'}; {len(used)} clips, {total/1e6:.1f} MB audio")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("candidates")
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true", help="re-render existing MP3s")
    args = ap.parse_args()
    {"candidates": cmd_candidates, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
