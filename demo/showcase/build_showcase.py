#!/usr/bin/env python3
"""Build the static (non-interactive) showcase page from saved sweep MIDI.

No generation happens here: every clip is an existing output of the final
guidance sweeps (per tokenizer, the same 16 validation prompts for every
model and method), so comparisons are like-for-like.

    python demo/showcase/build_showcase.py candidates [--tok ant]  # table to pick prompts from
    python demo/showcase/build_showcase.py build                   # selection.json -> site/

selection.json is a tree: section -> tabs (tokenizer) -> tabs (attribute or
combination) -> blocks. Block types, expanded here into clips:

  clips      explicit list of clip specs, shown as a grid
  steer      rows of methods x (down, unguided, up) at one target size
  intensity  EBT grid: rows = lambda, columns = target sizes, plus unguided
  note       a text card (e.g. "not available for this tokenizer")

Clip spec: {"tok", "kind": ground_truth|unguided|guided, "model", "method",
"attribute", "prompt", "sd", "strength"}. A spec (or a whole block/row) with
"placeholder": true plays PLACEHOLDER and is tagged as such on the page.

`build` renders each MIDI to MP3 (FluidSynth + MuseScore General, fixed
gain, no loudness normalisation so velocity steering stays audible) and
writes site/data.json with notes for the piano rolls.
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

AR = LOGS / "ar_guidance"
LM = LOGS / "listen_midi"

# Per tokenizer: where each run's MIDI lives. EBT dirs are lists because
# Anticipation's coarse and fine lambda grids are separate jobs.
RUNS = {
    "remi": {
        "ground_truth": LM / "25280946/ground_truth",
        "unguided": {
            "ebt": LM / "25280946/baseline",
            "gpt2": AR / "gpt2_tilt_20261005_153523_24940378/midi/baseline",
            "llama": AR / "llama_tilt_20261005_150439_24940374/midi/baseline",
        },
        "ebt": {
            "velocity": [LM / "25280946/guided"],
            "duration": [LM / "25280947/guided"],
            "pitch_register": [LM / "25280948/guided"],
        },
        "ar": {  # (method, model) -> run dir
            ("tilt", "gpt2"): AR / "gpt2_tilt_20261005_153523_24940378/midi",
            ("tilt", "llama"): AR / "llama_tilt_20261005_150439_24940374/midi",
            ("best_of_n", "gpt2"): AR / "gpt2_best_of_n_20261005_154305_24940380/midi",
            ("best_of_n", "llama"): AR / "llama_best_of_n_20261005_153108_24940376/midi",
            ("pplm", "llama"): AR / "llama_pplm_20261008_070421_25261493/midi",
        },
    },
    "ant": {  # control-free Anticipation prompts (commit 1802c3f)
        "ground_truth": LM / "25296307/ground_truth",
        "unguided": {
            "ebt": LM / "25296307/baseline",
            "gpt2": AR / "gpt2_best_of_n_20261008_154914_25296309/midi/baseline",
            "llama": AR / "llama_best_of_n_20261008_155455_25296310/midi/baseline",
        },
        "ebt": {
            "duration": [LM / "25296307/guided", LM / "25296410/guided"],
            "pitch_register": [LM / "25296308/guided", LM / "25296411/guided"],
        },
        "ar": {
            ("best_of_n", "gpt2"): AR / "gpt2_best_of_n_20261008_154914_25296309/midi",
            ("best_of_n", "llama"): AR / "llama_best_of_n_20261008_155455_25296310/midi",
        },
    },
}

# Operating points (docs/thesis_findings/2026-10-09_guidance_ebt_vs_ar.md).
OPERATING = {
    "remi": {"velocity": "0.03", "duration": "0.02", "pitch_register": "0.02"},
    "ant": {"duration": "0.005", "pitch_register": "0.01"},
}

# Played wherever a slot has no real output yet.
PLACEHOLDER = {"tok": "remi", "kind": "unguided", "model": "ebt", "prompt": 9235}

METHOD_LABEL = {"tilt": "Tilt", "best_of_n": "Best-of-N", "pplm": "PPLM", "ebt": "EBT"}
MODEL_LABEL = {"ebt": "EBT", "gpt2": "GPT-2", "llama": "Llama"}


# --------------------------------------------------------------------- paths

def _delta(name):
    return float(name.split("_d")[-1].split("_")[0])


def ebt_path(tok, attr, pid, sd, lam):
    """EBT files name the target delta in raw units; map sd (±0.5/1/2) onto them."""
    for d in RUNS[tok]["ebt"].get(attr, []):
        hits = sorted(d.glob(f"p{pid}_{attr}_r3{lam}_d*_generated.mid"))
        pos = sorted({_delta(h.name) for h in hits if _delta(h.name) > 0})
        if len(pos) != 3:
            continue
        raw = dict(zip([0.5, 1, 2], pos))[abs(sd)] * (1 if sd > 0 else -1)
        for h in hits:
            if abs(_delta(h.name) - raw) < 1e-6:
                return h
    return None


def resolve(spec):
    """Clip spec -> MIDI path (None if that output doesn't exist)."""
    if spec.get("placeholder"):
        spec = PLACEHOLDER
    runs, pid = RUNS[spec["tok"]], spec["prompt"]
    kind = spec["kind"]
    if kind == "ground_truth":
        return runs["ground_truth"] / f"p{pid}_ground_truth.mid"
    if kind == "unguided":
        return runs["unguided"][spec["model"]] / f"p{pid}_r{spec.get('draw', 0)}_generated.mid"
    attr, sd = spec["attribute"], spec["sd"]
    if spec["method"] == "ebt":
        return ebt_path(spec["tok"], attr, pid, sd, spec["strength"])
    run = runs["ar"].get((spec["method"], spec["model"]))
    if run is None:
        return None
    return run / attr / f"p{pid}_{attr}_{spec['method']}{spec['strength']}_d{sd:+g}_generated.mid"


def clip_id(spec):
    if spec.get("placeholder"):
        spec = PLACEHOLDER
    parts = [spec["tok"], spec["kind"], spec.get("method"), spec.get("model"),
             spec.get("attribute"), str(spec["prompt"]), spec.get("strength"),
             None if spec.get("sd") is None else f"{spec['sd']:+g}",
             None if not spec.get("draw") else f"r{spec['draw']}"]
    return "_".join(p for p in parts if p).replace(".", "p").replace("+", "u").replace("-", "d")


# ------------------------------------------------------------------ analysis

def _notes(path):
    """All notes in seconds: (start, end, pitch, velocity, program, is_drum)."""
    score = symusic.Score(str(path)).to("second")
    return sorted((round(n.time, 4), round(n.time + n.duration, 4), n.pitch, n.velocity,
                   t.program, t.is_drum) for t in score.tracks for n in t.notes)


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


def metrics(gen_path):
    cont, prompt_end = split_prompt(gen_path)
    pitched = [n for n in cont if not n[5]]
    pp = prompt_of(gen_path)
    ponset = {(b[0], b[1]) for b in _beat_lengths(pp)} if pp.exists() else set()
    lens = [b[2] for b in _beat_lengths(gen_path) if (b[0], b[1]) not in ponset]
    end = max((n[1] for n in cont), default=0.0)
    return {
        "n_notes": len(cont),
        "n_pitched": len(pitched),
        "seconds": round(end, 1),
        "prompt_end": round(prompt_end, 2),
        "velocity": round(sum(n[3] for n in cont) / len(cont), 1) if cont else None,
        "pitch": round(sum(n[2] for n in pitched) / len(pitched), 1) if pitched else None,
        "note_beats": round(sum(lens) / len(lens), 3) if lens else None,
        "dissonance": dissonance(gen_path),
    }


ATTR_KEY = {"velocity": "velocity", "duration": "note_beats", "pitch_register": "pitch"}

# General MIDI program families, for naming prompts by their instruments
# (the sweep prompts have no song titles in their metadata).
GM_FAMILY = ["piano", "keys", "organ", "guitar", "bass", "strings", "strings", "brass",
             "sax", "flute", "synth lead", "synth pad", "synth", "ethnic", "percussion", "effects"]
GM_SPECIAL = {0: "piano", 1: "piano", 2: "electric piano", 3: "piano", 4: "electric piano",
              5: "electric piano", 6: "harpsichord", 11: "vibraphone",
              40: "violin", 42: "cello", 46: "harp", 48: "strings", 52: "choir", 56: "trumpet",
              57: "trombone", 60: "horn", 65: "sax", 66: "sax", 68: "oboe", 71: "clarinet",
              73: "flute", 70: "bassoon"}


def prompt_name(tok, pid):
    """'Piano, bass & drums': instruments by note count in the original excerpt."""
    path = RUNS[tok]["ground_truth"] / f"p{pid}_ground_truth.mid"
    if not path.exists():
        return f"prompt {pid}"
    counts = {}
    for t in symusic.Score(str(path)).tracks:
        name = "drums" if t.is_drum else GM_SPECIAL.get(t.program, GM_FAMILY[t.program // 8])
        counts[name] = counts.get(name, 0) + len(t.notes)
    total = sum(counts.values())
    names = [n for n, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c >= 0.08 * total][:4]
    if not names:
        return f"prompt {pid}"
    if len(names) == 1:
        text = f"solo {names[0]}" if names[0] != "drums" else "drums only"
    else:
        text = ", ".join(names[:-1]) + " & " + names[-1]
    return text[0].upper() + text[1:]


def cmd_candidates(args):
    tok = args.tok
    runs = RUNS[tok]
    pids = sorted(int(p.name[1:].split("_")[0]) for p in runs["unguided"]["ebt"].glob("*_r0_generated.mid"))
    print(f"== {tok} unguided (r0): continuation seconds / notes / pitched notes")
    for pid in pids:
        row = [f"p{pid:<8}"]
        for m in runs["unguided"]:
            p = resolve({"tok": tok, "kind": "unguided", "model": m, "prompt": pid})
            if p.exists():
                x = metrics(p)
                row.append(f"{m}:{x['seconds']:5.1f}s {x['n_notes']:3d}n {x['n_pitched']:3d}p")
        print("  ".join(row))
    for attr in runs["ebt"]:
        key, lam = ATTR_KEY[attr], OPERATING[tok][attr]
        print(f"\n== {attr} ({key}); EBT at operating lambda {lam}: unguided | -2 / +2 SD")
        for pid in pids:
            u = resolve({"tok": tok, "kind": "unguided", "model": "ebt", "prompt": pid})
            lo, hi = ebt_path(tok, attr, pid, -2, lam), ebt_path(tok, attr, pid, 2, lam)
            if not (u.exists() and lo and hi):
                continue
            b, vlo, vhi = metrics(u)[key], metrics(lo)[key], metrics(hi)[key]
            ok = None not in (b, vlo, vhi) and vlo < b < vhi
            print(f"p{pid:<8} {b} | {vlo} / {vhi} {'OK' if ok else '--'}")


# ---------------------------------------------------------------- expansion

def _with(base, **kw):
    out = dict(base)
    out.update({k: v for k, v in kw.items() if v is not None})
    return out


def expand_block(b):
    """Block spec -> block with a flat list of clip specs (row/col/label set)."""
    t = b["type"]
    if t in ("clips", "note"):
        return b
    ctx = {"tok": b["tok"], "prompt": b["prompt"], "attribute": b.get("attribute")}
    clips = []
    if t == "steer":
        sd = b.get("sd", 2)
        for row in b["rows"]:
            ph = row.get("placeholder") or b.get("placeholder")
            un = {"kind": "unguided", "model": row["model"]}
            g = {"kind": "guided", "method": row["method"], "model": row["model"],
                 "strength": row["strength"]}
            for col, spec in (("down", _with(g, sd=-sd)), ("unguided", un), ("up", _with(g, sd=sd))):
                clips.append({**ctx, **spec, "row": row["label"], "row_sub": row.get("sub", ""),
                              "col": col, "placeholder": ph})
    elif t == "intensity":
        ph = b.get("placeholder")
        clips.append({**ctx, "kind": "unguided", "model": "ebt", "row": "unguided",
                      "col": "unguided", "placeholder": ph})
        for lam in b["lambdas"]:
            for sd in b["sds"]:
                clips.append({**ctx, "kind": "guided", "method": "ebt", "model": "ebt",
                              "strength": lam, "sd": sd, "row": lam, "col": f"{sd:+g}",
                              "operating": lam == OPERATING.get(b["tok"], {}).get(b.get("attribute")),
                              "placeholder": ph})
    else:
        raise ValueError(t)
    return {**{k: v for k, v in b.items() if k not in ("rows",)}, "clips": clips}


# ------------------------------------------------------------------- render

def render_mp3(midi, mp3):
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "x.wav"
        subprocess.run([str(BIN / "fluidsynth"), "-ni", "-q", "-r", "44100", "-g", "0.6",
                        "-F", str(wav), SOUNDFONT, str(midi)],
                       check=True, capture_output=True, timeout=120)
        subprocess.run([str(BIN / "lame"), "--quiet", "-V", "4", str(wav), str(mp3)],
                       check=True, timeout=120)


class Builder:
    def __init__(self, force):
        self.force = force
        self.used = set()
        self.cache = {}
        self.manifest = SITE / "audio" / "sources.json"
        self.sources = json.loads(self.manifest.read_text()) if self.manifest.exists() else {}
        self.missing = []

    def clip(self, spec):
        path = resolve(spec)
        if path is None or not Path(path).exists():
            # Not produced (yet): fall back to the placeholder so the layout still renders.
            self.missing.append({k: spec.get(k) for k in
                                 ("tok", "kind", "method", "model", "attribute", "prompt", "strength", "sd")})
            spec = {**spec, "placeholder": True}
            path = resolve(spec)
        cid = clip_id(spec)
        if cid not in self.cache:
            midi = Path(path)
            mp3 = SITE / "audio" / f"{cid}.mp3"
            src = str(midi.relative_to(LOGS)) if midi.is_relative_to(LOGS) else midi.name
            if self.force or not mp3.exists() or self.sources.get(cid) != src:
                print(f"  render {cid}")
                render_mp3(midi, mp3)
            self.sources[cid] = src
            m = metrics(midi)
            self.cache[cid] = {
                "audio": f"audio/{cid}.mp3",
                "prompt_end": m["prompt_end"],
                "notes": [[n[0], round(n[1] - n[0], 4), n[2], n[3], int(n[5])] for n in _notes(midi)],
                "metrics": m,
            }
        self.used.add(f"{cid}.mp3")
        keep = {k: v for k, v in spec.items()
                if k in ("label", "tag", "row", "row_sub", "col", "operating", "placeholder", "sd", "strength")
                and v not in (None, False)}
        return {"id": cid, **keep, **self.cache[cid]}

    def node(self, n):
        out = {k: v for k, v in n.items() if k not in ("tabs", "blocks")}
        if "blocks" in n:
            out["blocks"] = []
            for b in n["blocks"]:
                b = expand_block(b)
                if b.get("prompt") is not None and b.get("tok") and not b.get("placeholder"):
                    b = {"prompt_name": prompt_name(b["tok"], b["prompt"]), **b}
                if "clips" in b:
                    b = {**b, "clips": [self.clip({"tok": b.get("tok"), "prompt": b.get("prompt"),
                                                   "placeholder": b.get("placeholder"), **c})
                                        for c in b["clips"]]}
                out["blocks"].append(b)
        if "tabs" in n:
            out["tabs"] = [self.node(t) for t in n["tabs"]]
        return out


def cmd_build(args):
    sel = json.loads(SELECTION.read_text())
    (SITE / "audio").mkdir(parents=True, exist_ok=True)
    b = Builder(args.force)
    data = {"meta": sel["meta"], "sections": [b.node(s) for s in sel["sections"]]}

    for f in (SITE / "audio").glob("*.mp3"):
        if f.name not in b.used:
            f.unlink()
    b.manifest.write_text(json.dumps({k: v for k, v in b.sources.items() if f"{k}.mp3" in b.used},
                                     indent=1, sort_keys=True))
    (SITE / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    total = sum(f.stat().st_size for f in (SITE / "audio").glob("*.mp3"))
    print(f"wrote {SITE/'data.json'}; {len(b.used)} audio files, {total/1e6:.1f} MB")
    if b.missing:
        print(f"{len(b.missing)} slots have no output yet and use the placeholder, e.g.:")
        for m in b.missing[:5]:
            print("  ", m)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--tok", choices=list(RUNS), default="remi")
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true", help="re-render existing MP3s")
    args = ap.parse_args()
    {"candidates": cmd_candidates, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
