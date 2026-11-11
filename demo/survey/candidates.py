#!/usr/bin/env python3
"""Render every candidate survey clip so they can be picked by ear.

    python demo/survey/candidates.py render [--workers 4]   # MIDI -> small mono MP3 + manifest
    python demo/survey/candidates.py pack --out DIR          # base64 chunks + index.json for the picker page

Candidates (16 test prompts per tokenizer, EBT-recipe baselines, operating points
chosen on the held-out tune prompts, music_quality/sweeps/tune_{remi,ant_ar}.json):
  Part 1   original + unguided first draw (r0) of EBT, Llama, GPT-2
  Part 2   per attribute: each system's own unguided reference, steered -2 SD, +2 SD
  Catch    EBT pitch register far past its operating point (+2 SD)
  Pool     unguided_more: every other unguided draw of the same three models (all sweeps, test + tune
           prompts); unguided_songs: the 100 music-quality songs per tokenizer
Output: ~/orcd/scratch/rebcecca/music_EBT_logs/survey/candidates/{mp3/,manifest.json}
"""
import argparse
import base64
import json
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "showcase"))
import build_showcase as bs  # noqa: E402  (EBT path resolver, renderer helpers)

LM, AR = bs.LM, bs.AR
OUT = bs.LOGS.parent / "survey" / "candidates"
SD = 2

UNGUIDED = {
    "remi": {"original": LM / "25280946/ground_truth",
             "ebt": LM / "25280946/baseline",
             "llama": AR / "llama-ebtrecipe_tilt_20261019/midi/baseline",
             "gpt2": AR / "gpt2-ebtrecipe_tilt_20261019/midi/baseline"},
    "ant": {"original": LM / "25296307/ground_truth",
            "ebt": LM / "25296307/baseline",
            "llama": AR / "llama-ebtrecipe_best_of_n_ant_20261019/midi/baseline",
            "gpt2": AR / "gpt2-ebtrecipe_best_of_n_ant_20261019/midi/baseline"},
}
# (system id, label, run dir or None for EBT, method, {attribute: strength})
SYSTEMS = {
    "remi": [
        ("ebt", "Tonic", None, "ebt", {"velocity": "0.05", "duration": "0.055", "pitch_register": "0.01"}),
        ("ebt_l002", "Tonic λ 0.02 (sensitivity)", None, "ebt", {"pitch_register": "0.02"}),
        ("llama_pplm", "Llama PPLM", AR / "llama-ebtrecipe_pplm_20261104/midi", "pplm",
         {"velocity": "32", "duration": "32", "pitch_register": "16"}),
        ("llama_tilt", "Llama tilt", AR / "llama-ebtrecipe_tilt_20261019/midi", "tilt",
         {"velocity": "1", "duration": "1", "pitch_register": "0.75"}),
        ("llama_bon", "Llama best-of-N", AR / "llama-ebtrecipe_best_of_n_20261019/midi", "best_of_n",
         {"velocity": "16", "duration": "8", "pitch_register": "16"}),
    ],
    "ant": [
        ("ebt", "Tonic", None, "ebt", {"duration": "0.005", "pitch_register": "0.012"}),
        ("llama_pplm", "Llama PPLM", AR / "llama-ebtrecipe_pplm_ant_20261105/midi", "pplm",
         {"duration": "16", "pitch_register": "32"}),
        ("llama_bon", "Llama best-of-N", AR / "llama-ebtrecipe_best_of_n_ant_20261019/midi", "best_of_n",
         {"duration": "8", "pitch_register": "8"}),
    ],
}
ATTRIBUTES = {"remi": ["velocity", "duration", "pitch_register"], "ant": ["duration", "pitch_register"]}

# Every other unguided draw of the three comparably trained models (EBT and the EBT-recipe baselines),
# for a larger pool to pick from. Each sweep made its own 3 unguided draws per prompt (checked: no
# duplicates across sweeps), on the 16 test prompts or on the 16 held-out tune prompts.
# (system, label, baseline dir); the first entry per system also gives the original continuation.
POOL_SWEEPS = {
    "remi": {
        "test": [("ebt", "velocity sweep", LM / "25280946/baseline"), ("ebt", "duration sweep", LM / "25280947/baseline"),
                 ("ebt", "pitch sweep", LM / "25280948/baseline"),
                 ("llama", "tilt sweep", AR / "llama-ebtrecipe_tilt_20261019/midi/baseline"),
                 ("llama", "best-of-N sweep", AR / "llama-ebtrecipe_best_of_n_20261019/midi/baseline"),
                 ("llama", "PPLM sweep", AR / "llama-ebtrecipe_pplm_20261104/midi/baseline"),
                 ("gpt2", "tilt sweep", AR / "gpt2-ebtrecipe_tilt_20261019/midi/baseline"),
                 ("gpt2", "best-of-N sweep", AR / "gpt2-ebtrecipe_best_of_n_20261019/midi/baseline")],
        "tune": [("ebt", "velocity sweep", LM / "tune-ebt-remi-velocity/baseline"),
                 ("ebt", "duration sweep", LM / "tune-ebt-remi-duration/baseline"),
                 ("ebt", "pitch sweep", LM / "tune-ebt-remi-pitch_register/baseline"),
                 ("llama", "tilt sweep", AR / "tune_llama-ebtrecipe_tilt_remi_20261019/midi/baseline"),
                 ("llama", "best-of-N sweep", AR / "tune_llama-ebtrecipe_best_of_n_remi_20261019/midi/baseline"),
                 ("llama", "PPLM sweep", AR / "tune_llama-ebtrecipe_pplm_remi_20261104/midi/baseline"),
                 ("gpt2", "tilt sweep", AR / "tune_gpt2-ebtrecipe_tilt_remi_20261019/midi/baseline"),
                 ("gpt2", "best-of-N sweep", AR / "tune_gpt2-ebtrecipe_best_of_n_remi_20261019/midi/baseline")],
    },
    "ant": {
        "test": [("ebt", "duration sweep", LM / "25296307/baseline"), ("ebt", "pitch sweep", LM / "25296308/baseline"),
                 ("ebt", "duration fine sweep", LM / "25296410/baseline"), ("ebt", "pitch fine sweep", LM / "25296411/baseline"),
                 ("llama", "best-of-N sweep", AR / "llama-ebtrecipe_best_of_n_ant_20261019/midi/baseline"),
                 ("llama", "PPLM sweep", AR / "llama-ebtrecipe_pplm_ant_20261105/midi/baseline"),
                 ("gpt2", "best-of-N sweep", AR / "gpt2-ebtrecipe_best_of_n_ant_20261019/midi/baseline")],
        "tune": [("ebt", "duration sweep", LM / "tune-ebt-ant-duration-coarse/baseline"),
                 ("ebt", "pitch sweep", LM / "tune-ebt-ant-pitch_register-coarse/baseline"),
                 ("ebt", "duration fine sweep", LM / "tune-ebt-ant-duration-fine/baseline"),
                 ("ebt", "pitch fine sweep", LM / "tune-ebt-ant-pitch_register-fine/baseline"),
                 ("llama", "best-of-N sweep", AR / "tune_llama-ebtrecipe_best_of_n_ant_20261019/midi/baseline"),
                 ("llama", "PPLM sweep", AR / "tune_llama-ebtrecipe_pplm_ant_20261105/midi/baseline"),
                 ("gpt2", "best-of-N sweep", AR / "tune_gpt2-ebtrecipe_best_of_n_ant_20261019/midi/baseline")],
    },
}
# The music-quality runs: 100 validation songs, one unguided continuation per model (prompt = song start).
MQ = bs.LOGS.parent / "music_quality"
SONG_RUNS = {"remi": "remi", "ant": "ant-at-full-ar"}
CATCH_LAMBDA = {"remi": "0.16", "ant": "0.32"}


def prompts(tok):
    d = UNGUIDED[tok]["ebt"]
    return sorted(int(p.name[1:].split("_")[0]) for p in d.glob("*_r0_generated.mid"))


def steered(tok, sysdef, attr, pid, sd):
    _, _, run, method, strengths = sysdef
    s = strengths[attr]
    if method == "ebt":
        return bs.ebt_path(tok, attr, pid, sd, s)
    return run / attr / f"p{pid}_{attr}_{method}{s}_d{sd:+g}_generated.mid"


def reference(sysdef, attr, steered_path, pid):
    run = steered_path.parent.parent if sysdef[3] == "ebt" else sysdef[2]
    return run / "baseline" / f"p{pid}_r0_generated.mid"


def manifest():
    clips, missing = [], []

    def add(path, **info):
        if path is None or not Path(path).exists():
            missing.append({**info, "path": str(path)})
            return
        info["id"] = "-".join(str(info[k]) for k in ("tok", "section", "system", "prompt", "cond"))
        clips.append({**info, "midi": str(path)})

    for tok in UNGUIDED:
        for pid in prompts(tok):
            for m, d in UNGUIDED[tok].items():
                name = f"p{pid}_ground_truth.mid" if m == "original" else f"p{pid}_r0_generated.mid"
                add(d / name, tok=tok, section="unguided", system=m, prompt=pid, cond="unguided")
            for attr in ATTRIBUTES[tok]:
                for sysdef in SYSTEMS[tok]:
                    if attr not in sysdef[4]:
                        continue
                    info = dict(tok=tok, section=attr, system=sysdef[0], label=sysdef[1], prompt=pid,
                                strength=sysdef[4][attr])
                    down, up = (steered(tok, sysdef, attr, pid, s) for s in (-SD, SD))
                    if up is not None:
                        add(reference(sysdef, attr, Path(up), pid), **info, cond="reference")
                    add(down, **info, cond="down")
                    add(up, **info, cond="up")
            add(bs.ebt_path(tok, "pitch_register", pid, SD, CATCH_LAMBDA[tok]), tok=tok, section="catch",
                system="ebt", label=f"Tonic λ {CATCH_LAMBDA[tok]}", prompt=pid, cond="up",
                strength=CATCH_LAMBDA[tok])
        pool_clips(tok, add)
        song_clips(tok, add)
    return clips, missing


def pool_clips(tok, add):
    """Section unguided_more: every unguided draw of every sweep except the Part 1 clip (r0 of
    UNGUIDED), on the test prompts and the tune prompts; the original continuation for tune prompts."""
    for split, runs in POOL_SWEEPS[tok].items():
        first = runs[0][2]
        for pid in sorted(int(p.name[1:].split("_")[0]) for p in first.glob("*_r0_generated.mid")):
            info = dict(tok=tok, section="unguided_more", split=split, prompt=pid)
            if split == "tune":
                add(first.parent / "ground_truth" / f"p{pid}_ground_truth.mid", **info, system="original",
                    label="original", cond="original")
            for system, label, d in runs:
                for r in range(3):
                    path = d / f"p{pid}_r{r}_generated.mid"
                    if split == "test" and path == UNGUIDED[tok][system] / f"p{pid}_r0_generated.mid":
                        continue  # already in Part 1
                    tag = label.replace(" ", "_").replace("-", "")
                    add(path, **info, system=system, label=f"{label} · draw {r + 1}", cond=f"{split}_{tag}_r{r}")


def song_clips(tok, add):
    """Section unguided_songs: the 100 music-quality songs (same songs, prompt = song start, one draw
    per model). EBT's files are numbered by generation order: matched to songs by prompt tokens."""
    sys.path.insert(0, str(HERE.parents[1]))
    from eval.unguided_quality_ci import ebt_song_ids
    slug = SONG_RUNS[tok]
    ebt = {song: num for num, song in ebt_song_ids(MQ, slug).items()}
    base = {m: MQ / f"gen_{m}-ebtrecipe_{slug}" / "midi" for m in ("llama", "gpt2")}
    for song in sorted(ebt):
        info = dict(tok=tok, section="unguided_songs", prompt=song, cond="song")
        add(base["llama"] / f"sample_{song}_ground_truth.mid", **info, system="original", label="original")
        add(MQ / f"gen_ebt_{slug}" / "midi" / f"sample_{ebt[song]:03d}_generated.mid", **info, system="ebt", label="Tonic")
        for m, d in base.items():
            # Baselines save the continuation alone in *_generated.mid; the full clip is this file.
            add(d / f"sample_{song}_prompt_with_generated_continuation.mid", **info, system=m, label=m,
                prompt_midi=str(d / f"sample_{song}_prompt.mid"))


def render_one(clip):
    """MIDI -> mono 32 kbps MP3 (small enough to embed ~1,000 clips in one page)."""
    mp3 = OUT / "mp3" / f"{clip['id']}.mp3"
    midi = Path(clip["midi"])
    notes = bs._notes(midi)
    end = max((n[1] for n in notes), default=0.0) + bs.TAIL
    pp = Path(clip["prompt_midi"]) if clip.get("prompt_midi") else bs.prompt_of(midi)
    prompt_end = max((n[1] for n in bs._notes(pp)), default=0.0) if pp.exists() else None
    if not mp3.exists():
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "x.wav"
            subprocess.run([str(bs.BIN / "fluidsynth"), "-ni", "-q", "-r", "44100", "-g", "0.6",
                            "-F", str(wav), bs.SOUNDFONT, str(midi)],
                           check=True, capture_output=True, timeout=120)
            bs._trim_wav(wav, end)
            subprocess.run([str(bs.BIN / "lame"), "--quiet", "-m", "m", "--resample", "22.05",
                            "-b", "32", str(wav), str(mp3)], check=True, timeout=120)
    return {**clip, "seconds": round(end, 2),
            "prompt_end": None if prompt_end is None else round(prompt_end, 2)}


def cmd_render(args):
    clips, missing = manifest()
    print(f"{len(clips)} clips, {len(missing)} missing")
    for m in missing:
        print("  missing:", m)
    (OUT / "mp3").mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        done = []
        for i, c in enumerate(pool.imap(render_one, clips, chunksize=4)):
            done.append(c)
            if i % 100 == 0:
                print(f"  {i}/{len(clips)}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps({"clips": done, "missing": missing}, indent=1))
    print("wrote", OUT / "manifest.json")


# Sweep tables that hold the per-sample attribute values (the numbers the accuracy is computed from).
SW = bs.LOGS.parent / "music_quality" / "sweeps"
EBT_TABLES = {"25280946": "ebt_remi_tables/velocity_remi", "25280947": "ebt_remi_tables/duration_remi",
              "25280948": "ebt_remi_tables/pitch_register_remi",
              "25296307": "ebt_antar_tables/duration_ant-ar-coarse", "25296410": "ebt_antar_tables/duration_ant-ar-fine",
              "25296308": "ebt_antar_tables/pitch_register_ant-ar-coarse",
              "25296411": "ebt_antar_tables/pitch_register_ant-ar-fine"}
BUDGET = {"remi": "tune_remi.json", "ant": "tune_ant_ar.json"}


def _rows(table_path):
    t = json.loads(Path(table_path).read_text())
    return {r["sample_id"]: r for r in (dict(zip(t["columns"], row)) for row in t["data"])}


def _quality(csv_path):
    import csv
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def annotate(clips):
    """Attach attribute values (baseline mean of 3 draws, target, achieved, all in the attribute's
    [0, 1] scale) and harsh dissonance to every clip, from the sweep tables / quality_scores.csv."""
    tables, quals = {}, {}
    for c in clips:
        midi = Path(c["midi"])
        sid = midi.name.replace("_generated.mid", "").replace("_ground_truth.mid", "")
        run = midi.parent.parent   # LM/<job> or AR/<run>/midi
        if run not in quals:
            quals[run] = _quality(run / "quality_scores.csv")
        q = quals[run].get(sid)
        if q and q.get("sharp_dissonance") not in (None, ""):
            c["dissonance"] = round(float(q["sharp_dissonance"]), 3)
        if c["section"].startswith("unguided") or c["section"] == "catch":
            attr = None
        else:
            attr = c["section"]
        if attr is None:
            continue
        key = (run, attr)
        if key not in tables:
            if run.parent == LM:
                files = [SW / f"{EBT_TABLES[run.name]}.table.json"]
            else:
                files = sorted(run.parent.glob(f"*_{attr}.table.json"))
            tables[key] = {}
            for f in files:
                tables[key].update(_rows(f))
        r = tables[key].get(sid)
        if r is None:
            continue
        c["achieved"] = r["achieved_value"]
        c["baseline"] = r["baseline_value"]
        if r.get("target_delta") is not None:
            c["target_delta"] = r["target_delta"]
            c["sd_unit"] = abs(r["target_delta"]) / SD
    # Reference clips: express against the same baseline / SD unit as the steered clips of that row.
    by_row = {}
    for c in clips:
        if c.get("sd_unit"):
            by_row[(c["tok"], c["section"], c["system"], c["prompt"])] = (c["baseline"], c["sd_unit"])
    for c in clips:
        k = (c["tok"], c["section"], c["system"], c["prompt"])
        if c["cond"] == "reference" and k in by_row:
            c["baseline"], c["sd_unit"] = by_row[k]
    return {tok: json.loads((SW / f).read_text())["_budget"] for tok, f in BUDGET.items()}


def cmd_pack(args):
    """Chunks of base64 MP3 (<= ~10 MB each, one section per chunk group) + index.json."""
    man = json.loads((OUT / "manifest.json").read_text())
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    budget = annotate(man["clips"])
    groups = {}
    for c in man["clips"]:
        groups.setdefault((c["tok"], c["section"]), []).append(c)
    index = {"clips": {}, "chunks": {}, "budget": budget}
    for (tok, section), cs in groups.items():
        part, size, n = {}, 0, 0

        def flush():
            nonlocal part, size, n
            if part:
                name = f"data/{tok}-{section}-{n}.json"
                (out / name).write_text(json.dumps(part))
                index["chunks"][name] = list(part)
                part, size, n = {}, 0, n + 1
        for c in cs:
            b = base64.b64encode((OUT / "mp3" / f"{c['id']}.mp3").read_bytes()).decode()
            if size + len(b) > 10_000_000:
                flush()
            part[c["id"]] = b
            size += len(b)
            index["clips"][c["id"]] = {k: v for k, v in c.items() if k != "id"}
        flush()
    (out / "data" / "index.json").write_text(json.dumps(index))
    total = sum(p.stat().st_size for p in (out / "data").glob("*.json"))
    print(f"{len(index['clips'])} clips in {len(index['chunks'])} chunks, {total / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("pack")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"render": cmd_render, "pack": cmd_pack}[args.cmd](args)


if __name__ == "__main__":
    main()
