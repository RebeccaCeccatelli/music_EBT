#!/usr/bin/env python3
"""Combined-attribute control outputs (several attributes steered at once), for the Listening Archive.

    python demo/survey/combined_candidates.py render --tok remi [--workers 16]   # manifest + MP3s + harsh scores
    python demo/survey/combined_candidates.py pack --tok remi --out DIR          # page data (index + MP3 chunks)

Sources: attr_control/compose_guidance/<system>_<date>_<job>/ (compose_guidance_sweep.sh). REMI = velocity +
duration on the 16 test prompts, 8 target pairs (±1 / ±2 SD each, same or opposite directions), every strength
of each run, plus each run's unguided reference (r0). When a job ran more than once (preempted / requeued),
the most complete run directory is used. Runs listed in RUNS but not finished yet are skipped with a note,
so re-running render + pack later adds them.
Output: music_EBT_logs/survey/combined/<tok>/{mp3/,manifest.json}
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "showcase"))
sys.path.insert(0, str(HERE.parent.parent))
import build_showcase as bs  # noqa: E402

LOGS = bs.LOGS  # .../music_EBT_logs/attr_control
CG = LOGS / "compose_guidance"
OUT = LOGS.parent / "survey" / "combined"

# (tok, system id, label, job id). ebt_pa = EBT with one λ per attribute (strength "a:b").
RUNS = {
    "remi": [
        ("ebt", "Tonic (one shared λ)", "25416469"),
        ("ebt_pa", "Tonic (λ per attribute)", "28472156"),   # held-out operating points 0.05:0.055
        ("ebt_pa", "Tonic (λ per attribute)", "28472158"),   # chosen operating points 0.03:0.02, 0.04:0.025
        ("llama_pplm", "Llama PPLM", "28462282"),
        ("llama_tilt", "Llama tilt", "28462283"),
        ("gpt2_tilt", "GPT-2 tilt", "28462284"),
        ("llama_bon", "Llama best-of-N", "28462285"),
        ("gpt2_bon", "GPT-2 best-of-N", "28462286"),
    ],
    # Ant = duration + pitch register (tilt is REMI-only); the shared-λ EBT sweep was split in two λ halves
    "ant": [
        ("ebt", "Tonic (one shared λ)", "28462287"),          # λ 0.002, 0.004, 0.006
        ("ebt", "Tonic (one shared λ)", "28462288"),          # λ 0.008, 0.010, 0.012
        ("ebt_pa", "Tonic (λ per attribute)", "28472160"),    # held-out operating points 0.005:0.012
        ("ebt_pa", "Tonic (λ per attribute)", "28472162"),    # chosen operating points 0.005:0.005, 0.005:0.006
        ("llama_pplm", "Llama PPLM", "28462289"),
        ("llama_pplm", "Llama PPLM", "28472164"),           # per-attribute held-out operating points 16:32
        ("llama_bon", "Llama best-of-N", "28462290"),
        ("gpt2_bon", "GPT-2 best-of-N", "28462291"),
    ],
}
ATTRS = {"remi": ["velocity", "duration"], "ant": ["duration", "pitch_register"]}
# per-attribute EBT pairs to badge: held-out operating points (tune_*.json) and the by-ear ones (ear_operating_points.json)
OP_PAIRS = {"remi": {"ebt_pa": {"0.05:0.055"}}, "ant": {"ebt_pa": {"0.005:0.012"}, "llama_pplm": {"16:32"}}}
EAR_PAIRS = {"remi": {"ebt_pa": {"0.03:0.02", "0.04:0.025"}}, "ant": {"ebt_pa": {"0.005:0.005", "0.005:0.006"}}}
GEN_RE = re.compile(r"^p(\d+)_(?:r3|pplm|tilt|best_of_n)([\d.:]+)_c([+-][\d.]+)_([+-][\d.]+)_generated\.mid$")


def num(s):
    return ":".join(f"{float(x):g}" for x in str(s).split(":"))


def run_dir(job, attrs):
    """Most complete run directory of a job (requeued jobs leave a partial first directory)."""
    best = None
    for d in CG.glob(f"*_{job}"):
        if not list(d.glob("*.config.json")) or not all(list(d.glob(f"*_{a}.table.json")) for a in attrs):
            continue
        n = len(list((d / "midi" / attrs[0]).glob("*_generated.mid")))
        if best is None or n > best[0] or (n == best[0] and d.name > best[1].name):
            best = (n, d)
    return best[1] if best else None


def manifest(tok):
    attrs = ATTRS[tok]
    clips, notes = [], []
    for system, label, job in RUNS[tok]:
        d = run_dir(job, attrs)
        if d is None:
            notes.append(f"{label} (job {job}): not finished, skipped")
            continue
        cfg = json.loads(next(d.glob("*.config.json")).read_text())
        prompts = cfg["prompt_ids"]
        n_strengths = len(cfg.get("strengths") or cfg.get("strength_pairs") or [])
        expected = len(prompts) * len(cfg["combos"]) * max(n_strengths, 1)
        have = len(list((d / "midi" / attrs[0]).glob("*_generated.mid")))
        if have < expected:
            notes.append(f"{label} (job {job}): still running ({have}/{expected} clips), skipped")
            continue
        n_ok = 0
        for p in sorted((d / "midi" / attrs[0]).glob("*_generated.mid")):
            h = GEN_RE.match(p.name)
            if not h:
                continue
            pid, st, a, b = int(h.group(1)), num(h.group(2)), f"{float(h.group(3)):+g}", f"{float(h.group(4)):+g}"
            cond = f"{a}_{b}"
            clips.append({"tok": tok, "split": "test", "system": system, "label": label, "prompt": pid,
                          "strength": st, "cond": cond, "combo": [float(a), float(b)], "run": d.name,
                          "sample_id": p.name[: -len("_generated.mid")],
                          "id": f"{tok}-combo-{system}-{pid}-{st}-{cond}", "midi": str(p)})
            n_ok += 1
        for pid in prompts:
            p = d / "midi" / "baseline" / f"p{pid}_r0_generated.mid"
            if p.exists():
                clips.append({"tok": tok, "split": "test", "system": system, "label": label, "prompt": pid,
                              "strength": "ref", "cond": "reference", "combo": None, "run": d.name,
                              "sample_id": f"p{pid}_r0", "id": f"{tok}-combo-{system}-{job}-{pid}-ref",
                              "midi": str(p)})
        notes.append(f"{label} (job {job}): {d.name}, {n_ok} steered clips")
    seen, out = set(), []
    for c in clips:   # a strength can appear in two jobs of the same system: keep the first
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return out, notes


def render_one(clip):
    try:
        mp3 = OUT / clip["tok"] / "mp3" / f"{clip['id']}.mp3"
        midi = Path(clip["midi"])
        notes = bs._notes(midi)
        end = max((n[1] for n in notes), default=0.0) + bs.TAIL
        pp = bs.prompt_of(midi)
        prompt_end = max((n[1] for n in bs._notes(pp)), default=0.0) if pp.exists() else None
        if not mp3.exists():
            with tempfile.TemporaryDirectory() as tmp:
                wav = Path(tmp) / "x.wav"
                subprocess.run([str(bs.BIN / "fluidsynth"), "-ni", "-q", "-r", "44100", "-g", "0.6",
                                "-F", str(wav), bs.SOUNDFONT, str(midi)], check=True, capture_output=True, timeout=120)
                bs._trim_wav(wav, end)
                subprocess.run([str(bs.BIN / "lame"), "--quiet", "-m", "m", "--resample", "22.05",
                                "-b", "24", str(wav), str(mp3)], check=True, timeout=120)
        return {**clip, "seconds": round(end, 2), "prompt_end": None if prompt_end is None else round(prompt_end, 2)}
    except Exception as e:  # noqa: BLE001
        return {**clip, "failed": f"{type(e).__name__}: {str(e)[:200]}"}


def score_quality(run_dirs):
    """Per-file quality (harsh dissonance etc.) cached as <run>/midi/quality_scores.csv, as for the sweeps."""
    from eval.score_sweep_quality import score_root
    for d in sorted(run_dirs):
        score_root(CG / d / "midi")


def cmd_render(args):
    clips, notes = manifest(args.tok)
    for n in notes:
        print(" ", n, flush=True)
    (OUT / args.tok / "mp3").mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        done = list(pool.imap(render_one, clips, chunksize=8))
    failed = [c for c in done if c.get("failed")]
    done = [c for c in done if not c.get("failed")]
    for c in failed:
        print("  failed:", c["midi"], c["failed"])
    score_quality({c["run"] for c in done})
    (OUT / args.tok / "manifest.json").write_text(json.dumps({"clips": done, "failed": failed, "runs": notes}, indent=1))
    print(f"{len(done)} clips, {len(failed)} failed")


def annotate(clips, tok):
    """Per attribute: achieved, unguided level (mean of the 3 draws), SD unit, target in SD; harsh dissonance."""
    attrs, tables, quals = ATTRS[tok], {}, {}
    for c in clips:
        d = CG / c["run"]
        if d not in tables:
            tables[d] = {}
            for a in attrs:
                t = json.loads(next(d.glob(f"*_{a}.table.json")).read_text())
                tables[d][a] = {r["sample_id"]: r for r in (dict(zip(t["columns"], row)) for row in t["data"])}
            q = d / "midi" / "quality_scores.csv"
            if q.exists():
                import csv
                with open(q) as f:
                    quals[d] = {r["sample_id"]: r for r in csv.DictReader(f)}
        c["attrs"] = {}
        for i, a in enumerate(attrs):
            r = tables[d][a].get(c["sample_id"])
            if not r:
                continue
            e = {"achieved": r["achieved_value"], "baseline": r["baseline_value"]}
            if c["combo"] and r.get("target_delta"):
                e["sd"] = c["combo"][i]
                e["sd_unit"] = abs(r["target_delta"]) / abs(c["combo"][i])
            c["attrs"][a] = e
        q = quals.get(d, {}).get(c["sample_id"])
        if q and q.get("sharp_dissonance") not in (None, ""):
            c["dissonance"] = round(float(q["sharp_dissonance"]), 3)
    # references: SD unit of the same run and prompt
    unit = {}
    for c in clips:
        for a, e in c["attrs"].items():
            if "sd_unit" in e:
                unit[(c["run"], c["prompt"], a)] = e["sd_unit"]
    for c in clips:
        if c["cond"] == "reference":
            for a, e in c["attrs"].items():
                if (c["run"], c["prompt"], a) in unit:
                    e["sd_unit"] = unit[(c["run"], c["prompt"], a)]


def cmd_pack(args, chunk_bytes=8_000_000):
    man = json.loads((OUT / args.tok / "manifest.json").read_text())
    clips = man["clips"]
    annotate(clips, args.tok)
    sw = LOGS.parent / "music_quality" / "sweeps"
    budget = {}
    for tok, f in (("remi", "tune_remi.json"), ("ant", "tune_ant_ar.json")):
        try:
            budget[tok] = json.loads((sw / f).read_text())["_budget"]
        except Exception:  # noqa: BLE001
            pass
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    for old in (out / "data").glob("*"):
        old.unlink()
    section = "+".join(ATTRS[args.tok])
    index = {"clips": {}, "budget": budget, "tok": args.tok, "format": "mp3-concat", "attrs": ATTRS[args.tok],
             "runs": man.get("runs", [])}
    n, buf = 0, bytearray()

    def flush():
        nonlocal n, buf
        if buf:
            (out / f"data/{args.tok}-combined-{n}.mp3").write_bytes(bytes(buf))
            n, buf = n + 1, bytearray()
    for c in clips:
        b = (OUT / args.tok / "mp3" / f"{c['id']}.mp3").read_bytes()
        if buf and len(buf) + len(b) > chunk_bytes:
            flush()
        cid = f"{section}~{c['id']}"
        index["clips"][cid] = {k: v for k, v in c.items() if k not in ("id", "midi", "run", "sample_id")}
        index["clips"][cid]["section"] = section
        index["clips"][cid]["op"] = c["strength"] in OP_PAIRS.get(args.tok, {}).get(c["system"], set())
        index["clips"][cid]["ear"] = c["strength"] in EAR_PAIRS.get(args.tok, {}).get(c["system"], set())
        index["clips"][cid]["at"] = [f"data/{args.tok}-combined-{n}.mp3", len(buf), len(b)]
        buf += b
    flush()
    (out / "data" / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    files = sorted((out / "data").glob("*"))
    print(f"{len(index['clips'])} clips in {len(files) - 1} chunks, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--tok", required=True, choices=list(RUNS))
    r.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("pack")
    p.add_argument("--tok", required=True, choices=list(RUNS))
    p.add_argument("--out", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--tok", required=True, choices=list(RUNS))
    args = ap.parse_args()
    if args.cmd == "manifest":
        clips, notes = manifest(args.tok)
        print("\n".join(notes))
        print(len(clips), "clips")
        return
    {"render": cmd_render, "pack": cmd_pack}[args.cmd](args)


if __name__ == "__main__":
    main()
