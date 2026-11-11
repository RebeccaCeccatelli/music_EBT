#!/usr/bin/env python3
"""Render every guided clip of the final sweeps, for picking by ear (survey + showcase).

    python demo/survey/guided_candidates.py render [--workers 16]   # all guided clips -> MP3 + manifest
    python demo/survey/guided_candidates.py pack --tok remi --out DIR  # picker subset for one tokenizer

Sources: EBT (REMI 33,732 / Ant 88,800) and the EBT-recipe Llama / GPT-2, every strength x target
(±0.5/1/2 SD) on the 16 test and the 16 held-out tune prompts, plus each run's own unguided
reference (r0). Output: ~/orcd/scratch/rebcecca/music_EBT_logs/survey/guided/{mp3/,manifest.json}

The single-attribute picker subset (pack), one section per attribute:
  every system at its operating point (chosen on the tune prompts,
  music_quality/sweeps/tune_{remi,ant_ar}.json), all 6 targets, test + tune prompts
  (REMI EBT pitch also at λ 0.02, the threshold case, 2026-11-05_heldout_guidance.md),
  plus EBT at every λ x all 6 targets on the test prompts (the former λ explorer, merged 2026-11-09).
Only the operating-point clips are valid survey items: the other strengths have no matched
counterpart for the other systems.
Audio is packed as raw MP3 bytes concatenated into data/<tok>-<attr>-<n>.mp3 chunks; index.json gives
each clip's [chunk, offset, length] (base64 JSON made REMI 276 MB, over the 256 MB artifact cap).
"""
import argparse
import base64
import json
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "showcase"))
sys.path.insert(0, str(HERE))
import build_showcase as bs  # noqa: E402
import candidates as cand  # noqa: E402  (table / quality helpers)

LM, AR = bs.LM, bs.AR
OUT = bs.LOGS.parent / "survey" / "guided"
SW = bs.LOGS.parent / "music_quality" / "sweeps"
SDS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

# (tok, split, attr, EBT run dir, table under music_quality/sweeps)
EBT_RUNS = [
    ("remi", "test", "velocity", "25280946", "ebt_remi_tables/velocity_remi"),
    ("remi", "test", "duration", "25280947", "ebt_remi_tables/duration_remi"),
    ("remi", "test", "pitch_register", "25280948", "ebt_remi_tables/pitch_register_remi"),
    ("ant", "test", "duration", "25296307", "ebt_antar_tables/duration_ant-ar-coarse"),
    ("ant", "test", "pitch_register", "25296308", "ebt_antar_tables/pitch_register_ant-ar-coarse"),
    ("ant", "test", "duration", "25296410", "ebt_antar_tables/duration_ant-ar-fine"),
    ("ant", "test", "pitch_register", "25296411", "ebt_antar_tables/pitch_register_ant-ar-fine"),
    ("remi", "tune", "velocity", "tune-ebt-remi-velocity", "ebt_tune_tables/velocity_remi"),
    ("remi", "tune", "duration", "tune-ebt-remi-duration", "ebt_tune_tables/duration_remi"),
    ("remi", "tune", "pitch_register", "tune-ebt-remi-pitch_register", "ebt_tune_tables/pitch_register_remi"),
    ("ant", "tune", "duration", "tune-ebt-ant-duration-coarse", "ebt_tune_tables/duration_ant-ar-coarse"),
    ("ant", "tune", "pitch_register", "tune-ebt-ant-pitch_register-coarse", "ebt_tune_tables/pitch_register_ant-ar-coarse"),
    ("ant", "tune", "duration", "tune-ebt-ant-duration-fine", "ebt_tune_tables/duration_ant-ar-fine"),
    ("ant", "tune", "pitch_register", "tune-ebt-ant-pitch_register-fine", "ebt_tune_tables/pitch_register_ant-ar-fine"),
]
METHOD_SYS = {"tilt": "tilt", "best_of_n": "bon", "pplm": "pplm"}
METHOD_LABEL = {"tilt": "tilt", "best_of_n": "best-of-N", "pplm": "PPLM"}
EBT_RE = re.compile(r"p(\d+)_(\w+?)_r3([\d.]+)_d([+-][\d.]+)_generated\.mid$")
AR_RE = re.compile(r"p(\d+)_(\w+?)_(tilt|best_of_n|pplm)([\d.]+)_d([+-][\d.]+)_generated\.mid$")


def num(s):
    return f"{float(s):g}"


def ar_runs():
    """(tok, split, model, method, run midi dir) for every EBT-recipe baseline sweep."""
    for d in sorted(AR.glob("*ebtrecipe_*")):
        m = re.match(r"(tune_)?(llama|gpt2)-ebtrecipe_(tilt|best_of_n|pplm)_(.*)", d.name)
        if m and (d / "midi").is_dir():
            yield ("ant" if "ant" in m.group(4) else "remi", "tune" if m.group(1) else "test",
                   m.group(2), m.group(3), d / "midi")


def manifest():
    clips, seen = [], set()

    def add(path, **info):
        info["id"] = "-".join(str(info[k]) for k in ("tok", "split", "system", "attr", "prompt", "strength", "cond"))
        if info["id"] in seen:  # Ant EBT coarse and fine grids share λ 0.01 / 0.02: keep the first (coarse)
            return
        seen.add(info["id"])
        clips.append({**info, "midi": str(path)})

    refs = set()
    for tok, split, attr, job, table in EBT_RUNS:
        run = LM / job
        hits = [(EBT_RE.search(p.name), p) for p in sorted((run / "guided").glob("*_generated.mid"))]
        raws = sorted({float(h.group(4)) for h, _ in hits if h})
        assert len(raws) == 6, (job, raws)  # ±0.5/1/2 SD in raw units -> map in order
        to_sd = dict(zip(raws, SDS))
        for h, p in hits:
            if not h or h.group(2) != attr:
                continue
            pid, lam, sd = int(h.group(1)), num(h.group(3)), to_sd[float(h.group(4))]
            add(p, tok=tok, split=split, system="ebt", label="Tonic", attr=attr, prompt=pid, strength=lam,
                sd=sd, cond=f"d{sd:+g}", table=table)
            if (tok, split, "ebt", attr, pid) not in refs:
                refs.add((tok, split, "ebt", attr, pid))
                add(run / "baseline" / f"p{pid}_r0_generated.mid", tok=tok, split=split, system="ebt",
                    label="Tonic", attr=attr, prompt=pid, strength="ref", sd=None, cond="reference", table=table)
    for tok, split, model, method, mdir in ar_runs():
        system = f"{model}_{METHOD_SYS[method]}"
        label = f"{'Llama' if model == 'llama' else 'GPT-2'} {METHOD_LABEL[method]}"
        for adir in sorted(p for p in mdir.iterdir() if p.is_dir() and p.name not in ("baseline", "ground_truth")):
            for p in sorted(adir.glob("*_generated.mid")):
                h = AR_RE.search(p.name)
                if not h:
                    continue
                pid, attr, s, sd = int(h.group(1)), h.group(2), num(h.group(4)), float(h.group(5))
                add(p, tok=tok, split=split, system=system, label=label, attr=attr, prompt=pid, strength=s,
                    sd=sd, cond=f"d{sd:+g}")
                if (tok, split, system, attr, pid) not in refs:
                    refs.add((tok, split, system, attr, pid))
                    add(mdir / "baseline" / f"p{pid}_r0_generated.mid", tok=tok, split=split, system=system,
                        label=label, attr=attr, prompt=pid, strength="ref", sd=None, cond="reference")
    return clips


def render_one(clip):
    """render_one_ but never raises: a clip that fails (e.g. fluidsynth hangs, 2026-11-05) is skipped."""
    try:
        return render_one_(clip)
    except Exception as e:  # noqa: BLE001
        return {**clip, "failed": f"{type(e).__name__}: {str(e)[:200]}"}


def render_one_(clip):
    """MIDI -> mono 24 kbps MP3 (~27 KB per clip)."""
    mp3 = OUT / "mp3" / f"{clip['id']}.mp3"
    midi = Path(clip["midi"])
    notes = bs._notes(midi)
    end = max((n[1] for n in notes), default=0.0) + bs.TAIL
    pp = bs.prompt_of(midi)
    prompt_end = max((n[1] for n in bs._notes(pp)), default=0.0) if pp.exists() else None
    if not mp3.exists():
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "x.wav"
            subprocess.run([str(bs.BIN / "fluidsynth"), "-ni", "-q", "-r", "44100", "-g", "0.6",
                            "-F", str(wav), bs.SOUNDFONT, str(midi)],
                           check=True, capture_output=True, timeout=120)
            bs._trim_wav(wav, end)
            subprocess.run([str(bs.BIN / "lame"), "--quiet", "-m", "m", "--resample", "22.05",
                            "-b", "24", str(wav), str(mp3)], check=True, timeout=120)
    return {**clip, "seconds": round(end, 2),
            "prompt_end": None if prompt_end is None else round(prompt_end, 2)}


def cmd_render(args):
    clips = manifest()
    missing = [c for c in clips if not Path(c["midi"]).exists()]
    clips = [c for c in clips if Path(c["midi"]).exists()]
    print(f"{len(clips)} clips, {len(missing)} missing", flush=True)
    for m in missing[:20]:
        print("  missing:", m["midi"])
    if args.dry_run:
        from collections import Counter
        for k, v in sorted(Counter((c["tok"], c["split"], c["system"], c["attr"]) for c in clips).items()):
            print(v, k)
        return
    (OUT / "mp3").mkdir(parents=True, exist_ok=True)
    done = []
    with Pool(args.workers) as pool:
        for i, c in enumerate(pool.imap(render_one, clips, chunksize=8)):
            done.append(c)
            if i % 500 == 0:
                print(f"  {i}/{len(clips)}", flush=True)
                (OUT / "manifest.json").write_text(json.dumps({"clips": done, "missing": missing}))
    failed = [c for c in done if c.get("failed")]
    done = [c for c in done if not c.get("failed")]
    for c in failed:
        print("  failed:", c["midi"], c["failed"])
    (OUT / "manifest.json").write_text(json.dumps({"clips": done, "missing": missing, "failed": failed}, indent=1))
    print(f"wrote {OUT / 'manifest.json'}: {len(done)} clips, {len(failed)} failed")


def operating_points():
    """{(tok, attr, system): strength} from the tune-prompt scoring, plus REMI EBT pitch λ 0.02."""
    ops = {}
    for tok, f in (("remi", "tune_remi.json"), ("ant", "tune_ant_ar.json")):
        for attr, by in json.loads((SW / f).read_text())["_operating_points"].items():
            for key, e in by.items():
                if not e:
                    continue
                model, method = key.split("/")
                system = "ebt" if model == "ebt" else f"{model.split('-')[0]}_{METHOD_SYS[method]}"
                ops.setdefault((tok, attr, system), set()).add(num(e["strength"]))
    ops[("remi", "pitch_register", "ebt")].add("0.02")
    return ops


def annotate(clips):
    """Baseline (mean of 3 unguided draws), target, achieved (attribute scale) and harsh dissonance."""
    tables, quals = {}, {}
    for c in clips:
        midi = Path(c["midi"])
        sid = midi.name.replace("_generated.mid", "")
        run = midi.parent.parent
        if run not in quals:
            quals[run] = cand._quality(run / "quality_scores.csv")
        q = quals[run].get(sid)
        if q and q.get("sharp_dissonance") not in (None, ""):
            c["dissonance"] = round(float(q["sharp_dissonance"]), 3)
        key = (run, c["attr"])
        if key not in tables:
            files = ([SW / f"{c['table']}.table.json"] if c.get("table")
                     else sorted(run.parent.glob(f"*_{c['attr']}.table.json")))
            tables[key] = {}
            for f in files:
                tables[key].update(cand._rows(f))
        r = tables[key].get(sid)
        if r and c["sd"] is not None:
            c["achieved"], c["baseline"] = r["achieved_value"], r["baseline_value"]
            if r.get("target_delta"):
                c["sd_unit"] = abs(r["target_delta"]) / abs(c["sd"])
    # References: same baseline / SD unit as the steered clips of that (run, prompt).
    unit = {}
    for c in clips:
        if c.get("sd_unit"):
            unit[(c["tok"], c["split"], c["system"], c["attr"], c["prompt"])] = (c["baseline"], c["sd_unit"])
    tables_r = {}
    for c in clips:
        k = (c["tok"], c["split"], c["system"], c["attr"], c["prompt"])
        if c["cond"] == "reference" and k in unit:
            c["baseline"], c["sd_unit"] = unit[k]
            run = Path(c["midi"]).parent.parent
            sid = Path(c["midi"]).name.replace("_generated.mid", "")
            if run not in tables_r:
                tables_r[run] = cand._quality(run / "quality_scores.csv")
    return {tok: json.loads((SW / f).read_text())["_budget"] for tok, f in cand.BUDGET.items()}


def subset(clips, tok):
    """Picker clips for one tokenizer: operating points (all targets, test + tune) and EBT at every λ (test)."""
    ops = operating_points()
    ear = json.loads((HERE / "ear_operating_points.json").read_text()).get(tok, {})  # user's by-ear λ
    out = []
    for c in clips:
        if c["tok"] != tok:
            continue
        op = ops.get((tok, c["attr"], c["system"]), set())
        if c["cond"] == "reference" or c["strength"] in op or (c["system"] == "ebt" and c["split"] == "test"):
            out.append({**c, "section": c["attr"], "op": c["strength"] in op,
                        "ear": c["system"] == "ebt" and c["strength"] in ear.get(c["attr"], [])})
    return out


def cmd_pack(args, chunk_bytes=8_000_000):
    man = json.loads((OUT / "manifest.json").read_text())
    clips = subset(man["clips"], args.tok)
    budget = annotate(clips)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    for old in (out / "data").glob("*"):
        old.unlink()
    groups = {}
    for c in clips:
        groups.setdefault(c["section"], []).append(c)
    index = {"clips": {}, "chunks": {}, "budget": budget, "tok": args.tok, "format": "mp3-concat"}
    for section, cs in groups.items():
        n, buf = 0, bytearray()

        def flush():
            nonlocal n, buf
            if buf:
                (out / f"data/{args.tok}-{section}-{n}.mp3").write_bytes(bytes(buf))
                n, buf = n + 1, bytearray()
        for c in cs:
            b = (OUT / "mp3" / f"{c['id']}.mp3").read_bytes()
            if buf and len(buf) + len(b) > chunk_bytes:
                flush()
            cid = f"{c['section']}~{c['id']}"
            index["clips"][cid] = {k: v for k, v in c.items() if k not in ("id", "table")}
            index["clips"][cid]["at"] = [f"data/{args.tok}-{section}-{n}.mp3", len(buf), len(b)]
            buf += b
        flush()
    (out / "data" / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    files = sorted((out / "data").glob("*"))
    print(f"{len(index['clips'])} clips in {len(files) - 1} chunks, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")


def cmd_explorer(args):
    """Tonic explorer page data: every λ × all six targets + reference, test prompts, both tokenizers.
    Sections are the attributes; each λ is a row (same layout as the operating-point tabs)."""
    man = json.loads((OUT / "manifest.json").read_text())
    ops = operating_points()
    clips = [{**c, "section": c["attr"], "op": c["strength"] in ops.get((c["tok"], c["attr"], "ebt"), set())}
             for c in man["clips"] if c["system"] == "ebt" and c["split"] == "test"]
    budget = annotate(clips)
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    index = {"clips": {}, "chunks": {}, "budget": budget, "mode": "explorer"}
    groups = {}
    for c in clips:
        groups.setdefault((c["tok"], c["section"]), []).append(c)
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
            index["clips"][c["id"]] = {k: v for k, v in c.items() if k not in ("id", "table")}
        flush()
    (out / "data" / "index.json").write_text(json.dumps(index))
    total = sum(p.stat().st_size for p in (out / "data").glob("*.json"))
    print(f"{len(index['clips'])} clips in {len(index['chunks'])} chunks, {total / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--dry_run", action="store_true")
    p = sub.add_parser("pack")
    p.add_argument("--tok", required=True, choices=["remi", "ant"])
    p.add_argument("--out", required=True)
    e = sub.add_parser("explorer")
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    {"render": cmd_render, "pack": cmd_pack, "explorer": cmd_explorer}[args.cmd](args)


if __name__ == "__main__":
    main()
