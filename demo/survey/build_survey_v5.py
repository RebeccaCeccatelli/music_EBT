#!/usr/bin/env python3
"""Build survey v5 from the clips the author picked by ear (2026-11-06).

    python demo/survey/build_survey_v5.py pool     # print the trial pool, no rendering
    python demo/survey/build_survey_v5.py build    # render audio + survey.json + the v5 key

v4 (build_survey.py) chose prompts by a fixed rule and showed every system at its operating
point. v5 is a curated, qualitative study (diary 2026-11-06):
  Part 1  unguided  one set per picked prompt: the original + the picked EBT / Llama / GPT-2
                    (EBT-recipe) continuation. Prompts picked twice (two sets) are kept, but a
                    participant never gets two sets of the same prompt (`ukey`).
  Part 2  change    every picked steered clip (any λ / strength, any target size) vs that
                    system's own unguided reference (r0 of the same run). Scored on direction.
  Part 3  compare   per (tokenizer, attribute, prompt, direction): the picked clips of each
                    system; several picks of one system -> extra questions so every pick
                    appears; <= 4 versions (EBT always in when picked); one participant never
                    gets two questions of the same (prompt, attribute, direction) (`ukey`).
  Catch             Part 1: an EBT pitch clip far past its operating point among real/unguided
                    versions, on a test prompt no Part 1 set uses. Part 2: a reference vs itself.

Sources: picks snapshot music_EBT_logs/survey/picks_v5/{unguided,remi,ant,explorer}.json,
clip ids -> MIDI via survey/candidates/manifest.json (unguided) and survey/guided/manifest.json
(steered). Key: survey/key_v5.json (the v4 key.json is kept for decoding v4 responses).
"""

import argparse
import hashlib
import json
import random
import secrets
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "showcase"))
sys.path.insert(0, str(HERE))
import build_showcase as bs  # noqa: E402
from build_survey import ATTR_HELP, ATTR_NAME, SCALE  # noqa: E402  (wording unchanged from v4)

ROOT = bs.LOGS.parent                       # music_EBT_logs
SV = ROOT / "survey"
PICKS = SV / "picks_v5"
STAGE = SV / "v5_stage"                     # MIDI copied as <h>_generated.mid + <h>_prompt.mid
SITE = HERE.parent / "showcase" / "site" / "survey"
AUDIO = SITE / "audio"
KEY_PATH = SV / "key_v5.json"

PLAN = {"unguided": 6, "change": 12, "compare": 5, "catch_unguided": 1, "catch_change": 1}
MAX_VERSIONS = 4
SYSTEM_ORDER = ["ebt", "llama_pplm", "llama_tilt", "llama_bon", "gpt2_tilt", "gpt2_bon"]
CATCH_LAMBDA = {"remi": "0.16", "ant": "0.32"}


def picks(name):
    d = json.loads((PICKS / f"{name}.json").read_text())
    return (d.get("data") or d)["ids"]


def manifest(path):
    return {c["id"]: c for c in json.loads(path.read_text())["clips"]}


# ------------------------------------------------------------------ pool

def build_pool():
    cand = manifest(SV / "candidates" / "manifest.json")
    guided = manifest(SV / "guided" / "manifest.json")
    clips = {}          # source midi -> key info (+ prompt_midi override)

    def clip(c, **info):
        midi = c["midi"]
        if midi not in clips:
            clips[midi] = {**info, "prompt_midi": c.get("prompt_midi")}
        return midi

    pool = {"unguided": [], "change": [], "compare": [], "catch_unguided": [], "catch_change": []}
    problems = []

    # Part 1: sets = (tok, tab, prompt); the original comes from the matching candidate clip.
    sets = defaultdict(dict)
    for cid in picks("unguided"):
        c = cand[cid]
        if c["system"] == "original":
            continue  # added for every set below
        key = (c["tok"], c["section"], c["prompt"])
        if c["system"] in sets[key]:
            problems.append(f"Part 1 set {key} has two {c['system']} picks; kept the first")
            continue
        sets[key][c["system"]] = c
    used_unguided_prompts = defaultdict(set)
    for (tok, section, pid), by in sorted(sets.items()):
        if set(by) != {"ebt", "llama", "gpt2"}:
            problems.append(f"Part 1 set {(tok, section, pid)} incomplete: {sorted(by)}; skipped")
            continue
        split = next(iter(by.values())).get("split")
        if section == "unguided_songs":
            orig = f"{tok}-unguided_songs-original-{pid}-song"
        elif section == "unguided_more" and split == "tune":
            orig = f"{tok}-unguided_more-original-{pid}-original"
        else:
            orig = f"{tok}-unguided-original-{pid}-unguided"
        opts = [clip(cand[orig], tok=tok, prompt=pid, system="original", source=section)]
        for m in ("ebt", "llama", "gpt2"):
            opts.append(clip(by[m], tok=tok, prompt=pid, system=m, condition="unguided", source=section,
                             pick=by[m]["id"]))
        pool["unguided"].append({"tok": tok, "prompt": pid, "ukey": f"{tok}|{section}|{pid}"
                                 if section == "unguided_songs" else f"{tok}|{pid}", "options": opts})
        used_unguided_prompts[tok].add(pid)

    # Steered picks (guided pickers + EBT explorer), deduplicated by clip id.
    steered = {}
    for name in ("remi", "ant", "explorer"):
        for pid_ in picks(name):
            cid = pid_.split("~")[-1]
            if cid not in guided:
                problems.append(f"steered pick {cid} not in the guided manifest (render failed?)")
                continue
            if guided[cid]["cond"] != "reference":
                steered[cid] = guided[cid]

    def ref_of(c):
        rid = "-".join(str(x) for x in (c["tok"], c["split"], c["system"], c["attr"], c["prompt"], "ref", "reference"))
        return guided.get(rid)

    # Part 2: every steered pick vs its own reference.
    groups = defaultdict(list)
    for cid, c in sorted(steered.items()):
        r = ref_of(c)
        if r is None:
            problems.append(f"no reference for {cid}")
            continue
        info = dict(tok=c["tok"], prompt=c["prompt"], system=c["system"], attribute=c["attr"])
        rm = clip(r, **info, condition="unguided")
        tm = clip(c, **info, condition="steered", sd=c["sd"], strength=c["strength"], pick=cid)
        pool["change"].append({"tok": c["tok"], "attribute": c["attr"], "reference": rm, "test": tm,
                               "group": SYSTEM_ORDER.index(c["system"]), "direction": 1 if c["sd"] > 0 else -1})
        groups[(c["tok"], c["attr"], c["prompt"], 1 if c["sd"] > 0 else -1)].append(c)

    # Part 3: per (tok, attr, prompt, direction), questions from the picked clips.
    for (tok, attr, pid, direction), cs in sorted(groups.items()):
        by = defaultdict(list)
        for c in sorted(cs, key=lambda c: (float(c["strength"]), abs(c["sd"]))):
            by[c["system"]].append(c)
        systems = [s for s in SYSTEM_ORDER if s in by]
        if len(systems) < 2:
            continue
        unused = {s: list(by[s]) for s in systems}
        nxt = {s: 0 for s in systems}
        rot = 0
        while any(unused.values()):
            others = [s for s in systems if s != "ebt"]
            # systems with unused clips first (rotating), then the rest, up to MAX_VERSIONS.
            others = others[rot % len(others):] + others[:rot % len(others)] if others else []
            others.sort(key=lambda s: not unused[s])
            chosen = (["ebt"] if "ebt" in systems else []) + others
            chosen = chosen[:MAX_VERSIONS]
            opts = []
            for s in chosen:
                if unused[s]:
                    c = unused[s].pop(0)
                else:
                    c = by[s][nxt[s] % len(by[s])]
                    nxt[s] += 1
                info = dict(tok=tok, prompt=pid, system=s, attribute=attr)
                opts.append(clip(c, **info, condition="steered", sd=c["sd"], strength=c["strength"], pick=c["id"]))
            pool["compare"].append({"tok": tok, "attribute": attr, "direction": direction,
                                    "ukey": f"{tok}|{attr}|{pid}|{direction}", "options": opts})
            rot += 1

    # Catch trials on test prompts no Part 1 set uses (fixed rule, Random(0)).
    for tok in ("remi", "ant"):
        test_pids = sorted({c["prompt"] for c in cand.values() if c["tok"] == tok and c["section"] == "catch"})
        free = [p for p in test_pids if p not in used_unguided_prompts[tok]]
        random.Random(0).shuffle(free)
        for pid in free[:2]:
            ids = [f"{tok}-unguided-original-{pid}-unguided", f"{tok}-unguided-llama-{pid}-unguided",
                   f"{tok}-unguided-gpt2-{pid}-unguided", f"{tok}-catch-ebt-{pid}-up"]
            if not all(i in cand for i in ids):
                problems.append(f"catch {tok} p{pid}: clip missing")
                continue
            opts = [clip(cand[ids[0]], tok=tok, prompt=pid, system="original"),
                    clip(cand[ids[1]], tok=tok, prompt=pid, system="llama", condition="unguided"),
                    clip(cand[ids[2]], tok=tok, prompt=pid, system="gpt2", condition="unguided"),
                    clip(cand[ids[3]], tok=tok, prompt=pid, system="ebt", condition="catch_broken",
                         strength=CATCH_LAMBDA[tok])]
            pool["catch_unguided"].append({"tok": tok, "prompt": pid, "options": opts, "expect_worst": opts[-1]})
        attr = "pitch_register"
        steer_pids = {c["prompt"] for c in steered.values() if c["tok"] == tok}
        pid = next(p for p in test_pids if p not in steer_pids)
        rid = f"{tok}-test-ebt-{attr}-{pid}-ref-reference"
        if rid in guided:
            r = clip(guided[rid], tok=tok, prompt=pid, system="ebt", attribute=attr, condition="unguided")
            pool["catch_change"].append({"tok": tok, "attribute": attr, "reference": r, "test": r,
                                         "expect": "no difference"})
        else:
            problems.append(f"catch_change {tok}: reference {rid} missing")
    return pool, clips, problems


# -------------------------------------------------------------------- build

def cmd_pool(_args):
    pool, clips, problems = build_pool()
    for k, v in pool.items():
        print(f"{k:16s} {len(v):4d} trials")
    print(f"{len(clips)} distinct clips")
    from collections import Counter
    print("Part 2 by tok/attr/system:", dict(Counter((t["tok"], t["attribute"], SYSTEM_ORDER[t["group"]])
                                                     for t in pool["change"])))
    print("Part 3 versions per question:", dict(Counter(len(t["options"]) for t in pool["compare"])))
    for p in problems:
        print("  !", p)


def stage(src, info, h):
    """Copy the clip (+ its prompt) to STAGE/<h>_generated.mid so the showcase helpers find the prompt."""
    STAGE.mkdir(parents=True, exist_ok=True)
    g, p = STAGE / f"{h}_generated.mid", STAGE / f"{h}_prompt.mid"
    if not g.exists():
        shutil.copyfile(src, g)
    pm = Path(info["prompt_midi"]) if info.get("prompt_midi") else bs.prompt_of(src)
    if pm.exists() and not p.exists():
        shutil.copyfile(pm, p)
    return g


def cmd_build(args):
    pool, clips, problems = build_pool()
    for p in problems:
        print("  !", p)
    key = json.loads(KEY_PATH.read_text()) if KEY_PATH.exists() else {"salt": secrets.token_hex(16), "clips": {}}
    AUDIO.mkdir(parents=True, exist_ok=True)

    def hid(midi):
        rel = str(Path(midi).relative_to(ROOT))
        return hashlib.sha256((key["salt"] + rel).encode()).hexdigest()[:16]

    meta = {}
    for i, (midi, info) in enumerate(clips.items()):
        h = hid(midi)
        g = stage(midi, info, h)
        mp3 = AUDIO / f"{h}.mp3"
        if args.force or not mp3.exists():
            bs.render_mp3(g, mp3)
        m = bs.metrics(g)
        notes = bs._notes(g)
        meta[h] = {"prompt_end": m["prompt_end"],
                   "duration": round(max((n[1] for n in notes), default=0.0), 2),
                   "audio": bs.mp3_seconds(mp3)}
        key["clips"][h] = {**{k: v for k, v in info.items() if k != "prompt_midi" and v is not None},
                           "midi": str(Path(midi).relative_to(ROOT)), "metrics": m}
        if i % 50 == 0:
            print(f"  {i}/{len(clips)}", flush=True)

    used = set(meta)
    for f in AUDIO.glob("*.mp3"):
        if f.stem not in used:
            f.unlink()

    def ids(t):
        out = {k: v for k, v in t.items() if k not in ("expect_worst", "prompt")}
        for k in ("reference", "test"):
            if k in out:
                out[k] = hid(out[k])
        if "options" in out:
            out["options"] = [hid(o) for o in out["options"]]
        if "ukey" in out:   # opaque: the browser only needs equality
            out["ukey"] = hashlib.sha256((key["salt"] + out["ukey"]).encode()).hexdigest()[:10]
        return out

    survey = {"version": args.version, "plan": PLAN, "scale": SCALE, "attr_name": ATTR_NAME,
              "attr_help": ATTR_HELP, "clips": meta,
              "pool": {k: [ids(t) for t in v] for k, v in pool.items()}}
    (SITE / "survey.json").write_text(json.dumps(survey, separators=(",", ":")))
    key["version"] = args.version
    key["catch_unguided_expect_worst"] = [hid(t["expect_worst"]) for t in pool["catch_unguided"]]
    key["pool"] = {k: [{kk: (hid(vv) if kk in ("reference", "test") else
                             [hid(o) for o in vv] if kk == "options" else vv)
                        for kk, vv in t.items() if kk != "expect_worst"} for t in v] for k, v in pool.items()}
    KEY_PATH.write_text(json.dumps(key, indent=1))
    total = sum(f.stat().st_size for f in AUDIO.glob("*.mp3"))
    print(f"wrote {SITE / 'survey.json'}: {len(meta)} clips, {total / 1e6:.1f} MB; key -> {KEY_PATH}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool")
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")
    b.add_argument("--version", default="v5", help="stored with every response")
    args = ap.parse_args()
    {"pool": cmd_pool, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
