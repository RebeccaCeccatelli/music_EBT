#!/usr/bin/env python3
"""Build the blind listening survey (demo/showcase/site/survey/) from sweep MIDI.

    python demo/survey/build_survey.py pool     # print the trial pool, no rendering
    python demo/survey/build_survey.py build    # render audio + write survey.json and the key

Design (each participant gets a random subset, see PLAN):
  Part 1  unguided   4 blinded continuations of one prompt (EBT, Llama, GPT-2,
                     human original as a hidden anchor); pick most / least musical.
  Part 2a change     reference = a system's own unguided output, test = the same
                     system steered ±2 SD at its operating point; 5-point
                     "clearly lower ... clearly higher" scale. Measures whether
                     the steering is audible, per system.
  Part 2b compare    the steered outputs of all systems (same prompt, attribute,
                     direction); pick most / least musical.
  Catch trials       Part 1: one option is near-random EBT output (λ far past the
                     operating point). Part 2a: reference vs. itself.

Blinding: audio files are named by a salted hash. The key (hash -> system,
prompt, settings) is written to KEY_PATH on scratch, NOT into the public site.

Prompt rule (no listening involved): per tokenizer, the sweep prompts in
Random(0) order, keeping those whose unguided continuations are >= 3 s with
>= 10 pitched notes and are not identical across EBT / Llama / GPT-2.
"""

import argparse
import hashlib
import json
import random
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "showcase"))
import build_showcase as bs  # noqa: E402  (run table, resolver, renderer)

SITE = HERE.parent / "showcase" / "site" / "survey"
AUDIO = SITE / "audio"
KEY_PATH = bs.LOGS.parent / "survey" / "key.json"   # private: never inside the site

SD = 2  # target size (corpus SD) for every steered clip

# Systems compared in Part 2, at their operating points.
SYSTEMS = {
    "remi": [
        {"system": "ebt", "method": "ebt", "model": "ebt"},
        {"system": "llama_pplm", "method": "pplm", "model": "llama",
         "strength": {"velocity": "16", "duration": "16", "pitch_register": "32"}},
        {"system": "llama_tilt", "method": "tilt", "model": "llama", "strength": "1"},
        {"system": "llama_best_of_n", "method": "best_of_n", "model": "llama", "strength": "16"},
    ],
    "ant": [
        {"system": "ebt", "method": "ebt", "model": "ebt"},
        {"system": "llama_best_of_n", "method": "best_of_n", "model": "llama", "strength": "16"},
    ],
}
ATTRIBUTES = {"remi": ["velocity", "duration", "pitch_register"],
              "ant": ["duration", "pitch_register"]}
N_PROMPTS = {"unguided": 6, "steer": 4}   # per tokenizer, in the pool

# How many trials one participant gets (sampled from the pool in the browser).
PLAN = {"unguided": 6, "change": 12, "compare": 5, "catch_unguided": 1, "catch_change": 1}

SCALE = {
    "velocity": ["clearly softer", "slightly softer", "no difference", "slightly louder", "clearly louder"],
    "duration": ["clearly shorter, more detached notes", "slightly shorter notes", "no difference",
                 "slightly longer notes", "clearly longer, more connected notes"],
    "pitch_register": ["clearly lower", "slightly lower", "no difference", "slightly higher", "clearly higher"],
}
ATTR_NAME = {"velocity": "loudness", "duration": "note length", "pitch_register": "pitch"}


# ------------------------------------------------------------------ prompts

def _pids(tok):
    d = bs.RUNS[tok]["unguided"]["ebt"]
    return sorted(int(p.name[1:].split("_")[0]) for p in d.glob("*_r0_generated.mid"))


def eligible_prompts(tok):
    pids = _pids(tok)
    random.Random(0).shuffle(pids)
    keep = []
    for pid in pids:
        paths = [bs.resolve({"tok": tok, "kind": "unguided", "model": m, "prompt": pid})
                 for m in ("ebt", "llama", "gpt2")]
        if not all(p.exists() for p in paths):
            continue
        conts = [bs.split_prompt(p)[0] for p in paths]
        m = bs.metrics(paths[0])
        if m["seconds"] < 3 or m["n_pitched"] < 10 or conts[0] == conts[1] == conts[2]:
            continue
        keep.append(pid)
    return keep


# --------------------------------------------------------------------- pool

def steered(tok, sysdef, attr, pid, sd):
    strength = sysdef.get("strength")
    if isinstance(strength, dict):
        strength = strength[attr]
    if sysdef["method"] == "ebt":
        strength = bs.OPERATING[tok][attr]
    return bs.resolve({"tok": tok, "kind": "guided", "method": sysdef["method"],
                       "model": sysdef["model"], "attribute": attr, "prompt": pid,
                       "sd": sd, "strength": strength}), strength


def reference(tok, sysdef, attr, pid):
    """The system's own unguided draw from the same run as its steered outputs."""
    if sysdef["method"] == "ebt":
        run = bs.RUNS[tok]["ebt"][attr][0].parent
    else:
        run = bs.RUNS[tok]["ar"][(sysdef["method"], sysdef["model"])]
    return run / "baseline" / f"p{pid}_r0_generated.mid"


def build_pool():
    clips = {}     # midi path -> key info

    def clip(path, **info):
        path = Path(path)
        if path not in clips:
            clips[path] = info
        return path

    pool = {"unguided": [], "change": [], "compare": [], "catch_unguided": [], "catch_change": []}
    skipped = []
    for tok in bs.RUNS:
        prompts = eligible_prompts(tok)
        for pid in prompts[:N_PROMPTS["unguided"]]:
            opts = [clip(bs.resolve({"tok": tok, "kind": "ground_truth", "prompt": pid}),
                         tok=tok, prompt=pid, system="original")]
            for m in ("ebt", "llama", "gpt2"):
                opts.append(clip(bs.resolve({"tok": tok, "kind": "unguided", "model": m, "prompt": pid}),
                                 tok=tok, prompt=pid, system=m, condition="unguided"))
            pool["unguided"].append({"tok": tok, "prompt": pid, "options": opts})

        for attr in ATTRIBUTES[tok]:
            for pid in prompts[:N_PROMPTS["steer"]]:
                for direction in (-1, 1):
                    compare = []
                    for sd_ in SYSTEMS[tok]:
                        test, strength = steered(tok, sd_, attr, pid, direction * SD)
                        ref = reference(tok, sd_, attr, pid)
                        if not (test and Path(test).exists() and ref.exists()):
                            skipped.append((tok, attr, pid, sd_["system"], direction))
                            continue
                        info = dict(tok=tok, prompt=pid, system=sd_["system"], attribute=attr)
                        r = clip(ref, **info, condition="unguided")
                        t = clip(test, **info, condition="steered", sd=direction * SD, strength=strength)
                        pool["change"].append({"tok": tok, "attribute": attr, "reference": r, "test": t,
                                               "group": SYSTEMS[tok].index(sd_)})
                        compare.append(t)
                    if len(compare) >= 2:
                        pool["compare"].append({"tok": tok, "attribute": attr, "options": compare})

        # Catch trials.
        for pid in prompts[:2]:
            broken = bs.ebt_path(tok, "pitch_register", pid, 2, "0.16" if tok == "remi" else "0.32")
            if broken and broken.exists():
                opts = [clip(bs.resolve({"tok": tok, "kind": "ground_truth", "prompt": pid}),
                             tok=tok, prompt=pid, system="original"),
                        clip(bs.resolve({"tok": tok, "kind": "unguided", "model": "llama", "prompt": pid}),
                             tok=tok, prompt=pid, system="llama", condition="unguided"),
                        clip(bs.resolve({"tok": tok, "kind": "unguided", "model": "gpt2", "prompt": pid}),
                             tok=tok, prompt=pid, system="gpt2", condition="unguided"),
                        clip(broken, tok=tok, prompt=pid, system="ebt", condition="catch_broken")]
                pool["catch_unguided"].append({"tok": tok, "prompt": pid, "options": opts,
                                               "expect_worst": str(opts[-1])})
        for attr in ATTRIBUTES[tok][:1]:
            pid = prompts[0]
            ref = reference(tok, SYSTEMS[tok][0], attr, pid)
            if ref.exists():
                r = clip(ref, tok=tok, prompt=pid, system="ebt", attribute=attr, condition="unguided")
                pool["catch_change"].append({"tok": tok, "attribute": attr, "reference": r, "test": r,
                                             "expect": "no difference"})
    return pool, clips, skipped


# -------------------------------------------------------------------- build

def load_key():
    if KEY_PATH.exists():
        return json.loads(KEY_PATH.read_text())
    return {"salt": secrets.token_hex(16), "clips": {}}


def cmd_pool(_args):
    pool, clips, skipped = build_pool()
    for k, v in pool.items():
        print(f"{k:16s} {len(v):4d} trials")
    print(f"{len(clips)} distinct clips; {len(skipped)} (system, prompt) slots missing")
    for s in skipped[:10]:
        print("  missing:", s)
    for tok in bs.RUNS:
        print(tok, "eligible prompts:", eligible_prompts(tok))


def cmd_build(args):
    pool, clips, skipped = build_pool()
    key = load_key()
    AUDIO.mkdir(parents=True, exist_ok=True)

    def hid(path):
        rel = str(Path(path).relative_to(bs.LOGS))
        return hashlib.sha256((key["salt"] + rel).encode()).hexdigest()[:16]

    meta = {}
    for i, (path, info) in enumerate(clips.items()):
        h = hid(path)
        mp3 = AUDIO / f"{h}.mp3"
        if args.force or not mp3.exists():
            bs.render_mp3(path, mp3)
        m = bs.metrics(path)
        notes = bs._notes(path)
        meta[h] = {"prompt_end": m["prompt_end"],
                   "duration": round(max((n[1] for n in notes), default=0.0), 2)}
        key["clips"][h] = {**info, "midi": str(Path(path).relative_to(bs.LOGS)), "metrics": m}
        if i % 20 == 0:
            print(f"  {i}/{len(clips)}", flush=True)

    used = set(meta)
    for f in AUDIO.glob("*.mp3"):
        if f.stem not in used:
            f.unlink()

    def ids(t):
        out = {k: v for k, v in t.items() if k not in ("expect_worst",)}
        for k in ("reference", "test"):
            if k in out:
                out[k] = hid(out[k])
        if "options" in out:
            out["options"] = [hid(o) for o in out["options"]]
        out.pop("prompt", None)   # prompt ids are not needed in the browser
        return out

    survey = {
        "version": args.version,
        "plan": PLAN,
        "scale": SCALE,
        "attr_name": ATTR_NAME,
        "clips": meta,
        "pool": {k: [ids(t) for t in v] for k, v in pool.items()},
    }
    (SITE / "survey.json").write_text(json.dumps(survey, separators=(",", ":")))
    key["catch_unguided_expect_worst"] = [hid(t["expect_worst"]) for t in pool["catch_unguided"]]
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(json.dumps(key, indent=1))
    total = sum(f.stat().st_size for f in AUDIO.glob("*.mp3"))
    print(f"wrote {SITE/'survey.json'}: {len(meta)} clips, {total/1e6:.1f} MB; key -> {KEY_PATH}")
    if skipped:
        print(f"{len(skipped)} slots missing (skipped)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool")
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")
    b.add_argument("--version", default="v1", help="stored with every response")
    args = ap.parse_args()
    {"pool": cmd_pool, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
