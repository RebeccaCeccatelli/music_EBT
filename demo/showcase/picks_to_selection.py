#!/usr/bin/env python3
"""Fill the showcase's first two sections from the survey v5 picks (all of them, labelled).

    python demo/showcase/picks_to_selection.py          # rewrites sections "unguided" and "single" in selection.json

The survey shows each participant a sample of its trial pool; the showcase shows every
picked clip, with its system, target and strength in the open:
  Unguided Generation      one card per picked set: original, Llama, GPT-2, EBT (EBT-recipe baselines)
  Single-Attribute Control per tokenizer and attribute, one card per prompt; inside it one row per
                           system: that system's own unguided sample, then its picked steered versions.
Clips are referenced by their source MIDI (kind "file"), never by the survey's hashed file names,
so the public page cannot be used to decode the survey. The other sections are left untouched.
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "survey"))
import build_showcase as bs  # noqa: E402
import build_survey_v5 as sv  # noqa: E402

ROOT = bs.LOGS.parent
SYSTEMS = {  # system -> (block title, subtitle, model, method)
    "ebt": ("EBT", "the regressor's error is added to the energy the EBT minimises; λ = strength", "EBT", "ebt"),
    "llama_pplm": ("PPLM · Llama", "nudges the model's hidden state along the regressor's gradient", "Llama", "pplm"),
    "llama_tilt": ("Tilt · Llama", "reweights the attribute's own tokens (oracle value map)", "Llama", "tilt"),
    "llama_bon": ("Best-of-N · Llama", "keeps the closest of N unguided samples", "Llama", "best_of_n"),
    "gpt2_tilt": ("Tilt · GPT-2", "reweights the attribute's own tokens (oracle value map)", "GPT-2", "tilt"),
    "gpt2_bon": ("Best-of-N · GPT-2", "keeps the closest of N unguided samples", "GPT-2", "best_of_n"),
}
ATTR = {  # attribute -> (tab label, sublabel, metric key, metric label, unit, down word, up word)
    "velocity": ("Velocity", "loudness", "velocity", "mean velocity", "/127", "softer", "louder"),
    "duration": ("Duration", "staccato ↔ legato", "note_beats", "mean note length (beats)", "", "shorter", "longer"),
    "pitch_register": ("Pitch register", "low ↔ high", "pitch", "mean pitch", "", "lower", "higher"),
}
TOK = {"remi": "REMI", "ant": "Anticipation"}
TILT_NOTE = {"type": "note", "title": "Tilt: not applicable",
             "text": "Tilt needs a hand-written value map over the attribute's own tokens, which only exists for REMI."}
ANT_VELOCITY_NOTE = {"type": "note", "title": "Not available for Anticipation",
                     "text": "The Anticipation vocabulary has no velocity field: every note is rendered at the same "
                             "default velocity, so there is nothing to steer."}


def rel(midi):
    return str(Path(midi).relative_to(ROOT))


def clip_spec(midi, info, label, tag=None):
    """A 'file' clip: readable id (system, attribute, prompt, strength, target) + short path hash."""
    parts = [info["tok"], info.get("system"), info.get("attribute"), str(info["prompt"]),
             info.get("condition"), info.get("strength"),
             None if info.get("sd") is None else f"{info['sd']:+g}"]
    cid = "_".join(p for p in parts if p).replace(".", "p").replace("+", "u").replace("-", "d")
    cid = f"cur_{cid}_{hashlib.sha1(rel(midi).encode()).hexdigest()[:6]}"
    spec = {"kind": "file", "id": cid, "midi": rel(midi), "label": label, "trim": True}
    if info.get("prompt_midi"):
        spec["prompt_midi"] = info["prompt_midi"]
    if tag:
        spec["tag"] = tag
    return spec


def fmt_sd(sd):
    return f"{'+' if sd > 0 else '−'}{abs(sd):g} SD"


def strength_text(system, strength):
    if system == "ebt":
        return f"λ {strength}"
    if system.endswith("bon"):
        return f"best of {strength}"
    return f"strength {strength}"


def main():
    pool, clips, problems = sv.build_pool()
    for p in problems:
        print("  !", p)
    cand = sv.manifest(sv.SV / "candidates" / "manifest.json")
    originals = {(c["tok"], c["prompt"]): c["midi"] for c in cand.values()
                 if c["system"] == "original" and c["section"] == "unguided"}

    def name(tok, pid, orig=None):
        return bs.prompt_name(tok, pid, orig or originals.get((tok, pid)) or bs.prompt_of(Path("/nonexistent")))

    # ---------------------------------------------------------------- unguided
    order = {"original": 0, "llama": 1, "gpt2": 2, "ebt": 3}
    label = {"original": ("Original", "human"), "ebt": ("EBT", "ours"),
             "llama": ("Llama", "baseline"), "gpt2": ("GPT-2", "baseline")}
    ung = {"remi": [], "ant": []}
    seen = defaultdict(int)
    for t in sorted(pool["unguided"], key=lambda t: (t["tok"], t["prompt"], t["ukey"])):
        tok, pid = t["tok"], t["prompt"]
        opts = sorted(t["options"], key=lambda m: order[clips[m]["system"]])
        seen[(tok, pid)] += 1
        b = {"type": "clips", "tok": tok, "prompt": pid, "prompt_name": name(tok, pid, opts[0]),
             "clips": [clip_spec(m, clips[m], *label[clips[m]["system"]]) for m in opts]}
        if seen[(tok, pid)] > 1:
            b["subtitle"] = "same prompt, another set of picked continuations"
        ung[t["tok"]].append(b)

    # ------------------------------------------------------------------ steered
    # change trials: (reference, test) pairs; group by (tok, attr, prompt) -> system -> picks
    tree = defaultdict(lambda: defaultdict(lambda: {"ref": None, "picks": []}))
    for t in pool["change"]:
        c = clips[t["test"]]
        slot = tree[(c["tok"], c["attribute"], c["prompt"])][c["system"]]
        slot["ref"] = t["reference"]
        slot["picks"].append(t["test"])

    def prompt_card(tok, attr, pid, systems):
        lab, _, key, mlabel, unit, down, up = ATTR[attr]
        rows = []
        for s in [s for s in sv.SYSTEM_ORDER if s in systems]:
            title, sub, _, _ = SYSTEMS[s]
            slot = systems[s]
            picks = sorted(set(slot["picks"]), key=lambda m: (clips[m]["sd"], float(clips[m]["strength"])))
            specs = [clip_spec(slot["ref"], {**clips[slot["ref"]], "condition": "unguided"}, "Unguided")]
            for m in picks:
                c = clips[m]
                arrow = "▲" if c["sd"] > 0 else "▼"
                specs.append(clip_spec(m, c, f"{arrow} {up if c['sd'] > 0 else down} · {fmt_sd(c['sd'])} · "
                                             f"{strength_text(s, c['strength'])}",
                                       "ours" if s == "ebt" else "baseline"))
            rows.append({"type": "clips", "tok": tok, "prompt": pid, "title": title, "subtitle": sub,
                         "metric": key, "metric_label": mlabel, "metric_unit": unit, "clips": specs})
        return {"type": "group", "title": name(tok, pid),
                "subtitle": f"prompt {pid} · each row: that system's own unguided sample, then its picked "
                            f"versions (▲ {up}, ▼ {down}); arrows next to the metric compare with that unguided sample",
                "blocks": rows}

    single_tabs = []
    for tok in ("remi", "ant"):
        attr_tabs = []
        for attr in ("velocity", "duration", "pitch_register"):
            lab, sublab = ATTR[attr][:2]
            prompts = sorted(p for (t, a, p) in tree if t == tok and a == attr)
            blocks = []
            if tok == "ant" and attr == "velocity":
                blocks = [ANT_VELOCITY_NOTE]
            else:
                if tok == "ant":
                    blocks.append(TILT_NOTE)
                blocks += [prompt_card(tok, attr, p, tree[(tok, attr, p)]) for p in prompts]
            attr_tabs.append({"label": lab, "sublabel": sublab, "blocks": blocks})
        single_tabs.append({"label": TOK[tok], "tabs": attr_tabs})

    sel = json.loads(bs.SELECTION.read_text())
    secs = {s["id"]: s for s in sel["sections"]}
    secs["unguided"]["tabs"] = [{"label": TOK[t], "blocks": ung[t]} for t in ("remi", "ant")]
    secs["unguided"]["intro"] = (
        "Each model gets the same short prompt (shaded) and continues it, with no control. All models are small "
        "and trained on GigaMIDI, once per tokenizer; Llama and GPT-2 are trained with the same recipe as the EBT "
        "(same data, context length and number of training tokens). REMI encodes music as bar/position/pitch/"
        "velocity/duration tokens; Anticipation encodes (time, duration, note) triplets. The EBT generates each "
        "token by minimising a learned energy with a few gradient steps; Llama and GPT-2 are standard next-token "
        "predictors. Every continuation here was picked by ear.")
    single = secs["single"]
    single["tabs"] = single_tabs
    for b in single.get("blocks", []):   # the SD / strength explainer: strengths are now shown as numbers
        if b.get("type") == "note" and "<b>Strength</b>" in b.get("text", ""):
            b["text"] = b["text"].replace(
                "<b>Strength</b> (low / medium / high)",
                "<b>Strength</b> (λ for the EBT; a strength or N for the baselines)")
    single["intro"] = (
        "With a small regressor that predicts a musical attribute, generation can be steered toward a target "
        "without retraining. For the EBT, the regressor's error is added to the energy it already minimises, and "
        "λ sets how hard it pushes. The baselines steer Llama and GPT-2 (same training recipe as the EBT) with "
        "PPLM, Tilt or Best-of-N. Targets are given in standard deviations (SD) of the corpus, relative to each "
        "system's own unguided level. Every steered version here was picked by ear, at whatever strength and "
        "target sounded best, so several versions of one system can appear.")
    sel["meta"]["disclaimer"] = (
        "These are selected examples. In Unguided Generation and Single-Attribute Control, every clip was picked "
        "by ear by the author from the final evaluation outputs (the same pool as the listening study): they show "
        "what each system does at its best, not on average. The baselines there are Llama and GPT-2 trained with "
        "the same recipe as the EBT. Combined Control and Pushing Too Hard are not curated: every system gets the "
        "same prompt and its first sample, with the earlier Llama baselines. Leading silence is trimmed in the "
        "picked clips. The aggregate results are in the thesis.")
    bs.SELECTION.write_text(json.dumps(sel, indent=1, ensure_ascii=False) + "\n")
    n_ung = sum(len(b["clips"]) for v in ung.values() for b in v)
    n_st = sum(len(r["clips"]) for t in single_tabs for a in t["tabs"] for g in a["blocks"] for r in g.get("blocks", []))
    print(f"unguided: {sum(len(v) for v in ung.values())} sets, {n_ung} clips; steered cards: {n_st} clips")


if __name__ == "__main__":
    main()
