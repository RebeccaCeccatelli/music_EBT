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
from prompt_names import describe  # noqa: E402

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
    "velocity": ("Velocity", "piano ↔ forte", "velocity", "mean velocity", "/127", "softer", "louder"),
    "duration": ("Duration", "staccato ↔ legato", "note_beats", "mean note length (beats)", "", "shorter", "longer"),
    "pitch_register": ("Pitch register", "low ↔ high", "pitch", "mean pitch", "", "lower", "higher"),
}
TOK = {"remi": "REMI", "ant": "Anticipation"}
TOK_HINT = {"REMI": "notes on a beat grid", "Anticipation": "notes as timed events"}
TOKEN_NOTE = {
    "type": "note", "title": "REMI or Anticipation?",
    "text": "To learn music, a model reads it as a sequence of symbols (tokens), like a song spelled out with an "
            "alphabet. REMI and Anticipation are two such alphabets, called tokenizations. Every model is "
            "trained once with each."}
TILT_NOTE = {"type": "note", "title": "Tilt: not applicable",
             "text": "Tilt needs a hand-written value map over the attribute's own tokens, which only exists for REMI."}
ATTR_HELP_TEXT = ("An attribute is the musical property being steered: <b>velocity</b> is how loud the notes "
                  "are, <b>duration</b> is how long they are held (short and detached, or long and connected), "
                  "and <b>pitch register</b> is how high or low they sit. Velocity can only be steered with REMI: "
                  "Anticipation has no symbol for loudness.")
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
        """Descriptive name from the original excerpt (prompt_names.describe); instrument list as fallback."""
        path = orig or originals.get((tok, pid))
        return describe(path) if path and Path(path).exists() else bs.prompt_name(tok, pid)

    def instruments(tok, pid, orig=None):
        path = orig or originals.get((tok, pid))
        return bs.prompt_name(tok, pid, path) if path else ""

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
             "subtitle": instruments(tok, pid, opts[0]),
             "clips": [clip_spec(m, clips[m], *label[clips[m]["system"]]) for m in opts]}
        if seen[(tok, pid)] > 1:
            b["subtitle"] += " · same prompt, another set of selected continuations"
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
        """One 'ladder' card: rows = systems, columns = target SD left to right (reference in the middle).
        Only the EBT row has the unguided sample: it is the one reference for the whole card."""
        lab, _, key, mlabel, unit, down, up = ATTR[attr]
        specs = []
        for s in [s for s in sv.SYSTEM_ORDER if s in systems]:
            title, sub, _, _ = SYSTEMS[s]
            slot = systems[s]
            row = {"row": title, "row_sub": sub}
            if s == "ebt":
                specs.append({**clip_spec(slot["ref"], {**clips[slot["ref"]], "condition": "unguided"}, "Unguided"),
                              **row, "col": "ref"})
            lv = levels[(tok, attr, s)]
            for m in sorted(set(slot["picks"]), key=lambda m: (clips[m]["sd"], float(clips[m]["strength"]))):
                c = clips[m]
                specs.append({**clip_spec(m, c, strength_text(s, c["strength"]), "ours" if s == "ebt" else "baseline"),
                              **row, "col": "pick", "sd": c["sd"], "strength": c["strength"],
                              "level": lv.index(float(c["strength"])) + 1, "levels": len(lv)})
        assert specs and specs[0]["col"] == "ref", (tok, attr, pid)  # the reference lives in the EBT row
        return {"type": "ladder", "tok": tok, "prompt": pid, "title": name(tok, pid),
                "prompt_name": instruments(tok, pid) or f"prompt {pid}",
                "subtitle": f"one row per technique: ▼ {down} on the left, ▲ {up} on the right; in each cell choose how much "
                            f"change to ask for and how hard to push · the EBT's unguided sample in the middle is the reference",
                "metric": key, "metric_label": mlabel, "metric_unit": unit, "down": down, "up": up,
                "clips": specs}

    # Strength levels per (tok, attr, system), over all picks in that tab, so a meter reads the same in every card.
    levels = defaultdict(set)
    for (tok, attr, _), systems in tree.items():
        for s, slot in systems.items():
            levels[(tok, attr, s)] |= {float(clips[m]["strength"]) for m in slot["picks"]}
    levels = {k: sorted(v) for k, v in levels.items()}

    single_tabs = []
    for tok in ("remi", "ant"):
        attr_tabs = []
        for attr in ("velocity", "duration", "pitch_register"):
            lab, sublab = ATTR[attr][:2]
            prompts = sorted(p for (t, a, p) in tree if t == tok and a == attr)
            if tok == "ant" and attr == "velocity":
                continue   # Anticipation has no loudness symbol: no velocity tab at all
            blocks = [TILT_NOTE] if tok == "ant" else []
            blocks += [prompt_card(tok, attr, p, tree[(tok, attr, p)]) for p in prompts]
            attr_tabs.append({"label": lab, "sublabel": sublab, "numbered": True, "collapse_after": 5,
                              "blocks": blocks})
        single_tabs.append({"label": TOK[tok], "tabs_label": "Attribute", "tabs_help": ATTR_HELP_TEXT,
                            "tabs": attr_tabs})

    sel = json.loads(bs.SELECTION.read_text())
    secs = {s["id"]: s for s in sel["sections"]}
    secs["unguided"]["tabs"] = [{"label": TOK[t], "collapse_after": 5, "numbered": True, "blocks": ung[t]} for t in ("remi", "ant")]
    secs["unguided"]["intro"] = (
        "Each model gets the same short prompt (shaded) and continues it, with no control. All models are small "
        "and trained on GigaMIDI, once per tokenization; Llama and GPT-2 are trained with the same recipe as the EBT "
        "(same data, context length and number of training tokens). The REMI and Anticipation tabs are two "
        "tokenizations (see the ? next to Tokenization). The EBT generates each "
        "token by minimising a learned energy with a few gradient steps; Llama and GPT-2 are standard next-token "
        "predictors. Every continuation here was selected by ear.")
    single = secs["single"]
    single["tabs"] = single_tabs
    for b in single.get("blocks", []):   # the SD / strength explainer: strengths are now shown as numbers
        if b.get("type") == "note" and "<b>Strength</b>" in b.get("text", ""):
            b["collapsible"] = True   # a "learn more" card, closed until clicked
            b["title"] = "What do “asked for” and “push” mean?"
            b["teaser"] = "How far each clip is pushed, and how hard"
            b["text"] = (
                '<span class="nl"><b>Asked for</b> (the target) = <i>where</i> you ask the music to go: “slightly '
                'louder” (+0.5 SD), “louder” (+1 SD) or “much louder” (+2 SD), and the same downwards. SD (standard '
                'deviation) is how much loudness, note length or pitch usually varies between pieces.</span>'
                '<span class="nl"><b>Push</b> (the strength) = <i>how hard</i> the model is pushed to get there: how '
                'insistently you ask, not what you ask for. Gentle / medium / strong are ranked within each technique, '
                'so “strong” means the strongest setting shown for that technique, not the same force for all. '
                'Hover a label for the exact setting (λ for the EBT, a strength value for PPLM and Tilt). '
                'Best-of-N keeps its number: “best of 8 tries” draws 8 continuations and keeps the closest.</span>'
                '<span class="nl">Think of throwing a ball: the target is where you aim, the strength is how hard '
                'you throw. Too soft and it falls short; too hard and it flies past and goes wild, which is what '
                '“Pushing Too Hard” shows.</span>')
            b["text"] = b["text"].replace(
                "<b>Strength</b> (low / medium / high)",
                "<b>Strength</b> (gentle / medium / strong, ranked within each technique)")
    single["intro"] = (
        "With a small regressor that predicts a musical attribute, generation can be steered toward a target "
        "without retraining. For the EBT, the regressor's error is added to the energy it already minimises, and "
        "its strength, λ, sets how hard it pushes. The baselines steer Llama and GPT-2 (same training recipe as the EBT) with "
        "PPLM, Tilt or Best-of-N. Targets are given in standard deviations (SD) of the corpus, relative to each "
        "system's own unguided level. Every steered version here was selected by ear, at whatever strength and "
        "target sounded best, so several versions of one system can appear. In each card, the EBT's unguided "
        "sample is the one reference to listen against.")
    sel["meta"]["disclaimer"] = (
        "These are selected examples. In Unguided Generation and Single-Attribute Control, every clip was selected "
        "by ear by the author from the final evaluation outputs (the same pool as the listening study): they show "
        "what each system does at its best, not on average. The baselines there are Llama and GPT-2 trained with "
        "the same recipe as the EBT. Combined Control and Pushing Too Hard are not curated: every system gets the "
        "same prompt and its first sample, with the earlier Llama baselines. Leading silence is trimmed in the "
        "selected clips. The aggregate results are in the thesis.")
    # Tokenization: label the REMI / Anticipation tabs everywhere, and explain it once in plain words.
    for sec in sel["sections"]:
        tabs = sec.get("tabs", [])
        if tabs and {t["label"] for t in tabs} <= set(TOK_HINT):
            sec["tabs_label"] = "Tokenization"
            sec["tabs_help"] = TOKEN_NOTE["text"]   # shown in a box behind a "?" next to the label
            for t in tabs:
                t.pop("sublabel", None)
    secs["unguided"]["blocks"] = []
    secs["combined"]["intro"] = secs["combined"]["intro"].replace(
        "combinations (REMI).", "combinations (REMI tokenization only).")
    bs.SELECTION.write_text(json.dumps(sel, indent=1, ensure_ascii=False) + "\n")
    n_ung = sum(len(b["clips"]) for v in ung.values() for b in v)
    n_st = sum(len(g.get("clips", [])) for t in single_tabs for a in t["tabs"] for g in a["blocks"])
    print(f"unguided: {sum(len(v) for v in ung.values())} sets, {n_ung} clips; steered cards: {n_st} clips")


if __name__ == "__main__":
    main()
