"""Clip Library: one local page with every rendered clip from the sweeps, for listening and collecting.

    python demo/library/build_library.py          # writes music_EBT_logs/survey/library/ and ~/transfer/clip_library.zip

Sources (already rendered, nothing is re-rendered here):
  survey/candidates/  manifest.json + mp3/   unguided clips (test/tune prompts, every sweep draw, 100 songs)
  survey/guided/      manifest.json + mp3/   every guided clip of the final sweeps (all strengths x ±0.5/1/2 SD + ref)
The candidates' attribute/catch sections duplicate guided clips, so only their unguided sections are used.
Starting collection = the picks saved from the four picker pages (survey/picks_v5/*.json).
"""

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "showcase"))
import prompt_names  # noqa: E402

SURVEY = Path.home() / "orcd/scratch/rebcecca/music_EBT_logs/survey"
OUT = SURVEY / "library"
SITE = Path(__file__).parent / "site"
ZIP = Path.home() / "transfer/clip_library.zip"

SYS_ORDER = ["ebt", "ebt_l002", "llama_pplm", "llama_tilt", "llama_bon", "gpt2_tilt", "gpt2_bon"]
SYS_LABEL = {"ebt": "EBT", "llama_pplm": "Llama PPLM", "llama_tilt": "Llama tilt", "llama_bon": "Llama best-of-N",
             "gpt2_tilt": "GPT-2 tilt", "gpt2_bon": "GPT-2 best-of-N"}
SCALE = {"ebt": "λ", "llama_pplm": "strength", "llama_tilt": "strength", "gpt2_tilt": "strength",
         "llama_bon": "N", "gpt2_bon": "N"}
UNG_SYS = ["original", "ebt", "llama", "gpt2"]


def instruments(path):
    progs = prompt_names.features(path)["programs"]
    total = sum(progs.values()) or 1
    names = []
    for p, c in progs.most_common():
        n = "bass" if 32 <= p <= 39 else prompt_names._lead_name(p)
        if n not in names and c >= 0.05 * total:
            names.append(n)
    return ", ".join(names[:4])


def main():
    cand = json.load(open(SURVEY / "candidates/manifest.json"))["clips"]
    guided = json.load(open(SURVEY / "guided/manifest.json"))["clips"]

    # prompt names from the original (ground-truth) MIDI of each prompt
    prompts = {}   # key "tok|set|pid" -> info
    for x in cand:
        if x["system"] != "original":
            continue
        pset = "song" if x["section"] == "unguided_songs" else x.get("split") or "test"
        key = f'{x["tok"]}|{pset}|{x["prompt"]}'
        try:
            name, inst = prompt_names.describe(x["midi"]), instruments(x["midi"])
        except Exception as e:  # unreadable MIDI: keep the id only
            name, inst = f"Prompt {x['prompt']}", ""
            print("name failed", key, e)
        prompts[key] = {"tok": x["tok"], "set": pset, "pid": x["prompt"], "name": name, "inst": inst}

    clips = {}   # id -> [audio path, seconds, prompt_end]

    # unguided: tok|set|pid -> system -> [[id, label]]
    ung = defaultdict(lambda: defaultdict(list))
    # the Part 1 sample (cond "unguided") is draw 1 of some sweep: name it after its sibling draws' sweep
    sweep_of = {str(Path(x["midi"]).parent): x["label"].split(" · ")[0]
                for x in cand if x["section"] == "unguided_more" and x.get("label")}
    for x in cand:
        if not x["section"].startswith("unguided"):
            continue
        pset = "song" if x["section"] == "unguided_songs" else x.get("split") or "test"
        key = f'{x["tok"]}|{pset}|{x["prompt"]}'
        clips[x["id"]] = ["u/" + x["id"] + ".mp3", x["seconds"], x["prompt_end"]]
        if x["system"] == "original":
            label = "original"
        else:
            label = {"song": "continuation"}.get(x["cond"], x.get("label") or x["cond"])
            if x["cond"] == "unguided":
                label = sweep_of.get(str(Path(x["midi"]).parent), "sweep") + " · draw 1"
        ung[key][x["system"]].append([x["id"], label])
    for key in ung:
        for lst in ung[key].values():
            lst.sort(key=lambda c: c[1])

    # guided: tok|set|pid -> attr -> system -> {"ref": id, "grid": {strength: {sd: id}}}
    gd = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"ref": None, "grid": defaultdict(dict)})))
    for x in guided:
        key = f'{x["tok"]}|{x["split"]}|{x["prompt"]}'
        clips[x["id"]] = ["g/" + x["id"] + ".mp3", x["seconds"], x["prompt_end"]]
        cell = gd[key][x["attr"]][x["system"]]
        if x["strength"] == "ref":
            cell["ref"] = x["id"]
        else:
            cell["grid"][x["strength"]][f'{x["sd"]:+g}'] = x["id"]

    for key in list(ung) + list(gd):
        if key not in prompts:
            tok, pset, pid = key.split("|")
            prompts[key] = {"tok": tok, "set": pset, "pid": int(pid), "name": f"Prompt {pid}", "inst": ""}

    picks = []
    for n in ["unguided", "remi", "ant", "explorer"]:
        for i in json.load(open(SURVEY / f"picks_v5/{n}.json"))["ids"]:
            i = i.split("~", 1)[-1]
            if i in clips and i not in picks:
                picks.append(i)

    # audio files must exist
    missing = [i for i, c in clips.items()
               if not (SURVEY / ("guided/mp3" if c[0][0] == "g" else "candidates/mp3") / c[0][2:]).exists()]
    for i in missing:
        del clips[i]
    print(f"{len(clips)} clips, {len(missing)} without audio, {len(prompts)} prompts, {len(picks)} starting picks")

    catalog = {"prompts": prompts, "unguided": ung, "guided": gd, "clips": clips, "picks": picks,
               "sys_label": SYS_LABEL, "scale": SCALE, "sys_order": SYS_ORDER, "ung_sys": UNG_SYS}
    OUT.mkdir(exist_ok=True)
    for f in ["index.html", "app.js", "style.css"]:
        shutil.copy(SITE / f, OUT / f)
    (OUT / "catalog.js").write_text("window.CATALOG = " + json.dumps(catalog, separators=(",", ":")) + ";\n")
    (OUT / "audio").mkdir(exist_ok=True)
    for link, target in [("u", SURVEY / "candidates/mp3"), ("g", SURVEY / "guided/mp3")]:
        if not (OUT / "audio" / link).is_symlink():
            os.symlink(target, OUT / "audio" / link)

    if "--no-zip" not in sys.argv:
        ZIP.unlink(missing_ok=True)
        # store (mp3s don't compress); zip follows the audio symlinks; top folder "clip_library/"
        stage = OUT.parent / "_libzip"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir()
        os.symlink(OUT, stage / "clip_library")
        subprocess.run(["nice", "ionice", "-c3", "zip", "-q", "-r", "-0", str(ZIP), "clip_library"],
                       cwd=stage, check=True)
        shutil.rmtree(stage)
        print("wrote", ZIP, f"{ZIP.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
