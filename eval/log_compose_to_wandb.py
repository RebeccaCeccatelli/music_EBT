"""
Log finished compositional (multi-attribute) guidance sweeps
(attribute_control/compose_guidance_sweep.py) to wandb as playable audio,
without regenerating anything: one wandb run per sweep directory.

Per run:
  - Media panels, one section per prompt (key "p<id>/..."): the original song,
    one unguided draw, and one panel per joint target (direction combo × size
    in SD, e.g. "velUP_durDN_2SD"). The wandb step is the strength index, so
    the step slider on a panel walks through the steering strengths; the
    caption gives the strength, both attributes' baseline → achieved values
    and ✓/✗ (strict: moved the requested way). Original and unguided clips are
    repeated at every step so they stay visible next to the slider.
  - "compose_grid" table: one row per sample (filterable by prompt,
    direction, size, strength, ✓), with the audio embedded.
  - Per-step scalars (strength, joint / per-attribute accuracy) for charts.

Clips are prompt + continuation, rendered with the showcase renderer
(FluidSynth + MuseScore General, release tail trimmed) and cached as MP3
under <run_dir>/audio_mp3/.

Usage:
    python eval/log_compose_to_wandb.py --run_dirs <dir> [<dir> ...] --workers 8
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo" / "showcase"))
from build_showcase import render_mp3  # noqa: E402

SHORT = {"velocity": "vel", "duration": "dur", "pitch_register": "pitch"}
ARROW = {1: "↑", -1: "↓"}


def _render(job):
    midi, mp3 = job
    if not Path(mp3).exists():
        render_mp3(midi, mp3)
    return mp3


def load_samples(run_dir: Path):
    """sample_id -> merged row: per-attribute values plus strength/combo/condition.
    Attributes come back in the sweep's --attributes order, which is the order of
    each row's "combo" signs."""
    config = json.loads(next(run_dir.glob("*_compose.config.json")).read_text())
    attrs = config["attributes"].split(",")
    samples = {}
    for t in sorted(run_dir.glob("*_compose_*.table.json")):
        d = json.loads(t.read_text())
        for r in (dict(zip(d["columns"], row)) for row in d["data"]):
            a = r["attribute"]
            s = samples.setdefault(r["sample_id"], {
                "sample_id": r["sample_id"], "prompt_id": r["prompt_id"], "condition": r["condition"],
                "model": r["model"], "method": r["method"], "strength": r["lambda"],
                "combo": r["combo"], "attr": {}})
            s["attr"][a] = r
    return samples, attrs


def midi_path(run_dir: Path, s, attrs):
    sid = s["sample_id"]
    if s["condition"].startswith("ground truth"):
        return run_dir / "midi" / "ground_truth" / f"{sid}_ground_truth.mid"
    if s["combo"] is None:
        return run_dir / "midi" / "baseline" / f"{sid}_generated.mid"
    return run_dir / "midi" / attrs[0] / f"{sid}_generated.mid"


def combo_key(combo, attrs):
    """[2, -2] -> 'velUP_durDN_2SD' (panel key) and 'vel↑ dur↓ (2 SD)' (label)."""
    sign = lambda x: 1 if x > 0 else -1
    size = int(abs(combo[0])) if all(abs(c) == abs(combo[0]) for c in combo) else "mixed"
    key = "_".join(f"{SHORT.get(a, a)}{'UP' if sign(c) > 0 else 'DN'}" for a, c in zip(attrs, combo))
    lab = " ".join(f"{SHORT.get(a, a)}{ARROW[sign(c)]}" for a, c in zip(attrs, combo))
    return f"{key}_{size}SD", f"{lab} ({size} SD)", lab, size


def moved_ok(r):
    return (r["achieved_value"] - r["baseline_value"]) * r["target_delta"] > 0


def caption(s, attrs, strength_name):
    parts = []
    if s["combo"] is None:
        head = "original song" if s["condition"].startswith("ground truth") else "unguided (draw 1 of 3)"
        for a in attrs:
            parts.append(f"{SHORT.get(a, a)} {s['attr'][a]['achieved_value']:.3f}")
        return f"p{s['prompt_id']} {head} · " + " · ".join(parts)
    for a, c in zip(attrs, s["combo"]):
        r = s["attr"][a]
        parts.append(f"{SHORT.get(a, a)} {r['baseline_value']:.3f}→{r['achieved_value']:.3f} "
                     f"({c:+g} SD) {'✓' if moved_ok(r) else '✗'}")
    return f"p{s['prompt_id']} {strength_name}={s['strength']:g} · " + " · ".join(parts)


def log_run(run_dir: Path, args):
    samples, attrs = load_samples(run_dir)
    config = json.loads(next(run_dir.glob("*_compose.config.json")).read_text())
    summary_f = next(run_dir.glob("*_compose.summary.json"), None)
    first = next(iter(samples.values()))
    model, method = first["model"], first["method"]
    strength_name = "λ" if method == "r3" else method

    audio_dir = run_dir / "audio_mp3"
    audio_dir.mkdir(exist_ok=True)
    jobs = {sid: (str(midi_path(run_dir, s, attrs)), str(audio_dir / f"{sid}.mp3"))
            for sid, s in samples.items()}
    missing = [j[0] for j in jobs.values() if not Path(j[0]).exists()]
    if missing:
        sys.exit(f"{len(missing)} MIDI files missing, e.g. {missing[0]}")
    print(f"{run_dir.name}: rendering {len(jobs)} clips with {args.workers} workers")
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(_render, jobs.values(), chunksize=8))

    run_name = f"compose-{model}_{method}-{'+'.join(attrs)}-remi"
    wandb.init(project=args.wandb_project, name=run_name, job_type="compose_listening",
               group="compose-listening", config={**config, "source_dir": str(run_dir)})
    if summary_f:
        wandb.summary["joint_summary"] = json.loads(summary_f.read_text())

    strengths = sorted({s["strength"] for s in samples.values() if s["combo"] is not None})
    fixed = {}  # (prompt, panel) -> sample, for the original / unguided panels
    guided = defaultdict(dict)  # strength -> {(prompt, panel): sample}
    for s in samples.values():
        pid = s["prompt_id"]
        if s["combo"] is None:
            panel = "0_original" if s["condition"].startswith("ground truth") else "1_unguided"
            fixed[(pid, panel)] = s
        else:
            guided[s["strength"]][(pid, combo_key(s["combo"], attrs)[0])] = s

    audio = lambda s: wandb.Audio(jobs[s["sample_id"]][1], caption=caption(s, attrs, strength_name))
    for step, lam in enumerate(strengths):
        log = {f"p{pid}/{panel}": audio(s) for (pid, panel), s in {**fixed, **guided[lam]}.items()}
        rows = list(guided[lam].values())
        oks = [[moved_ok(s["attr"][a]) for a in attrs] for s in rows]
        log["strength"] = lam
        log["acc/joint"] = sum(all(o) for o in oks) / len(oks)
        for i, a in enumerate(attrs):
            log[f"acc/{a}"] = sum(o[i] for o in oks) / len(oks)
        wandb.log(log, step=step)
        print(f"  step {step}: {strength_name}={lam:g}, {len(log)} keys, joint {log['acc/joint']:.3f}")

    cols = ["prompt", "direction", "size_sd", "strength"]
    for a in attrs:
        cols += [f"{a}_baseline", f"{a}_achieved", f"{a}_ok"]
    cols += ["joint_ok", "audio", "sample_id"]
    table = wandb.Table(columns=cols)
    order = lambda s: (s["prompt_id"], s["combo"] is not None, s["strength"] or 0, str(s["combo"]))
    for s in sorted(samples.values(), key=order):
        if s["combo"] is None:
            direction = "original" if s["condition"].startswith("ground truth") else "unguided"
            row = [s["prompt_id"], direction, None, None]
            for a in attrs:
                r = s["attr"][a]
                row += [r["baseline_value"], r["achieved_value"], None]
            row += [None]
        else:
            _, _, lab, size = combo_key(s["combo"], attrs)
            row = [s["prompt_id"], lab, size, s["strength"]]
            ok = []
            for a in attrs:
                r = s["attr"][a]
                ok.append(moved_ok(r))
                row += [r["baseline_value"], r["achieved_value"], ok[-1]]
            row += [all(ok)]
        table.add_data(*row, audio(s), s["sample_id"])
    wandb.log({"compose_grid": table}, step=len(strengths) - 1)
    url = wandb.run.url
    wandb.finish()
    print(f"  wandb: {url}")
    return url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--wandb_project", default="mus_symb_attr_control")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    urls = [log_run(d, args) for d in args.run_dirs]
    print("\n".join(["", "Runs:"] + urls))


if __name__ == "__main__":
    main()
