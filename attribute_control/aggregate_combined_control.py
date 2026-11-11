"""Combined-control results at each method's operating point (thesis Results).

For every system, reads the per-attribute tables of one compose_guidance run,
pairs the two attributes of each guided sample, and reports:
  - joint accuracy (both attributes moved in the requested direction, strict),
    per-attribute accuracy, joint accuracy on the mixed-direction combos;
  - 95% bootstrap CI over prompts (10,000 draws);
  - exact two-sided sign-flip test vs the EBT at its held-out point, paired by
    prompt (per-prompt joint accuracy over the 8 target combos), Holm over all
    comparisons of both tokenizations;
  - mean harsh dissonance of the guided samples and of the same system's
    unguided draws (quality_scores.csv cached in each run's midi/).
Operating points are the single-attribute held-out ones (tune_*.json), plus
compute-matched best-of-4 and the EBT's by-ear points.

Usage: python attribute_control/aggregate_combined_control.py OUT.json
"""
import csv
import itertools
import json
import random
import re
import sys
from pathlib import Path

D = Path("/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/attr_control/compose_guidance")

# (tokenizer, label, run dir, strength key in the tables)
SYSTEMS = [
    ("remi", "EBT (held-out)", "ebt_r3_20261109_114142_28472156", "0.05:0.055"),
    ("remi", "EBT (by ear) 0.03:0.02", "ebt_r3_20261109_114142_28472158", "0.03:0.02"),
    ("remi", "EBT (by ear) 0.04:0.025", "ebt_r3_20261109_114142_28472158", "0.04:0.025"),
    ("remi", "Llama PPLM 32", "llama_pplm_20261109_101003_28462282", "32.0"),
    ("remi", "Llama tilt 1.0", "llama_tilt_20261109_101003_28462283", "1.0"),
    ("remi", "GPT-2 tilt 1.0", "gpt2_tilt_20261109_101014_28462284", "1.0"),
    ("remi", "Llama best-of-16", "llama_best_of_n_20261109_102537_28462285", "16"),
    ("remi", "GPT-2 best-of-16", "gpt2_best_of_n_20261109_103823_28462286", "16"),
    ("remi", "Llama best-of-4", "llama_best_of_n_20261109_102537_28462285", "4"),
    ("remi", "GPT-2 best-of-4", "gpt2_best_of_n_20261109_103823_28462286", "4"),
    ("ant", "EBT (held-out)", "ebt_r3_20261109_114200_28472160", "0.005:0.012"),
    ("ant", "EBT (by ear) 0.005:0.005", "ebt_r3_20261109_120807_28472162", "0.005:0.005"),
    ("ant", "EBT (by ear) 0.005:0.006", "ebt_r3_20261109_120807_28472162", "0.005:0.006"),
    ("ant", "Llama PPLM 16:32", "llama_pplm_20261109_114346_28472164", "16:32"),
    ("ant", "Llama best-of-8", "llama_best_of_n_20261109_102525_28462290", "8"),
    ("ant", "GPT-2 best-of-8", "gpt2_best_of_n_20261109_102525_28462291", "8"),
    ("ant", "Llama best-of-4", "llama_best_of_n_20261109_102525_28462290", "4"),
    ("ant", "GPT-2 best-of-4", "gpt2_best_of_n_20261109_102525_28462291", "4"),
]
REFERENCE = "EBT (held-out)"
TESTED = {"Llama PPLM 32", "Llama tilt 1.0", "GPT-2 tilt 1.0", "Llama best-of-16", "GPT-2 best-of-16",
          "Llama best-of-4", "GPT-2 best-of-4", "Llama PPLM 16:32", "Llama best-of-8", "GPT-2 best-of-8"}


def norm_strength(x):
    s = str(x)
    try:
        return str(float(s))
    except ValueError:
        return s


def load_run(run, strength):
    cfg = json.loads(next((D / run).glob("*config.json")).read_text())
    attrs = cfg["attributes"].split(",")
    rows = {}
    for i, attr in enumerate(attrs):
        t = json.loads(next((D / run).glob(f"*_{attr}.table.json")).read_text())
        cols = t["columns"]
        for r in t["data"]:
            d = dict(zip(cols, r))
            if d.get("combo") is None or norm_strength(d["lambda"]) != norm_strength(strength):
                continue
            ok = (d["achieved_value"] - d["baseline_value"]) * d["combo"][i] > 0
            e = rows.setdefault(d["sample_id"], {"prompt": d["prompt_id"], "combo": tuple(d["combo"]), "ok": {}})
            e["ok"][attr] = ok
    samples = [v for v in rows.values() if len(v["ok"]) == len(attrs)]
    q = {}
    qf = D / run / "midi" / "quality_scores.csv"
    if qf.exists():
        with open(qf) as f:
            for r in csv.DictReader(f):
                try:
                    q[r["sample_id"]] = float(r["sharp_dissonance"])
                except (ValueError, KeyError):
                    pass
    guided_ids = [k for k, v in rows.items() if len(v["ok"]) == len(attrs)]
    harsh_g = [q[k] for k in guided_ids if k in q and q[k] == q[k]]
    # unguided draws are named p<prompt>_r<k> (k = repeat); guided ids carry strength and combo
    harsh_u = [v for k, v in q.items() if re.fullmatch(r"p\d+_r\d+", k) and v == v]
    return attrs, samples, harsh_g, harsh_u


def per_prompt(samples, key):
    by = {}
    for s in samples:
        by.setdefault(s["prompt"], []).append(key(s))
    return {p: sum(v) / len(v) for p, v in by.items()}


def boot_ci(vals, n=10000, seed=0):
    rng = random.Random(seed)
    k = len(vals)
    m = sorted(sum(rng.choice(vals) for _ in range(k)) / k for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


def signflip_p(diffs):
    obs = abs(sum(diffs))
    count = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-12:
            count += 1
    return count / total


def main(out):
    res, joint_pp = [], {}
    for tok, label, run, strength in SYSTEMS:
        attrs, samples, hg, hu = load_run(run, strength)
        jp = per_prompt(samples, lambda s: all(s["ok"].values()))
        joint_pp[(tok, label)] = jp
        mixed = [s for s in samples if s["combo"][0] * s["combo"][1] < 0]
        lo, hi = boot_ci(list(jp.values()))
        r = {
            "tok": tok, "system": label, "run": run, "strength": strength, "n": len(samples),
            "n_prompts": len(jp),
            "joint_accuracy": sum(all(s["ok"].values()) for s in samples) / len(samples),
            "joint_ci": [lo, hi],
            "joint_accuracy_mixed": sum(all(s["ok"].values()) for s in mixed) / max(1, len(mixed)),
            "harsh_guided": sum(hg) / len(hg) if hg else None, "n_harsh_guided": len(hg),
            "harsh_unguided": sum(hu) / len(hu) if hu else None, "n_harsh_unguided": len(hu),
        }
        for a in attrs:
            r[f"acc_{a}"] = sum(s["ok"][a] for s in samples) / len(samples)
        res.append(r)
    tests = []
    for tok, label, *_ in SYSTEMS:
        if label not in TESTED:
            continue
        ref, other = joint_pp[(tok, REFERENCE)], joint_pp[(tok, label)]
        common = sorted(set(ref) & set(other))
        diffs = [other[p] - ref[p] for p in common]
        tests.append({"tok": tok, "system": label, "n_prompts": len(common),
                      "diff_vs_ebt": sum(diffs) / len(diffs), "p": signflip_p(diffs)})
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    m, running = len(tests), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * tests[i]["p"]))
        tests[i]["p_holm"] = running
    Path(out).write_text(json.dumps({"systems": res, "tests_vs_ebt": tests}, indent=1))
    for r in res:
        accs = " ".join(f"{k[4:]}={v:.2f}" for k, v in r.items() if k.startswith("acc_"))
        hg = f"{r['harsh_guided']:.3f}" if r["harsh_guided"] is not None else "-"
        hu = f"{r['harsh_unguided']:.3f}" if r["harsh_unguided"] is not None else "-"
        print(f"{r['tok']:4s} {r['system']:26s} n={r['n']:3d} joint={r['joint_accuracy']:.3f} "
              f"[{r['joint_ci'][0]:.2f},{r['joint_ci'][1]:.2f}] mixed={r['joint_accuracy_mixed']:.3f} {accs} "
              f"harsh {hu}->{hg} (n {r['n_harsh_guided']})")
    for t in tests:
        print(f"{t['tok']:4s} {t['system']:20s} vs EBT: {t['diff_vs_ebt']:+.3f}  p={t['p']:.4f}  Holm={t['p_holm']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
