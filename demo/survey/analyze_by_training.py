#!/usr/bin/env python3
"""Listening survey: compare participants by musical training (None / Some / Extensive).

    python demo/survey/analyze_by_training.py responses.csv [--no-caco 686e30f1] [--json out.json]

Per group, with 95% bootstrap CIs over participants, plus the Extensive − None
difference (bootstrap within each group):
  - cacophony ticks: share of clip appearances ticked, overall and by clip type
    (original / unguided / steered / broken catch clip)
  - attention checks: catch_unguided pass (valid pools), catch_change "no difference"
  - Part 2: share heard in the requested direction, signed response, "no difference"
  - Part 1: best-worst score of the human original and of EBT (per tokenizer)
  - skip rate and median seconds per trial
Same inclusion as analyze_survey.py (participants with at least one answer).
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_survey import BAD_CATCH_POOLS, load, mean  # noqa: E402
from build_survey_v5 import KEY_PATH  # noqa: E402

GROUPS = {"None": "None", "Some (a few years, hobby)": "Some",
          "Extensive (performer, music degree, producer)": "Extensive"}


def boot_mean(by_pid, n=2000, seed=0):
    """Mean over pooled trials, CI by resampling participants."""
    pids = [p for p in by_pid if by_pid[p]]
    if not pids:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    point = mean([v for p in pids for v in by_pid[p]])
    reps = sorted(mean([v for p in (rng.choice(pids) for _ in pids) for v in by_pid[p]]) for _ in range(n))
    return point, reps[int(0.025 * n)], reps[int(0.975 * n) - 1]


def boot_diff(a, b, n=2000, seed=0):
    """mean(a) − mean(b), each group resampled over its own participants."""
    pa, pb = [p for p in a if a[p]], [p for p in b if b[p]]
    if not pa or not pb:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    pool = lambda d, ps: mean([v for p in ps for v in d[p]])
    point = pool(a, pa) - pool(b, pb)
    reps = sorted(pool(a, [rng.choice(pa) for _ in pa]) - pool(b, [rng.choice(pb) for _ in pb]) for _ in range(n))
    return point, reps[int(0.025 * n)], reps[int(0.975 * n) - 1]


def collect(subs, key, no_caco):
    """{metric: {pid: [values]}} plus {pid: group}."""
    m = defaultdict(lambda: defaultdict(list))
    group = {}
    for s in subs:
        pid = s["id"]
        group[pid] = GROUPS.get(s.get("background", {}).get("training"), "n/a")
        for r in s["responses"]:
            a = r["answer"]
            m["skip rate"][pid].append(bool(a.get("skipped")))
            if a.get("skipped"):
                continue
            if r.get("ms") and r["ms"] < 600_000:
                m["seconds per trial"][pid].append(r["ms"] / 1000)
            if not any(pid.startswith(x) for x in no_caco):
                ticked = set(a.get("cacophonous", []))
                for h in dict.fromkeys(r["order"]):
                    c = key["clips"][h]
                    if r["kind"] == "catch_unguided" and h in key["catch_unguided_expect_worst"]:
                        typ = "broken catch clip"
                    elif c["system"] == "original":
                        typ = "original"
                    elif r["kind"] == "catch_change":
                        continue
                    else:
                        typ = c.get("condition", "unguided")
                    m["caco: all clips"][pid].append(h in ticked)
                    m[f"caco: {typ}"][pid].append(h in ticked)
                    if typ in ("unguided", "steered"):
                        m[f"caco: {typ} {'EBT' if c['system'] == 'ebt' else 'AR'}"][pid].append(h in ticked)
            if r["kind"] == "catch_unguided" and r["pool_index"] not in BAD_CATCH_POOLS:
                m["catch_unguided pass"][pid].append(a.get("worst") in key["catch_unguided_expect_worst"])
            if r["kind"] == "catch_change":
                m["catch_change 'no difference'"][pid].append(a.get("scale") == 0)
            if r["kind"] == "change":
                pool = key["pool"]["change"][r["pool_index"]]
                v = pool["direction"] * a["scale"]
                fam = "EBT" if key["clips"][pool["test"]]["system"] == "ebt" else "AR"
                for tag in ("all", fam):
                    m[f"P2 right direction ({tag})"][pid].append(v > 0)
                    m[f"P2 signed −2..2 ({tag})"][pid].append(v)
                m["P2 'no difference'"][pid].append(v == 0)
            if r["kind"] in ("unguided", "compare"):
                for h in r["order"]:
                    c = key["clips"][h]
                    if r["kind"] == "unguided" and c["system"] in ("original", "ebt"):
                        v = 1 if a.get("best") == h else -1 if a.get("worst") == h else 0
                        m[f"P1 best-worst {c['tok']} {c['system']}"][pid].append(v)
    return m, group


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--no-caco", nargs="*", default=[])
    ap.add_argument("--key", type=Path, default=KEY_PATH)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    key = json.loads(args.key.read_text())
    subs = [s for s in load(args.inputs) if s.get("finished")
            and any(not r["answer"].get("skipped") for r in s["responses"])]
    m, group = collect(subs, key, args.no_caco)
    names = ["None", "Some", "Extensive"]
    counts = {g: sum(v == g for v in group.values()) for g in names}
    print(f"{len(subs)} participants: {counts} (+{sum(v == 'n/a' for v in group.values())} n/a)")
    print(f"{'metric':38s}" + "".join(f"{g + ' (' + str(counts[g]) + ')':>24s}" for g in names) + f"{'Extensive − None':>26s}")
    out = []
    for metric in sorted(m):
        by_g = {g: {p: v for p, v in m[metric].items() if group[p] == g} for g in names}
        row = {g: boot_mean(by_g[g]) for g in names}
        d = boot_diff(by_g["Extensive"], by_g["None"])
        fmt = "{:.0f} [{:.0f}, {:.0f}]" if metric.startswith("seconds") else "{:+.2f} [{:+.2f}, {:+.2f}]"
        cell = lambda t: fmt.format(*t)
        flag = " *" if (d[1] > 0 or d[2] < 0) else ""
        print(f"{metric:38s}" + "".join(f"{cell(row[g]):>24s}" for g in names) + f"{cell(d) + flag:>26s}")
        out.append(dict(metric=metric, **{g: row[g] for g in names}, diff_ext_none=d))
    print("\n* = 95% CI of Extensive − None excludes 0 (no correction for the number of metrics)")
    if args.json:
        args.json.write_text(json.dumps(dict(counts=counts, rows=out), indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
