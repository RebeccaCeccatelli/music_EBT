#!/usr/bin/env python3
"""Analyse listening-survey responses.

    python demo/survey/analyze_survey.py responses.csv [more.json ...] [--keep-failed-catch]

Inputs: the Google Sheet exported as CSV (the `json` column holds one
submission per row) and/or JSON files downloaded in test mode. The key
(hash -> system) is read from build_survey.KEY_PATH.

Reports, with 95% bootstrap CIs over participants:
  Part 1  best-worst score per system ((#best - #worst) / #times shown), per tokenizer
  Part 2  perceived change per tokenizer x attribute x system:
          signed response in the requested direction (-2..+2), share heard in
          the right direction, share "no difference"
  Part 3  best-worst score per system among steered versions
Skipped questions are ignored. Participants who fail an attention check are
excluded unless --keep-failed-catch.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_survey import KEY_PATH  # noqa: E402

csv.field_size_limit(10**8)


def load(paths):
    subs = []
    for p in map(Path, paths):
        if p.suffix == ".csv":
            with open(p, newline="") as f:
                subs += [json.loads(r["json"]) for r in csv.DictReader(f) if r.get("json")]
        else:
            subs.append(json.loads(p.read_text()))
    seen, out = set(), []
    for s in subs:   # keep the last submission per participant
        if s["id"] in seen:
            out = [o for o in out if o["id"] != s["id"]]
        seen.add(s["id"])
        out.append(s)
    return out


def boot(values_by_pid, stat, n=2000, seed=0):
    pids = list(values_by_pid)
    if not pids:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    point = stat([v for p in pids for v in values_by_pid[p]])
    reps = sorted(stat([v for p in (rng.choice(pids) for _ in pids) for v in values_by_pid[p]])
                  for _ in range(n))
    return point, reps[int(0.025 * n)], reps[int(0.975 * n) - 1]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def passed_catch(sub, key):
    ok = True
    for r in sub["responses"]:
        if r["answer"].get("skipped"):
            continue   # a skipped check is neither passed nor failed
        if r["kind"] == "catch_unguided":
            ok &= r["answer"].get("worst") in key["catch_unguided_expect_worst"]
        if r["kind"] == "catch_change":
            ok &= r["answer"].get("scale") == 0
    return ok


def best_worst(subs, key, kinds, group):
    """{group: {system: {pid: [+1 best / -1 worst / 0 shown]}}}"""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for s in subs:
        for r in s["responses"]:
            if r["kind"] not in kinds or r["answer"].get("skipped"):
                continue
            for hsh in r["order"]:
                k = key["clips"][hsh]
                v = 1 if r["answer"]["best"] == hsh else -1 if r["answer"]["worst"] == hsh else 0
                out[group(k)][k["system"]][s["id"]].append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--keep-failed-catch", action="store_true")
    args = ap.parse_args()
    key = json.loads(KEY_PATH.read_text())
    subs = [s for s in load(args.inputs) if s.get("finished")]
    failed = [s["id"] for s in subs if not passed_catch(s, key)]
    if not args.keep_failed_catch:
        subs = [s for s in subs if s["id"] not in failed]
    print(f"{len(subs)} participants analysed ({len(failed)} failed an attention check"
          f"{', kept' if args.keep_failed_catch else ', excluded'})")
    bg = defaultdict(int)
    for s in subs:
        bg[s.get("background", {}).get("training", "n/a")] += 1
    print("training:", dict(bg))
    n_all = sum(len(s["responses"]) for s in subs)
    n_skip = sum(r["answer"].get("skipped", False) for s in subs for r in s["responses"])
    print(f"skipped: {n_skip} of {n_all} answers")

    print("\n== Part 1: unguided, best-worst score (−1..+1)")
    for tok, systems in sorted(best_worst(subs, key, {"unguided"}, lambda k: k["tok"]).items()):
        for system, by_pid in sorted(systems.items(), key=lambda kv: -mean([v for x in kv[1].values() for v in x])):
            m, lo, hi = boot(by_pid, mean)
            print(f"  {tok:5s} {system:10s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  n={sum(map(len, by_pid.values()))}")

    print("\n== Part 2: perceived change in the requested direction")
    cells = defaultdict(lambda: defaultdict(list))
    for s in subs:
        for r in s["responses"]:
            if r["kind"] != "change" or r["answer"].get("skipped"):
                continue
            t = key["clips"][r["order"][1]]
            sign = 1 if t["sd"] > 0 else -1
            cells[(t["tok"], t["attribute"], t["system"])][s["id"]].append(sign * r["answer"]["scale"])
    print(f"  {'tok':5s} {'attribute':15s} {'system':16s} {'signed (−2..2)':>22s} {'right dir.':>8s} {'no diff.':>8s}  n")
    for (tok, attr, system), by_pid in sorted(cells.items()):
        m, lo, hi = boot(by_pid, mean)
        right = mean([v > 0 for x in by_pid.values() for v in x])
        none = mean([v == 0 for x in by_pid.values() for v in x])
        n = sum(map(len, by_pid.values()))
        print(f"  {tok:5s} {attr:15s} {system:16s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]   {right:8.2f} {none:8.2f}  {n}")

    print("\n== Part 3: steered versions, best-worst score (−1..+1)")
    for (tok, attr), systems in sorted(best_worst(subs, key, {"compare"}, lambda k: (k["tok"], k["attribute"])).items()):
        for system, by_pid in sorted(systems.items(), key=lambda kv: -mean([v for x in kv[1].values() for v in x])):
            m, lo, hi = boot(by_pid, mean)
            print(f"  {tok:5s} {attr:15s} {system:16s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  n={sum(map(len, by_pid.values()))}")


if __name__ == "__main__":
    main()
