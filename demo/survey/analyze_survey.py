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
  Ticks   share of appearances marked "cacophonous", per system and condition
Skipped questions are ignored. Exclusion (--catch):
  none    keep everyone who answered at least one trial (default)
  strict  also drop participants who fail catch_unguided on a valid pool
          (pool 1, prompt 15922, is a bad item; catch_change is never used
          because of the pre-2026-11-09 playback bug)
--no-caco ID ... drops those participants' cacophony ticks (e.g. 686e30f1,
who ticked every trial). --exclude ID ... drops participants entirely.
--key picks the key (default: v5). --json writes the numbers for figures.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_survey_v5 import KEY_PATH  # noqa: E402

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


BAD_CATCH_POOLS = {1}   # catch_unguided pool 1 (remi 15922): the other clips are near-identical


def passed_catch(sub, key):
    """catch_unguided only, valid pools only; skipped or bad-pool checks count as passed."""
    ok = True
    for r in sub["responses"]:
        if r["answer"].get("skipped"):
            continue
        if r["kind"] == "catch_unguided" and r["pool_index"] not in BAD_CATCH_POOLS:
            ok &= r["answer"].get("worst") in key["catch_unguided_expect_worst"]
    return ok


def catch_report(subs, key):
    for kind in ("catch_unguided", "catch_change"):
        rs = [(r, s["id"]) for s in subs for r in s["responses"] if r["kind"] == kind]
        ans = [r for r, _ in rs if not r["answer"].get("skipped")]
        if kind == "catch_unguided":
            for label, sel in (("all pools", ans), ("valid pools", [r for r in ans if r["pool_index"] not in BAD_CATCH_POOLS])):
                ok = sum(r["answer"].get("worst") in key["catch_unguided_expect_worst"] for r in sel)
                print(f"  {kind}, {label}: {ok}/{len(sel)} passed ({len(rs) - len(ans)} skipped)")
        else:
            c = defaultdict(int)
            for r in ans:
                c[r["answer"].get("scale")] += 1
            print(f"  {kind}: {c.get(0, 0)}/{len(ans)} 'no difference', answers {dict(sorted(c.items()))} ({len(rs) - len(ans)} skipped)")


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
    ap.add_argument("--catch", choices=["none", "strict"], default="none")
    ap.add_argument("--exclude", nargs="*", default=[], help="participant id prefixes to drop")
    ap.add_argument("--no-caco", nargs="*", default=[], help="participant id prefixes whose cacophony ticks are dropped")
    ap.add_argument("--key", type=Path, default=KEY_PATH)
    ap.add_argument("--json", type=Path, help="write all numbers here")
    args = ap.parse_args()
    key = json.loads(args.key.read_text())
    out = {}
    subs = [s for s in load(args.inputs) if s.get("finished")]
    answered = lambda s: any(not r["answer"].get("skipped") for r in s["responses"])
    empty = [s["id"] for s in subs if not answered(s)]
    subs = [s for s in subs if answered(s) and not any(s["id"].startswith(x) for x in args.exclude)]
    print(f"{len(subs) + len(empty)} finished submissions; {len(empty)} with no answers dropped {empty}")
    print("== Attention checks (before exclusion)")
    catch_report(subs, key)
    failed = [s["id"] for s in subs if not passed_catch(s, key)]
    if args.catch == "strict":
        subs = [s for s in subs if s["id"] not in failed]
    print(f"{len(subs)} participants analysed ({len(failed)} fail catch_unguided on a valid pool, "
          f"{'excluded' if args.catch == 'strict' else 'kept'})")
    out["n"] = len(subs)
    for field in ("training", "device"):
        c = defaultdict(int)
        for s in subs:
            c[s.get("background", {}).get(field) or "n/a"] += 1
        print(f"{field}:", dict(c))
        out[field] = dict(c)
    langs = defaultdict(int)
    for s in subs:
        langs[s.get("lang", "en")] += 1
    print("language:", dict(langs))
    out["lang"] = dict(langs)
    n_all = sum(len(s["responses"]) for s in subs)
    n_skip = sum(r["answer"].get("skipped", False) for s in subs for r in s["responses"])
    print(f"skipped: {n_skip} of {n_all} answers")
    trial_s = sorted(r["ms"] / 1000 for s in subs for r in s["responses"] if r.get("ms") and r["ms"] < 600_000)
    if trial_s:
        print(f"seconds per trial (< 10 min): median {trial_s[len(trial_s) // 2]:.0f}, "
              f"IQR {trial_s[len(trial_s) // 4]:.0f}-{trial_s[3 * len(trial_s) // 4]:.0f}")
        per_p = sorted(sum(r["ms"] for r in s["responses"] if r.get("ms") and r["ms"] < 600_000) / 60000 for s in subs)
        print(f"minutes per participant (active trials): median {per_p[len(per_p) // 2]:.0f}, range {per_p[0]:.0f}-{per_p[-1]:.0f}")

    def report_bw(title, tag, kinds, group):
        print(f"\n== {title}, best-worst score (−1..+1)")
        out[tag] = []
        for g, systems in sorted(best_worst(subs, key, kinds, group).items()):
            for system, by_pid in sorted(systems.items(), key=lambda kv: -mean([v for x in kv[1].values() for v in x])):
                m, lo, hi = boot(by_pid, mean)
                vals = [v for x in by_pid.values() for v in x]
                best, worst = mean([v == 1 for v in vals]), mean([v == -1 for v in vals])
                print(f"  {str(g):28s} {system:12s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  best {best:.2f} worst {worst:.2f}  n={len(vals)}")
                out[tag].append(dict(group=g, system=system, score=m, lo=lo, hi=hi, best=best, worst=worst, n=len(vals)))

    report_bw("Part 1: unguided", "part1", {"unguided"}, lambda k: k["tok"])

    print("\n== Part 2: perceived change in the requested direction (signed −2..+2)")
    trials = []   # (pid, tok, attribute, system, signed)
    for s in subs:
        for r in s["responses"]:
            if r["kind"] != "change" or r["answer"].get("skipped"):
                continue
            pool = key["pool"]["change"][r["pool_index"]]
            t = key["clips"][pool["test"]]
            trials.append((s["id"], pool["tok"], pool["attribute"], t["system"], pool["direction"] * r["answer"]["scale"]))
    out["part2"] = []
    groupings = [("overall", lambda t: "all"), ("system family", lambda t: "EBT" if t[3] == "ebt" else "AR baselines"),
                 ("tok x family", lambda t: (t[1], "EBT" if t[3] == "ebt" else "AR")),
                 ("tok x attribute x family", lambda t: (t[1], t[2], "EBT" if t[3] == "ebt" else "AR")),
                 ("tok x attribute x system", lambda t: (t[1], t[2], t[3]))]
    for gname, gf in groupings:
        print(f"  -- by {gname}")
        cells = defaultdict(lambda: defaultdict(list))
        for t in trials:
            cells[gf(t)][t[0]].append(t[4])
        for g, by_pid in sorted(cells.items(), key=lambda kv: str(kv[0])):
            m, lo, hi = boot(by_pid, mean)
            vals = [v for x in by_pid.values() for v in x]
            right, wrong, none = (mean([v > 0 for v in vals]), mean([v < 0 for v in vals]), mean([v == 0 for v in vals]))
            rlo, rhi = boot(by_pid, lambda xs: mean([v > 0 for v in xs]))[1:]
            print(f"  {str(g):42s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  right {right:.2f} [{rlo:.2f}, {rhi:.2f}]  "
                  f"wrong {wrong:.2f}  none {none:.2f}  n={len(vals)} ({len(by_pid)} people)")
            out["part2"].append(dict(grouping=gname, group=g, signed=m, lo=lo, hi=hi, right=right, right_lo=rlo,
                                     right_hi=rhi, wrong=wrong, none=none, n=len(vals), people=len(by_pid)))

    report_bw("Part 3: steered versions by tokenizer", "part3_tok", {"compare"}, lambda k: k["tok"])
    report_bw("Part 3: steered versions by tokenizer x attribute", "part3", {"compare"},
              lambda k: (k["tok"], k["attribute"]))

    print("\n== Optional ticks: share of appearances marked cacophonous")
    caco_subs = [s for s in subs if not any(s["id"].startswith(x) for x in args.no_caco)]
    print(f"  ({len(subs) - len(caco_subs)} participants' ticks dropped)")
    out["caco"] = []
    for (tok, system, cond), by_pid in sorted(cacophony(caco_subs, key).items()):
        m, lo, hi = boot(by_pid, mean)
        n = sum(map(len, by_pid.values()))
        print(f"  {tok:5s} {system:16s} {cond:13s} {m:.2f} [{lo:.2f}, {hi:.2f}]  n={n}")
        out["caco"].append(dict(tok=tok, system=system, condition=cond, share=m, lo=lo, hi=hi, n=n))
    print("\n== Cacophony ticks within Part 2 pairs: reference (unguided) vs steered, same trials")
    out["caco_part2"] = []
    pairs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for s in caco_subs:
        for r in s["responses"]:
            if r["kind"] != "change" or r["answer"].get("skipped"):
                continue
            pool = key["pool"]["change"][r["pool_index"]]
            sysname = key["clips"][pool["test"]]["system"]
            ticked = set(r["answer"].get("cacophonous", []))
            pairs[(pool["tok"], sysname)]["reference"][s["id"]].append(pool["reference"] in ticked)
            pairs[(pool["tok"], sysname)]["steered"][s["id"]].append(pool["test"] in ticked)
    for (tok, system), d in sorted(pairs.items()):
        row = dict(tok=tok, system=system)
        for cond in ("reference", "steered"):
            m, lo, hi = boot(d[cond], mean)
            row[cond] = dict(share=m, lo=lo, hi=hi, n=sum(map(len, d[cond].values())))
        print(f"  {tok:5s} {system:12s} reference {row['reference']['share']:.2f}  steered {row['steered']['share']:.2f} "
              f"[{row['steered']['lo']:.2f}, {row['steered']['hi']:.2f}]  n={row['steered']['n']}")
        out["caco_part2"].append(row)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str))
        print(f"\nwrote {args.json}")


def cacophony(subs, key):
    """Share of appearances in which a clip was ticked as cacophonous,
    by tokenizer x system x condition (unguided / steered / original / catch)."""
    out = defaultdict(lambda: defaultdict(list))
    for s in subs:
        for r in s["responses"]:
            if r["answer"].get("skipped"):
                continue
            ticked = set(r["answer"].get("cacophonous", []))
            for hsh in dict.fromkeys(r["order"]):
                k = key["clips"][hsh]
                cond = r["kind"] if r["kind"].startswith("catch") else k.get("condition", "original")
                out[(k["tok"], k["system"], cond)][s["id"]].append(hsh in ticked)
    return out


if __name__ == "__main__":
    main()
