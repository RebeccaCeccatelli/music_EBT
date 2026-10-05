"""
Aggregate guidance-sweep tables — EBT's (listen_density_sweep.py, saved as
wandb Table JSON in attribute_control/sweep_tables/) and the AR baselines'
(ar_guidance_sweep.py, same layout) — into per-(system, attribute, strength)
metrics, so EBT and every AR method are scored by one piece of code:

  - accuracy: fraction of guided samples whose achieved value moved off the
    prompt's own baseline in the direction the target asked for
  - mae:      mean |achieved - (baseline + target_delta)|
  - bigram_ll / grammar_violation_rate: mean musicality of the guided samples

Plus each system's unguided reference (baseline rows) for bigram_ll, so a
musicality cost can be read as a drop from that system's own baseline.

Usage:
    python attribute_control/aggregate_guidance_sweeps.py \
        attribute_control/sweep_tables/velocity_remi.table.json \
        <ar_out_dir>/llama_tilt_velocity.table.json ... \
        --out attribute_control/sweep_tables/comparison.json

An EBT table has no method/model/attribute columns; those are taken from its
filename (<attribute>_remi.table.json -> system "ebt").
"""

import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

_ATTRS = ("velocity", "duration", "pitch_register", "density")


def load_rows(path):
    t = json.loads(Path(path).read_text())
    cols = t["columns"]
    rows = [dict(zip(cols, r)) for r in t["data"]]
    if "method" not in cols:  # EBT sweep table
        attr = next((a for a in _ATTRS if Path(path).name.startswith(a + "_")), None)
        if attr is None:
            raise ValueError(f"Can't infer the attribute of EBT table {path} from its name")
        for r in rows:
            r["method"], r["model"], r["attribute"] = "r3", "ebt", attr
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def aggregate(rows):
    guided = defaultdict(list)
    baseline = defaultdict(list)
    for r in rows:
        system = f"{r['model']}/{r['method']}"
        if r.get("target_delta") is not None:
            guided[(system, r["attribute"], r["lambda"])].append(r)
        elif str(r.get("condition", "")).startswith("baseline"):
            baseline[(system, r["attribute"])].append(r)

    out = {}
    for (system, attr, strength), rs in sorted(guided.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        correct = [
            (r["achieved_value"] - r["baseline_value"]) * r["target_delta"] > 0
            for r in rs
        ]
        base = baseline.get((system, attr), [])
        out.setdefault(system, {}).setdefault(attr, {})[str(strength)] = {
            "n": len(rs),
            "accuracy": sum(correct) / len(correct),
            "mae": _mean(abs(r["achieved_value"] - (r["baseline_value"] + r["target_delta"]))
                         for r in rs),
            "bigram_ll": _mean(r.get("bigram_ll") for r in rs),
            "grammar_violation_rate": _mean(r.get("grammar_violation_rate") for r in rs),
            "baseline_bigram_ll": _mean(r.get("bigram_ll") for r in base),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tables", nargs="+")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = [r for t in args.tables for r in load_rows(t)]
    agg = aggregate(rows)
    for system, attrs in agg.items():
        for attr, by_strength in attrs.items():
            print(f"\n{system}  {attr}")
            print(f"  {'strength':>9} {'n':>4} {'acc':>6} {'mae':>8} {'bigram_ll':>10} {'gvr':>7}")
            for s, m in by_strength.items():
                bll = f"{m['bigram_ll']:.3f}" if m["bigram_ll"] is not None else "-"
                gvr = (f"{m['grammar_violation_rate']:.4f}"
                       if m["grammar_violation_rate"] is not None else "-")
                print(f"  {s:>9} {m['n']:>4} {m['accuracy']:>6.3f} {m['mae']:>8.4f} {bll:>10} {gvr:>7}")
    if args.out:
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
