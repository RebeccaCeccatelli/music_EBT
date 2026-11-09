"""
Paired comparison of guidance methods on the same prompts.

score_sweep_quality.py gives each method's strict accuracy with its own 95%
prompt-bootstrap CI. Overlapping per-method CIs are not a test: every method
runs on the same 16 prompts, so the right comparison is paired by prompt.

For each attribute and each other system, at the operating points stored in
score_sweep_quality.py's JSON (`_operating_points`) and, for best-of-N, also at
N=4 (≈ EBT's compute), this reports:
  - per-prompt accuracy difference (other − EBT), averaged over prompts;
  - 95% CI from resampling prompts (10k bootstrap);
  - exact two-sided sign-flip permutation p-value over prompts (2^n_prompts
    flips; null: method labels exchangeable within a prompt).
Caveats: the operating points were picked on these same prompts, and nothing
here covers training-seed variance.

Usage:
    python eval/paired_guidance_stats.py --results final_remi.json \\
        --sweep "<table glob>[::<midi root>]" ... --out paired_remi.json
"""

import argparse
import glob
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_control.aggregate_guidance_sweeps import load_rows

REF = 'ebt/r3'


def per_prompt_accuracy(rows) -> dict:
    """(system, attribute, strength) -> {prompt_id: strict accuracy over that prompt's targets}."""
    hits = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get('target_delta') is None:
            continue
        key = (f"{r['model']}/{r['method']}", r['attribute'], str(r['lambda']))
        hits[key][r['prompt_id']].append((r['achieved_value'] - r['baseline_value']) * r['target_delta'] > 0)
    return {k: {p: float(np.mean(v)) for p, v in by_p.items()} for k, by_p in hits.items()}


def paired(ref: dict, other: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    prompts = sorted(set(ref) & set(other))
    d = np.array([other[p] - ref[p] for p in prompts])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    # Exact sign-flip test: |mean| under all 2^n sign assignments.
    obs = abs(d.mean())
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    p = float((np.abs((flips * d).mean(1)) >= obs - 1e-12).mean())
    return {'n_prompts': len(prompts), 'acc_ref': float(np.mean([ref[q] for q in prompts])),
            'acc_other': float(np.mean([other[q] for q in prompts])), 'diff': float(d.mean()),
            'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'p_signflip': p, 'prompts_other_better': int((d > 0).sum()),
            'prompts_tied': int((d == 0).sum()), 'prompts_ref_better': int((d < 0).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, help="score_sweep_quality.py JSON (for _operating_points)")
    ap.add_argument('--sweep', action='append', required=True,
                    help='"<table glob>[::<midi root>[::<model label>]]", the same specs as score_sweep_quality.py')
    ap.add_argument('--extra_best_of_n', default='4', help='also compare best-of-N at these N (comma list)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.sweep:
        parts = spec.split('::')
        tables = sorted(glob.glob(parts[0]))
        if not tables:
            sys.exit(f"no tables match {spec}")
        for t in tables:
            for r in load_rows(t):
                if len(parts) > 2:  # rename the model, e.g. llama -> llama-ebtrecipe
                    r['model'] = parts[2]
                rows.append(r)
    acc = per_prompt_accuracy(rows)
    ops = json.loads(Path(args.results).read_text())['_operating_points']

    out = []
    for attr, by_sys in ops.items():
        if not by_sys.get(REF):
            continue
        ref_s = by_sys[REF]['strength']
        cmp = [(s, e['strength'], 'operating point') for s, e in by_sys.items() if s != REF and e]
        cmp += [(s, n, f'N={n}') for s in by_sys if s.endswith('best_of_n')
                for n in args.extra_best_of_n.split(',') if (s, attr, n) in acc]
        print(f"\n{attr}  (EBT at λ {ref_s}: acc {np.mean(list(acc[(REF, attr, ref_s)].values())):.3f})")
        for system, strength, what in cmp:
            r = paired(acc[(REF, attr, ref_s)], acc[(system, attr, strength)])
            r.update(attribute=attr, ref=REF, ref_strength=ref_s, other=system, other_strength=strength, kind=what)
            out.append(r)
            print(f"  {system:16s} {what:16s} acc {r['acc_other']:.3f}  diff {r['diff']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  p={r['p_signflip']:.3f}  prompts "
                  f"other>EBT {r['prompts_other_better']}, tie {r['prompts_tied']}, EBT>other {r['prompts_ref_better']}")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"→ {args.out}")


if __name__ == '__main__':
    main()
