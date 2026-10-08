"""
Controllability vs music quality for guidance sweeps: for every
(system, attribute, strength) of one or more sweeps, strict directional
accuracy / MAE (attribute_control/aggregate_guidance_sweeps.py) next to the
MIDI-level quality of the same samples (eval/music_quality.py).

Each sweep is a set of table JSONs plus the MIDI root its samples were saved
under (<root>/<group>/<sample_id>_generated.mid with a matching
<sample_id>_prompt.mid, as written by ar_guidance_sweep.py and
listen_density_sweep.py --save_midi_dir). Only the continuation is scored.
Per-file scores are cached in <root>/quality_scores.csv.

Quality is reported two ways, as mean KDE overlap (OA) over the 8 quality
metrics:
  - OA_real: vs a real-data reference set (eval/dump_reference_midi.py;
    real-vs-real ≈ 0.87-0.89). Absolute, but it also reflects the sweep's
    prompt selection (16 hand-picked prompts are not random real music).
  - OA_ung: vs the same system's own unguided samples on the same prompts
    (every saved baseline draw, <root>/baseline/). This isolates what guidance
    changes: ~1 = musically indistinguishable from no guidance.
plus the raw means of the metrics that track "cacophony" and "aimlessness".
The unguided samples themselves are reported as strength "none".

Fairness additions (see docs/diary/2026-10-08.md):
  - acc_lo / acc_hi: 95% bootstrap CI of strict accuracy, resampling prompts
    (the unit that was actually sampled; ~16 prompts per sweep).
  - fwd_per_token: generation compute in forward-pass equivalents per token
    (backward ≈ 2 forwards): tilt 1, PPLM ~1 (its gradient is w.r.t. the
    output logits, through the regressor only), best_of_n N, EBT
    mcmc_steps × 3 (each MCMC step = forward + backward through the model).
  - operating points: per (system, attribute), the strength with the highest
    accuracy among those that stay musically acceptable (harsh dissonance ≤
    real mean + 1 SD, and OA_ung ≥ --min_oa_unguided). One number per method
    instead of whole curves whose strength grids were tuned unevenly.

Usage:
    python eval/score_sweep_quality.py \\
        --sweep "<ar_out_dir>/*.table.json::<ar_out_dir>/midi" \\
        --sweep "<ebt_tables_glob>::<listen_midi/jobid>" \\
        --ref ~/orcd/scratch/.../music_quality/reference_remi_256tok/scores.csv \\
        --out comparison_quality.json
"""

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_control.aggregate_guidance_sweeps import aggregate, load_rows
from eval.music_quality import (COHERENCE_METRICS, METRICS, _read_csv, _sample_base,
                                overlapping_area, score_midi)

# Raw means reported next to OA (the rest of METRICS still feed the mean OA).
SHOWN = ['sharp_dissonance', 'scale_consistency', 'pitch_class_entropy',
         'bar_self_similarity', 'key_continuity']


def score_root(root: Path) -> dict[str, dict]:
    """sample_id -> metrics for every generated / ground-truth MIDI under root (cached)."""
    cache = root / 'quality_scores.csv'
    if cache.exists():
        with open(cache) as f:
            return {r['sample_id']: {k: float(v) if v not in ('', None) else np.nan
                                     for k, v in r.items() if k != 'sample_id'}
                    for r in csv.DictReader(f)}
    scores, n_files = {}, 0
    for p in sorted(root.rglob('*.mid')):
        if p.stem.endswith('_prompt'):
            continue
        n_files += 1
        try:
            r = score_midi(p)
        except Exception as e:  # undecodable file: count it as unscored, don't abort
            print(f"  skip {p.relative_to(root)}: {type(e).__name__}: {e}")
            r = None
        if r is not None:
            scores[_sample_base(p)] = r
    fields = ['n_notes'] + METRICS + COHERENCE_METRICS
    with open(cache, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id'] + fields, restval='')
        w.writeheader()
        for sid, r in scores.items():
            w.writerow({'sample_id': sid, **r})
    print(f"  scored {len(scores)}/{n_files} files under {root} → {cache.name}")
    return scores


def _cols(score_rows: list[dict]) -> dict[str, np.ndarray]:
    return {m: np.array([r.get(m, np.nan) for r in score_rows], dtype=float)
            for m in METRICS + COHERENCE_METRICS}


def quality_summary(score_rows: list[dict], ref: dict[str, np.ndarray],
                    unguided: dict[str, np.ndarray] | None) -> dict:
    if not score_rows:
        return {'n_scored': 0}
    cols = _cols(score_rows)
    col = cols.__getitem__
    out = {'n_scored': len(score_rows),
           'oa_real': float(np.nanmean([overlapping_area(col(m), ref[m]) for m in METRICS]))}
    if unguided is not None:
        out['oa_unguided'] = float(np.nanmean([overlapping_area(col(m), unguided[m]) for m in METRICS]))
    for m in SHOWN:
        v = col(m)
        out[m] = float(np.nanmean(v)) if np.isfinite(v).any() else None
    return out


def bootstrap_accuracy(pairs: list[tuple], n_boot: int = 2000, seed: int = 0):
    """95% CI of accuracy from (prompt_id, correct) pairs, resampling prompts."""
    by_prompt = defaultdict(list)
    for pid, ok in pairs:
        by_prompt[pid].append(ok)
    prompts = list(by_prompt)
    if len(prompts) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_boot):
        sample = [by_prompt[prompts[i]] for i in rng.integers(len(prompts), size=len(prompts))]
        flat = [ok for oks in sample for ok in oks]
        accs.append(sum(flat) / len(flat))
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def fwd_per_token(method: str, strength: str, ebt_mcmc_steps: int) -> float | None:
    if method == 'best_of_n':
        return float(strength) if strength != 'none' else 1.0
    if method in ('tilt', 'pplm'):
        return 1.0
    if method == 'r3':  # EBT
        return 3.0 * ebt_mcmc_steps
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep', action='append', required=True,
                    help='"<table glob>::<midi root>" (repeatable)')
    ap.add_argument('--ref', required=True, help='scores.csv of a real-data reference set')
    ap.add_argument('--out', default=None)
    ap.add_argument('--min_oa_unguided', type=float, default=0.75,
                    help='operating-point constraint: overlap with own unguided output')
    ap.add_argument('--ebt_mcmc_steps', type=int, default=2,
                    help='MCMC steps per token at generation (final EBT checkpoints: 2)')
    args = ap.parse_args()
    ref = _read_csv(Path(args.ref))

    rows, scores, unguided_by_root = [], {}, {}
    for spec in args.sweep:
        tables_glob, root = spec.split('::')
        tables = sorted(glob.glob(tables_glob))
        if not tables:
            sys.exit(f"no tables match {tables_glob}")
        root = Path(root).expanduser()
        root_scores = score_root(root)
        # Every saved unguided draw (tables may list only the first per prompt).
        base_ids = {_sample_base(p) for p in (root / 'baseline').glob('*_generated.mid')}
        unguided = [root_scores[i] for i in sorted(base_ids) if i in root_scores]
        for t in tables:
            for r in load_rows(t):
                r['_scores'] = root_scores.get(r.get('sample_id'))
                r['_root'] = str(root)
                rows.append(r)
        unguided_by_root[str(root)] = unguided
        scores.update(root_scores)

    acc = aggregate(rows)  # strict accuracy / MAE, same code as for the tables alone
    groups, root_of, correct = defaultdict(list), {}, defaultdict(list)
    for r in rows:
        system = f"{r['model']}/{r['method']}"
        root_of[(system, r['attribute'])] = r['_root']
        if r.get('target_delta') is not None:
            key = (system, r['attribute'], str(r['lambda']))
            # strict: a sample that didn't move from its baseline is a miss
            correct[key].append((r['prompt_id'],
                                 (r['achieved_value'] - r['baseline_value']) * r['target_delta'] > 0))
            if r['_scores'] is not None:
                groups[key].append(r['_scores'])
    for (system, attr), root in root_of.items():
        groups[(system, attr, 'none')] = unguided_by_root[root]

    result = {}
    for (system, attr, strength), srows in sorted(groups.items()):
        ung = unguided_by_root[root_of[(system, attr)]]
        entry = quality_summary(srows, ref, _cols(ung) if ung else None)
        entry.update(acc.get(system, {}).get(attr, {}).get(strength, {}))
        if (system, attr, strength) in correct:
            entry['acc_lo'], entry['acc_hi'] = bootstrap_accuracy(correct[(system, attr, strength)])
        entry['fwd_per_token'] = fwd_per_token(system.split('/')[-1], strength, args.ebt_mcmc_steps)
        result.setdefault(system, {}).setdefault(attr, {})[strength] = entry

    # Operating point per (system, attribute): best accuracy within the quality budget.
    max_diss = float(np.nanmean(ref['sharp_dissonance']) + np.nanstd(ref['sharp_dissonance']))
    operating = {}
    for system, attrs in result.items():
        for attr, by_s in attrs.items():
            ok = [(s, e) for s, e in by_s.items() if s != 'none' and e.get('accuracy') is not None
                  and e.get('sharp_dissonance') is not None and e['sharp_dissonance'] <= max_diss
                  and (e.get('oa_unguided') or 0) >= args.min_oa_unguided]
            if ok:
                s, e = max(ok, key=lambda se: (se[1]['accuracy'], -float(se[0])))
                operating.setdefault(attr, {})[system] = {'strength': s, **e}
            else:
                operating.setdefault(attr, {})[system] = None

    hdr = (f"  {'strength':>9} {'n':>4} {'acc':>6} {'95% CI':>13} {'mae':>7} {'OA_real':>7} {'OA_ung':>6} {'fwd/tok':>7} "
           + ''.join(f"{m[:10]:>11}" for m in SHOWN))
    for system, attrs in result.items():
        for attr, by_s in attrs.items():
            print(f"\n{system}  {attr}\n{hdr}")
            order = ['none'] + sorted((s for s in by_s if s != 'none'), key=float)
            for s in (s for s in order if s in by_s):
                e = by_s[s]
                f = lambda k, w, p: f"{e[k]:>{w}.{p}f}" if e.get(k) is not None else f"{'-':>{w}}"
                ci = (f"[{e['acc_lo']:.2f}, {e['acc_hi']:.2f}]" if e.get('acc_lo') is not None else '-')
                print(f"  {s:>9} {e['n_scored']:>4} {f('accuracy', 6, 3)} {ci:>13} {f('mae', 7, 4)} "
                      f"{f('oa_real', 7, 2)} {f('oa_unguided', 6, 2)} {f('fwd_per_token', 7, 0)}"
                      + ''.join(f(m, 11, 3) for m in SHOWN))
    print(f"\nOperating points: best strict accuracy with harsh dissonance ≤ {max_diss:.3f} "
          f"(real mean + 1 SD) and OA_ung ≥ {args.min_oa_unguided}")
    for attr, by_sys in operating.items():
        print(f"\n  {attr}")
        for system, e in sorted(by_sys.items()):
            if e is None:
                print(f"    {system:18s} no strength within budget")
                continue
            print(f"    {system:18s} strength {e['strength']:>6}  acc {e['accuracy']:.3f} "
                  f"[{e['acc_lo']:.2f}, {e['acc_hi']:.2f}]  OA_ung {e['oa_unguided']:.2f}  "
                  f"dissonance {e['sharp_dissonance']:.3f}  fwd/tok {e['fwd_per_token']:.0f}")
    if args.out:
        Path(args.out).write_text(json.dumps({**result, '_operating_points': operating,
                                              '_budget': {'max_dissonance': max_diss,
                                                          'min_oa_unguided': args.min_oa_unguided}},
                                             indent=2))
        print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
