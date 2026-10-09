"""
Tests the "guidance scrambles the attribute" explanation for EBT's asymmetric
steering (docs/thesis_findings/2026-10-09_guidance_ebt_vs_ar.md).

Two measurements per (system, attribute, strength):
  1. Entropy of the guided attribute's values in each continuation
     (pitch: 128 MIDI pitches; duration: log2 bins of note length in
     quarters; velocity: 32 bins of 4), normalized by the max entropy of
     the bins, so 1 = uniformly random values. Compared with the system's
     own unguided samples.
  2. The attribute's "random-token value": attributes.py's own compute_*
     applied to tokens drawn uniformly from that attribute's value tokens.
     If strong guidance flattens the attribute-token choice, the achieved
     value should drift toward it, whatever direction the target asked for.

Usage:
    python eval/guidance_mechanism.py --sweep "<tables>::<midi root>" ... \\
        --tokenizer REMI --out_dir <dir> --title REMI
"""

import argparse
import glob
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_control.aggregate_guidance_sweeps import load_rows
from attribute_control.attributes import ATTRIBUTES
from eval.music_quality import _prompt_path, _sample_base
from eval.plot_guidance_sweeps import (ATTR_ORDER, GRID, INK, INK2, MARKERS, SERIES,
                                       _boot_mean, _style)

DUR_BINS = np.arange(-5, 5)  # log2(quarters): 1/32 note … 16 quarters


def continuation_values(path: Path) -> dict[str, np.ndarray]:
    """Pitch / duration (quarters) / velocity of pitched notes after the prompt."""
    from symusic import Score
    score = Score(str(path)).to('quarter')
    cut = -1.0
    pp = _prompt_path(path)
    if pp is not None:
        ps = Score(str(pp)).to('quarter')
        onsets = [n.time for t in ps.tracks for n in t.notes]
        cut = max(onsets) + 1e-6 if onsets else -1.0
    p, d, v = [], [], []
    for track in score.tracks:
        if track.is_drum:
            continue
        a = track.notes.numpy()
        keep = a['time'] > cut
        p.append(a['pitch'][keep]); d.append(a['duration'][keep]); v.append(a['velocity'][keep])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    return {'pitch': cat(p), 'duration': cat(d), 'velocity': cat(v)}


def norm_entropy(vals: np.ndarray, attr: str) -> float:
    if len(vals) < 4:
        return float('nan')
    if attr == 'pitch_register':
        counts, n_bins = np.bincount(vals.astype(int), minlength=128), 128
    elif attr == 'duration':
        b = np.clip(np.round(np.log2(np.maximum(vals, 1e-3))), DUR_BINS[0], DUR_BINS[-1])
        counts, n_bins = np.bincount((b - DUR_BINS[0]).astype(int), minlength=len(DUR_BINS)), len(DUR_BINS)
    else:  # velocity
        counts, n_bins = np.bincount(np.clip(vals.astype(int) // 4, 0, 31), minlength=32), 32
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum() / math.log2(n_bins))


ATTR_FIELD = {'pitch_register': 'pitch', 'duration': 'duration', 'velocity': 'velocity'}


def random_token_value(attr: str, tokenizer: str, n: int = 20000, seed: int = 0) -> float:
    """compute_<attr> on tokens drawn uniformly from the attribute's value tokens."""
    rng = random.Random(seed)
    if tokenizer == 'REMI':
        from attribute_control import attributes as A
        lo, hi = {'velocity': (A.REMI_VELOCITY_MIN_ID, A.REMI_VELOCITY_MAX_ID),
                  'duration': (A.REMI_DURATION_MIN_ID, A.REMI_DURATION_MAX_ID),
                  'pitch_register': (A.REMI_PITCH_REGISTER_MIN_ID, A.REMI_PITCH_REGISTER_MAX_ID)}[attr]
        return ATTRIBUTES[attr]([rng.randint(lo, hi) for _ in range(n)], 'REMI')
    from attribute_control.attributes import _ensure_anticipation_on_path
    _ensure_anticipation_on_path()
    from anticipation.vocab_ant import TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET
    from anticipation.config import MAX_DUR
    toks, t = [], 0
    for _ in range(n // 3):  # (time, duration, note) triplets; instrument 0, so never a drum
        dur = rng.randrange(MAX_DUR) if attr == 'duration' else 50
        pitch = rng.randrange(128) if attr == 'pitch_register' else 60
        toks += [TIME_OFFSET + t, DUR_OFFSET + dur, NOTE_OFFSET + pitch]
        t = min(t + 1, 9999)
    return ATTRIBUTES[attr](toks, 'Anticipation-Arrival-Time')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep', action='append', required=True, help='"<table glob>::<midi root>"')
    ap.add_argument('--tokenizer', required=True, choices=['REMI', 'Anticipation-Arrival-Time'])
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--title', default='')
    args = ap.parse_args()

    # per (system, attr): λ -> {'up': [(pid, achieved)], 'down': [...], 'ent': [(pid, H)]}
    data = defaultdict(lambda: defaultdict(lambda: {'up': [], 'down': [], 'ent': []}))
    unguided_ent, baseline_val = defaultdict(list), defaultdict(list)
    for spec in args.sweep:
        tables_glob, root = spec.split('::')
        root = Path(root).expanduser()
        index = {_sample_base(p): p for p in root.rglob('*_generated.mid')}
        for t in sorted(glob.glob(tables_glob)):
            rows = load_rows(t)
            for r in rows:
                system, attr = f"{r['model']}/{r['method']}", r['attribute']
                sid = r.get('sample_id')
                if r.get('target_delta') is None:
                    if str(r['condition']).startswith('baseline'):
                        baseline_val[(system, attr)].append(r['baseline_value'])
                    continue
                d = data[(system, attr)][r['lambda']]
                d['up' if r['target_delta'] > 0 else 'down'].append((r['prompt_id'], r['achieved_value']))
                if sid in index:
                    try:
                        h = norm_entropy(continuation_values(index[sid])[ATTR_FIELD[attr]], attr)
                    except Exception:
                        h = float('nan')
                    if math.isfinite(h):
                        d['ent'].append((r['prompt_id'], h))
            attr = rows[0]['attribute'] if rows else None
            system = f"{rows[0]['model']}/{rows[0]['method']}" if rows else None
            if attr and not unguided_ent[(system, attr)]:
                for p in sorted((root / 'baseline').glob('*_generated.mid')):
                    try:
                        h = norm_entropy(continuation_values(p)[ATTR_FIELD[attr]], attr)
                    except Exception:
                        continue
                    if math.isfinite(h):
                        unguided_ent[(system, attr)].append(h)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    attrs = [a for a in ATTR_ORDER if any(k[1] == a for k in data)]
    fig, axes = plt.subplots(2, len(attrs), figsize=(4.4 * len(attrs), 7), squeeze=False)
    summary = {}
    for c, attr in enumerate(attrs):
        rand_val = random_token_value(attr, args.tokenizer)
        ebt = ('ebt/r3', attr)
        lams = sorted(data[ebt])
        base = float(np.mean(baseline_val[ebt])) if baseline_val[ebt] else float('nan')

        ax = axes[0][c]
        ax.axhline(base, color=INK2, lw=1.2, ls=':', label='unguided (baseline)')
        ax.axhline(rand_val, color=SERIES[3], lw=1.6, ls='--', label='random-token value')
        for i, direction in enumerate(('up', 'down')):
            pts = [(l, *_boot_mean(data[ebt][l][direction])) for l in lams if data[ebt][l][direction]]
            xs = [p[0] for p in pts]
            ax.plot(xs, [p[1] for p in pts], color=SERIES[i], marker=MARKERS[i], lw=2, ms=6,
                    mec='white', mew=1.2, label=f'push {direction}')
            ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=SERIES[i], alpha=0.15, lw=0)
        ax.set_xscale('log')
        _style(ax, '', 'achieved attribute value' if c == 0 else '')
        ax.set_title(attr, color=INK, loc='left')

        ax = axes[1][c]
        ung = unguided_ent[ebt]
        if ung:
            ax.axhline(float(np.mean(ung)), color=INK2, lw=1.2, ls=':', label='EBT unguided')
        ax.axhline(1.0, color=SERIES[3], lw=1.6, ls='--', label='uniformly random')
        pts = [(l, *_boot_mean(data[ebt][l]['ent'])) for l in lams if data[ebt][l]['ent']]
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], color=SERIES[2], marker='o', lw=2, ms=6, mec='white', mew=1.2,
                label='EBT guided')
        ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=SERIES[2], alpha=0.15, lw=0)
        # AR methods at their strongest setting, for contrast
        for j, (sysname, by_l) in enumerate(sorted((k[0], v) for k, v in data.items()
                                                  if k[1] == attr and k[0] != 'ebt/r3')):
            top = max(by_l)
            if by_l[top]['ent']:
                ax.axhline(_boot_mean(by_l[top]['ent'])[0], color=SERIES[(j + 4) % len(SERIES)], lw=1,
                           alpha=0.8, label=f'{sysname} @ max strength')
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        _style(ax, 'EBT guidance strength λ (log)', 'normalized entropy of the\nattribute\'s values' if c == 0 else '')

        summary[attr] = {
            'baseline_value': base, 'random_token_value': rand_val,
            'easy_direction_predicted': 'up' if rand_val > base else 'down',
            'unguided_entropy': float(np.mean(ung)) if ung else None,
            'by_lambda': {str(l): {'up': _boot_mean(data[ebt][l]['up'])[0],
                                   'down': _boot_mean(data[ebt][l]['down'])[0],
                                   'entropy': _boot_mean(data[ebt][l]['ent'])[0]} for l in lams},
            'ar_entropy_at_max': {k[0]: _boot_mean(v[max(v)]['ent'])[0] for k, v in data.items()
                                  if k[1] == attr and k[0] != 'ebt/r3' and v[max(v)]['ent']},
        }
    axes[0][0].legend(frameon=False, fontsize=8)
    axes[1][0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{args.title}: strong EBT guidance pushes the attribute toward its random-token value "
                 f"and makes its values more random (bands: 95% CI over prompts)",
                 color=INK, x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'guidance_mechanism.png', dpi=160)
    (out / 'guidance_mechanism.json').write_text(json.dumps(summary, indent=2))
    for attr, s in summary.items():
        ent = [v['entropy'] for v in s['by_lambda'].values()]
        print(f"{attr:15s} baseline {s['baseline_value']:.3f}  random-token {s['random_token_value']:.3f}  "
              f"→ predicted easy direction: {s['easy_direction_predicted']:4s} | entropy unguided "
              f"{s['unguided_entropy']:.3f}, EBT λmin {ent[0]:.3f} → λmax {ent[-1]:.3f} | "
              f"AR at max: {', '.join(f'{k} {v:.3f}' for k, v in s['ar_entropy_at_max'].items())}")
    print(f"→ {out / 'guidance_mechanism.png'}")


if __name__ == '__main__':
    main()
