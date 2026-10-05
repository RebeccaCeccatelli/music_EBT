"""
Music-quality metrics computed on decoded MIDI, compared against real data.

Why this exists: the token-level signals in attribute_control/musicality_metrics.py
(ebt_energy, bigram_ll, repetition_ratio, grammar_violation_rate) say whether a
sequence is well-formed, not whether it sounds musical — samples could score
well there and still be cacophonic on listening. These metrics look at the
music itself (tonality, harmony, rhythm, structure) and are tokenizer-agnostic,
so REMI, Anticipation and the baselines are scored the same way.

None of them is "higher is better". Real music has a typical range for each
(e.g. it repeats, but not endlessly; it is mostly in key, but not 100%), so a
generated set is judged by how closely its distribution matches real GigaMIDI
windows: the overlapping area (OA) of the two kernel-density estimates, as in
Yang & Lerch (2020), "On the evaluation of generative models in music". OA is
1.0 for identical distributions and 0.0 for disjoint ones.

Usage:
    # 1. score a directory of MIDI files (generated, or real reference windows
    #    from eval/dump_reference_midi.py) into a CSV
    python eval/music_quality.py score --midi_dir <dir> --out <name>.csv

    # 2. compare one or more generated sets against the reference set
    python eval/music_quality.py compare --ref ref.csv --gen ebt.csv llama.csv
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

# Interval classes (semitones mod 12) heard as harsh when sounding together:
# minor 2nd / major 7th and the tritone. Major 2nds / minor 7ths are left out —
# they're common in real voicings and would mostly measure jazz-ness, not noise.
SHARP_DISSONANT_ICS = {1, 6, 11}

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
HARMONIC_MINOR_SCALE = [0, 2, 3, 5, 7, 8, 11]

GRID = 4                    # positions per quarter note (16th-note grid)
MIN_NOTES = 8               # fewer pitched notes → metrics are meaningless, skip piece

METRICS = [
    'pitch_class_entropy',  # bits; ~0 = one note, log2(12)=3.58 = all 12 equally
    'scale_consistency',    # fraction of notes in the best-fitting major/minor scale
    'sharp_dissonance',     # fraction of simultaneously-sounding pitch pairs that are m2/M7/tritone
    'polyphony',            # mean number of pitches sounding when anything sounds
    'pitch_range',          # semitones between lowest and highest pitched note
    'empty_beat_rate',      # fraction of beats with nothing sounding
    'groove_consistency',   # 1 - mean Hamming distance between consecutive bars' onset patterns
    'bar_self_similarity',  # mean best Jaccard match of each bar's (position, pitch class) set to an earlier bar
]


def _load_notes(path: Path):
    """Return (pitched_notes, all_onsets, bar_length) in quarter-note units.
    pitched_notes is an (N, 3) array of (start, end, pitch) from non-drum tracks."""
    from symusic import Score
    score = Score(str(path)).to('quarter')

    pitched, onsets = [], []
    for track in score.tracks:
        arr = track.notes.numpy()
        if len(arr['time']) == 0:
            continue
        onsets.append(arr['time'])
        if not track.is_drum:
            pitched.append(np.stack([arr['time'], arr['time'] + arr['duration'],
                                     arr['pitch']], axis=1))
    pitched = np.concatenate(pitched) if pitched else np.zeros((0, 3))
    onsets = np.concatenate(onsets) if onsets else np.zeros(0)

    bar_length = 4.0
    if len(score.time_signatures):
        ts = score.time_signatures[0]
        bar_length = ts.numerator * 4.0 / ts.denominator
    return pitched, onsets, bar_length


def _sounding_grid(pitched: np.ndarray, n_steps: int) -> List[np.ndarray]:
    """Pitches sounding at each 16th-note grid point."""
    starts = np.floor(pitched[:, 0] * GRID).astype(int)
    ends = np.maximum(np.ceil(pitched[:, 1] * GRID).astype(int), starts + 1)
    grid = [[] for _ in range(n_steps)]
    for s, e, p in zip(starts, ends, pitched[:, 2].astype(int)):
        for t in range(max(s, 0), min(e, n_steps)):
            grid[t].append(p)
    return [np.unique(g) for g in grid]


def score_midi(path: Path) -> Dict[str, float] | None:
    pitched, onsets, bar_length = _load_notes(path)
    if len(pitched) < MIN_NOTES:
        return None

    pitches = pitched[:, 2].astype(int)
    pcs = pitches % 12
    out = {'n_notes': len(pitched)}

    # ── Tonality ──────────────────────────────────────────────────────────
    hist = np.bincount(pcs, minlength=12).astype(float)
    p = hist / hist.sum()
    out['pitch_class_entropy'] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    out['scale_consistency'] = max(
        hist[[(root + d) % 12 for d in scale]].sum() / hist.sum()
        for scale in (MAJOR_SCALE, HARMONIC_MINOR_SCALE) for root in range(12))
    out['pitch_range'] = float(pitches.max() - pitches.min())

    # ── Harmony / texture (sampled on the 16th-note grid) ─────────────────
    t0 = min(pitched[:, 0].min(), onsets.min())
    pitched = pitched.copy()
    pitched[:, :2] -= t0
    onsets = onsets - t0
    total = max(pitched[:, 1].max(), onsets.max())
    # +1 so a zero-length note starting exactly at the end still gets a grid slot/bar.
    n_steps = int(math.ceil(total * GRID)) + 1
    grid = _sounding_grid(pitched, n_steps)

    n_pairs = n_harsh = 0
    sounding_counts = []
    for g in grid:
        if len(g) == 0:
            continue
        sounding_counts.append(len(g))
        if len(g) > 1:
            diffs = (g[None, :] - g[:, None])[np.triu_indices(len(g), 1)] % 12
            n_pairs += len(diffs)
            n_harsh += int(np.isin(diffs, list(SHARP_DISSONANT_ICS)).sum())
    out['sharp_dissonance'] = n_harsh / n_pairs if n_pairs else 0.0
    out['polyphony'] = float(np.mean(sounding_counts)) if sounding_counts else 0.0

    beats = [grid[i:i + GRID] for i in range(0, n_steps, GRID)]
    out['empty_beat_rate'] = sum(all(len(g) == 0 for g in b) for b in beats) / max(len(beats), 1)

    # ── Rhythm and structure (per bar) ────────────────────────────────────
    steps_per_bar = int(round(bar_length * GRID))
    n_bars = int(math.ceil(n_steps / steps_per_bar))
    onset_pat = np.zeros((n_bars, steps_per_bar), dtype=bool)
    idx = np.floor(onsets * GRID).astype(int)
    onset_pat[idx // steps_per_bar, idx % steps_per_bar] = True
    if n_bars > 1:
        out['groove_consistency'] = 1.0 - float(np.mean(
            [np.mean(onset_pat[i] != onset_pat[i + 1]) for i in range(n_bars - 1)]))
    else:
        out['groove_consistency'] = float('nan')

    bar_sets = [set() for _ in range(n_bars)]
    p_idx = np.floor(pitched[:, 0] * GRID).astype(int)
    for i, pc in zip(p_idx, pcs):
        bar_sets[i // steps_per_bar].add((i % steps_per_bar, pc))
    sims = []
    for i in range(1, n_bars):
        if not bar_sets[i]:
            continue
        sims.append(max((len(bar_sets[i] & bar_sets[j]) / len(bar_sets[i] | bar_sets[j])
                         for j in range(i) if bar_sets[j]), default=0.0))
    out['bar_self_similarity'] = float(np.mean(sims)) if sims else float('nan')
    return out


def overlapping_area(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap of the two samples' Gaussian KDEs (Yang & Lerch 2020)."""
    from scipy.stats import gaussian_kde
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    if np.ptp(a) == 0 and np.ptp(b) == 0:
        return float(a[0] == b[0])
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    pad = 0.1 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 1000)
    # Add a hair of jitter so a constant-valued sample doesn't make the KDE singular.
    jitter = 1e-6 * (hi - lo + 1e-12)
    ka = gaussian_kde(a + np.random.default_rng(0).normal(0, jitter, len(a)))
    kb = gaussian_kde(b + np.random.default_rng(1).normal(0, jitter, len(b)))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 only has trapz
    return float(trapezoid(np.minimum(ka(xs), kb(xs)), xs))


def _read_csv(path: Path) -> Dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {m: np.array([float(r[m]) for r in rows]) for m in METRICS}


def cmd_score(args):
    paths = sorted(Path(args.midi_dir).glob(args.glob))
    rows, skipped = [], 0
    for p in paths:
        try:
            r = score_midi(p)
        except Exception as e:  # corrupt/undecodable file: report, don't abort the set
            print(f"  skip {p.name}: {type(e).__name__}: {e}")
            r = None
        if r is None:
            skipped += 1
            continue
        rows.append({'file': p.name, **r})
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'n_notes'] + METRICS)
        w.writeheader()
        w.writerows(rows)
    print(f"Scored {len(rows)}/{len(paths)} files ({skipped} skipped: < {MIN_NOTES} "
          f"pitched notes or unreadable) → {args.out}")


def cmd_compare(args):
    ref = _read_csv(Path(args.ref))
    gens = {Path(g).stem: _read_csv(Path(g)) for g in args.gen}
    names = ['reference'] + list(gens)
    print(f"{'metric':22s}" + ''.join(f"{n:>24s}" for n in names))
    for m in METRICS:
        cells = []
        for n in names:
            v = ref[m] if n == 'reference' else gens[n][m]
            v = v[np.isfinite(v)]
            cell = f"{v.mean():.3f}±{v.std():.3f}"
            if n != 'reference':
                cell += f" OA={overlapping_area(v, ref[m][np.isfinite(ref[m])]):.2f}"
            cells.append(cell)
        print(f"{m:22s}" + ''.join(f"{c:>24s}" for c in cells))
    print(f"{'mean OA':22s}{'':>24s}" + ''.join(
        f"{np.nanmean([overlapping_area(gens[n][m], ref[m]) for m in METRICS]):>24.2f}"
        for n in gens))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('--midi_dir', required=True)
    s.add_argument('--glob', default='*.mid')
    s.add_argument('--out', required=True)
    c = sub.add_parser('compare')
    c.add_argument('--ref', required=True)
    c.add_argument('--gen', nargs='+', required=True)
    args = ap.parse_args()
    {'score': cmd_score, 'compare': cmd_compare}[args.cmd](args)


if __name__ == '__main__':
    main()
