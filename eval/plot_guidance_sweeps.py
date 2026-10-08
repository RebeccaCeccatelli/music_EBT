"""
Static figures for guidance sweeps (thesis-ready PNGs).

  strength  — per-attribute curves vs guidance strength from sweep tables:
              strict directional accuracy, MAE, bigram_ll, and accuracy split
              by push direction. Rebuilds the figures of
              docs/thesis_findings/2026-09-24_remi_guidance_strength_sweeps.md.
  tradeoff  — controllability vs music quality from score_sweep_quality.py's
              JSON: one panel per attribute, x = OA vs the system's own
              unguided samples (musical change; ~0.93 = ceiling at these n),
              y = strict accuracy, one line per system through its strengths.

Accuracy is strict: a sample whose achieved value equals its baseline (guidance
changed nothing) counts as a miss in both directions — the 2026-09-24 figures
counted such ties as "down" hits (diary 2026-10-05).

Usage:
    python eval/plot_guidance_sweeps.py strength --out_dir docs/thesis_findings/figures/remi \\
        attribute_control/sweep_tables/{velocity,duration,pitch_register}_remi.table.json
    python eval/plot_guidance_sweeps.py tradeoff --out_dir <dir> <score_sweep_quality.json> ...
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attribute_control.aggregate_guidance_sweeps import load_rows

# Reference categorical palette, fixed order (dataviz skill references/palette.md),
# plus a distinct marker per series so identity never relies on color alone.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
ATTR_ORDER = ['velocity', 'duration', 'pitch_register', 'density']


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def _line(ax, i, xs, ys, label):
    ax.plot(xs, ys, color=SERIES[i % len(SERIES)], marker=MARKERS[i % len(MARKERS)],
            lw=2, ms=6, mec='white', mew=1.2, label=label)


def per_strength(rows):
    """attribute -> sorted [(strength, metrics)] for guided rows; strict accuracy."""
    by = defaultdict(lambda: defaultdict(list))
    base_bll = defaultdict(list)
    for r in rows:
        if r.get('target_delta') is not None:
            by[r['attribute']][r['lambda']].append(r)
        elif str(r.get('condition', '')).startswith('baseline') and r.get('bigram_ll') is not None:
            base_bll[r['attribute']].append(r['bigram_ll'])
    out = {}
    for attr, groups in by.items():
        pts = []
        for lam, rs in sorted(groups.items()):
            move = [(r['achieved_value'] - r['baseline_value']) for r in rs]
            up = [m > 0 for m, r in zip(move, rs) if r['target_delta'] > 0]
            down = [m < 0 for m, r in zip(move, rs) if r['target_delta'] < 0]
            bll = [r['bigram_ll'] for r in rs if r.get('bigram_ll') is not None]
            pts.append((lam, {
                'accuracy': (sum(up) + sum(down)) / len(rs),
                'acc_up': sum(up) / len(up) if up else None,
                'acc_down': sum(down) / len(down) if down else None,
                'mae': statistics.mean(abs(r['achieved_value'] - (r['baseline_value'] + r['target_delta']))
                                       for r in rs),
                'bigram_ll': statistics.mean(bll) if bll else None,
            }))
        out[attr] = (pts, statistics.mean(base_bll[attr]) if base_bll[attr] else None)
    return out


def cmd_strength(args):
    rows = [r for t in args.tables for r in load_rows(t)]
    data = per_strength(rows)
    attrs = [a for a in ATTR_ORDER if a in data]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = [('accuracy', 'directional accuracy (strict)', 'guidance_accuracy_vs_lambda.png'),
             ('mae', 'MAE to target (attribute units)', 'guidance_mae_vs_lambda.png'),
             ('bigram_ll', 'bigram log-likelihood (higher = more plausible)', 'guidance_bigram_ll_vs_lambda.png')]
    for key, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for i, a in enumerate(attrs):
            pts, base = data[a]
            xs = [l for l, m in pts if m[key] is not None]
            _line(ax, i, xs, [m[key] for l, m in pts if m[key] is not None], a)
            if key == 'bigram_ll' and base is not None:
                ax.axhline(base, color=SERIES[i], lw=1, ls='--')
        if key == 'accuracy':
            ax.axhline(0.5, color=INK2, lw=1, ls=':')
            ax.set_ylim(0, 1.02)
        _style(ax, 'guidance strength λ', ylabel)
        title = {'accuracy': 'Directional accuracy vs λ (dotted: chance)',
                 'mae': 'Target-tracking error vs λ',
                 'bigram_ll': 'Musicality vs λ (dashed: unguided baseline)'}[key]
        ax.set_title(title, color=INK, loc='left')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(attrs), figsize=(4.2 * len(attrs), 3.8), sharey=True)
    for ax, a in zip(axes if len(attrs) > 1 else [axes], attrs):
        pts, _ = data[a]
        for i, (k, lab) in enumerate((('acc_up', 'push up'), ('acc_down', 'push down'))):
            _line(ax, i, [l for l, m in pts if m[k] is not None], [m[k] for l, m in pts if m[k] is not None], lab)
        ax.axhline(0.5, color=INK2, lw=1, ls=':')
        ax.set_ylim(0, 1.02)
        _style(ax, 'λ', 'accuracy (strict)' if a == attrs[0] else '')
        ax.set_title(a, color=INK, loc='left')
    (axes[0] if len(attrs) > 1 else axes).legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'guidance_accuracy_by_direction.png', dpi=160)
    plt.close(fig)

    for a in attrs:
        pts, base = data[a]
        print(f"\n{a}  (unguided bigram_ll {base:.3f})" if base is not None else f"\n{a}")
        print(f"  {'λ':>7} {'acc':>6} {'up':>6} {'down':>6} {'mae':>8} {'bigram_ll':>10}")
        for l, m in pts:
            f = lambda v, w, p: f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
            print(f"  {l:>7} {f(m['accuracy'], 6, 3)} {f(m['acc_up'], 6, 3)} {f(m['acc_down'], 6, 3)} "
                  f"{f(m['mae'], 8, 4)} {f(m['bigram_ll'], 10, 3)}")
    print(f"\n→ {out}")


def cmd_tradeoff(args):
    systems = {}
    for p in args.results:
        systems.update(json.loads(Path(p).read_text()))
    attrs = [a for a in ATTR_ORDER if any(a in v for v in systems.values())]
    names = sorted(systems)
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.4 * len(attrs), 4), sharey=True)
    axes = axes if len(attrs) > 1 else [axes]
    for ax, a in zip(axes, attrs):
        for i, s in enumerate(names):
            pts = [(float(k), v) for k, v in systems[s].get(a, {}).items()
                   if k != 'none' and v.get('accuracy') is not None and v.get('oa_unguided') is not None]
            if not pts:
                continue
            pts.sort()
            _line(ax, i, [v['oa_unguided'] for _, v in pts], [v['accuracy'] for _, v in pts], s)
        ax.axhline(0.5, color=INK2, lw=1, ls=':')
        ax.set_ylim(0, 1.02)
        ax.invert_xaxis()  # left → right = more musical change
        _style(ax, 'overlap with own unguided output (←less change)', 'directional accuracy (strict)' if a == attrs[0] else '')
        ax.set_title(a, color=INK, loc='left')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Controllability vs musical change, by method', color=INK, x=0.01, ha='left')
    fig.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'guidance_tradeoff.png', dpi=160)
    plt.close(fig)
    print(f"→ {out / 'guidance_tradeoff.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('strength')
    s.add_argument('tables', nargs='+')
    s.add_argument('--out_dir', required=True)
    t = sub.add_parser('tradeoff')
    t.add_argument('results', nargs='+')
    t.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    {'strength': cmd_strength, 'tradeoff': cmd_tradeoff}[args.cmd](args)


if __name__ == '__main__':
    main()
