"""
Thesis versions of the guidance trade-off and mechanism figures, redrawn from
cached results with readable labels (no recomputation).

Inputs (docs/thesis_findings/figures/):
  guidance_heldout/final_{remi,ant_ar}_ebtrecipe.json   test-prompt curves (+ 95% CIs)
  guidance_heldout/tune_{remi,ant_ar}.json              operating points (tuning prompts)
  guidance_final_ebtrecipe/{remi,ant}/guidance_mechanism.json   means per strength

Outputs: eth-mit-master-thesis/figures/results/{tradeoff,mechanism}_{remi,ant}.pdf

    ~/.conda/envs/music_EBT/bin/python eval/plot_thesis_guidance.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'docs/thesis_findings/figures'
OUT = ROOT / 'eth-mit-master-thesis/figures/results'

WIDTH = 5.9  # inches = \textwidth (15 cm)
plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 8.5, 'legend.fontsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.titleweight': 'bold',
    'pdf.fonttype': 42,
})
INK2, GRID = '#52514e', '#e4e3df'
TONIC = '#6b4fd8'
METHOD_COLOR = {'pplm': '#2a78d6', 'tilt': '#1baf7a', 'best_of_n': '#eda100'}
METHOD_NAME = {'pplm': 'PPLM', 'tilt': 'tilting', 'best_of_n': 'best-of-$N$'}
ATTR_NAME = {'velocity': 'Velocity', 'duration': 'Duration', 'pitch_register': 'Pitch register'}
ATTR_ORDER = ['velocity', 'duration', 'pitch_register']


def style(sysname):
    """Label, colour, line style and marker for a system key such as 'llama-ebtrecipe/pplm'."""
    if sysname == 'ebt/r3':
        return dict(label='Tonic', color=TONIC, ls='-', marker='o', mfc=TONIC, lw=2.2, z=5)
    model, method = sysname.split('/')
    llama = model.startswith('llama')
    c = METHOD_COLOR[method]
    return dict(label=f"{'Llama' if llama else 'GPT-2'} · {METHOD_NAME[method]}", color=c,
                ls='-' if llama else '--', marker='s' if llama else '^',
                mfc=c if llama else 'white', lw=1.4, z=3)


def grid(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def keep(s):
    return s == 'ebt/r3' or 'ebtrecipe' in s


def order(s):
    if s == 'ebt/r3':
        return (0, '')
    model, method = s.split('/')
    return (1 + list(METHOD_COLOR).index(method), 0 if model.startswith('llama') else 1)


def tradeoff(final, tune, out):
    d = json.loads((FIG / 'guidance_heldout' / final).read_text())
    ops = json.loads((FIG / 'guidance_heldout' / tune).read_text())['_operating_points']
    systems = sorted((s for s in d if not s.startswith('_') and keep(s)), key=order)
    attrs = [a for a in ATTR_ORDER if any(a in d[s] for s in systems)]
    fig, axes = plt.subplots(1, len(attrs), figsize=(WIDTH, 2.9), sharey=True, squeeze=False, layout='constrained')
    for ax, a in zip(axes[0], attrs):
        for s in systems:
            st = style(s)
            pts = sorted((float(k), v) for k, v in d[s].get(a, {}).items()
                         if k != 'none' and v.get('accuracy') is not None and v.get('oa_unguided') is not None)
            if not pts:
                continue
            xs = [v['oa_unguided'] for _, v in pts]
            ys = [v['accuracy'] for _, v in pts]
            ax.errorbar(xs, ys, yerr=[[y - v['acc_lo'] for y, (_, v) in zip(ys, pts)],
                                      [v['acc_hi'] - y for y, (_, v) in zip(ys, pts)]],
                        fmt='none', ecolor=st['color'], elinewidth=0.8, alpha=0.45, capsize=1.5, zorder=st['z'] - 1)
            ax.plot(xs, ys, color=st['color'], ls=st['ls'], lw=st['lw'], marker=st['marker'], ms=4,
                    mfc=st['mfc'], mec=st['color'], mew=1, zorder=st['z'])
            op = ops.get(a, {}).get(s)
            if op:  # operating point chosen on the tuning prompts, shown on the test curve
                v = d[s][a].get(op['strength'])
                if v and v.get('accuracy') is not None:
                    ax.plot([v['oa_unguided']], [v['accuracy']], marker='o', ms=11, mfc='none',
                            mec=st['color'], mew=1.6, zorder=st['z'] + 1)
        ax.axhline(0.5, color=INK2, lw=0.9, ls=':')
        ax.set_ylim(0, 1.03)
        ax.invert_xaxis()
        ax.set_title(ATTR_NAME[a], loc='left')
        grid(ax)
    axes[0][0].set_ylabel('directional accuracy (strict)')
    fig.supxlabel('overlap with the unguided output   (→ the music changes more)', fontsize=8.5)
    handles = []
    for s in systems:
        st = style(s)
        handles.append(Line2D([], [], color=st['color'], ls=st['ls'], lw=st['lw'], marker=st['marker'],
                              ms=4, mfc=st['mfc'], mec=st['color'], label=st['label']))
    handles += [Line2D([], [], ls='none', marker='o', ms=9, mfc='none', mec=INK2, mew=1.4,
                       label='operating point'),
                Line2D([], [], color=INK2, ls=':', lw=0.9, label='chance (0.5)')]
    fig.legend(handles=handles, loc='outside upper center', ncol=len(handles) // 2, frameon=False,
               columnspacing=1.6, handlelength=2.2)
    fig.savefig(out)
    plt.close(fig)
    print('→', out)


def mechanism(src, out):
    d = json.loads((FIG / 'guidance_final_ebtrecipe' / src / 'guidance_mechanism.json').read_text())
    attrs = [a for a in ATTR_ORDER if a in d]
    fig, axes = plt.subplots(2, len(attrs), figsize=(WIDTH, 4.6), squeeze=False, sharex='col', layout='constrained')
    up_c, down_c = '#d9572b', '#2a78d6'
    for c, a in enumerate(attrs):
        v = d[a]
        lams = sorted(v['by_lambda'], key=float)
        xs = [float(l) for l in lams]
        ax = axes[0][c]
        ax.axhline(v['baseline_value'], color=INK2, lw=1, ls=':')
        ax.axhline(v['random_token_value'], color='#eda100', lw=1.6, ls='--')
        ax.plot(xs, [v['by_lambda'][l]['up'] for l in lams], color=up_c, marker='^', ms=4, lw=1.8)
        ax.plot(xs, [v['by_lambda'][l]['down'] for l in lams], color=down_c, marker='v', ms=4, lw=1.8)
        ax.set_xscale('log')
        ax.set_title(ATTR_NAME[a], loc='left')
        grid(ax)

        ax = axes[1][c]
        ax.axhline(1.0, color='#eda100', lw=1.6, ls='--')
        if v.get('unguided_entropy') is not None:
            ax.axhline(v['unguided_entropy'], color=INK2, lw=1, ls=':')
        ar = list(v.get('ar_entropy_at_max', {}).values())
        if ar:
            ax.axhspan(min(ar), max(ar) if max(ar) > min(ar) + 0.01 else min(ar) + 0.01,
                       color='#9a9ab4', alpha=0.35, lw=0)
        ax.plot(xs, [v['by_lambda'][l]['entropy'] for l in lams], color=TONIC, marker='o', ms=4, lw=2)
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        grid(ax)
    axes[0][0].set_ylabel('achieved attribute value')
    axes[1][0].set_ylabel('spread of the values\n(normalised entropy)')
    top = [Line2D([], [], color=up_c, marker='^', ms=4, lw=1.8, label='asked for more (louder / longer / higher)'),
           Line2D([], [], color=down_c, marker='v', ms=4, lw=1.8, label='asked for less'),
           Line2D([], [], color=INK2, ls=':', lw=1, label='unguided'),
           Line2D([], [], color='#eda100', ls='--', lw=1.6, label='uniformly random tokens')]
    bottom = [Line2D([], [], color=TONIC, marker='o', ms=4, lw=2, label='Tonic, guided'),
              plt.Rectangle((0, 0), 1, 1, color='#9a9ab4', alpha=0.35, lw=0,
                            label='Llama / GPT-2 methods at their strongest setting')]
    fig.supxlabel('Tonic guidance strength λ (log scale)', fontsize=8.5)
    fig.legend(handles=top + bottom, loc='outside upper center', ncol=2, frameon=False, columnspacing=1.5)
    fig.savefig(out)
    plt.close(fig)
    print('→', out)


if __name__ == '__main__':
    tradeoff('final_remi_ebtrecipe.json', 'tune_remi.json', OUT / 'tradeoff_remi.pdf')
    tradeoff('final_ant_ar_ebtrecipe.json', 'tune_ant_ar.json', OUT / 'tradeoff_ant.pdf')
    mechanism('remi', OUT / 'mechanism_remi.pdf')
    mechanism('ant', OUT / 'mechanism_ant.pdf')
