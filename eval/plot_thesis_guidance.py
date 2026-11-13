"""
Thesis versions of the guidance trade-off and mechanism figures, redrawn from
cached results with readable labels (no recomputation).

Inputs (docs/thesis_findings/figures/):
  guidance_heldout/final_{remi,ant_ar}_ebtrecipe.json   test-prompt curves (+ 95% CIs)
  guidance_heldout/tune_{remi,ant_ar}.json              operating points (tuning prompts)
  guidance_final_ebtrecipe/{remi,ant}/guidance_mechanism.json   means per strength

  steering_remi_duration.json   progress by direction and dissonance per strength
                                 (built once from the sweep tables and cached MIDI scores)

Outputs: eth-mit-master-thesis/figures/results/{tradeoff,mechanism}_{remi,ant}.pdf,
         steering_remi_duration.pdf

    ~/.conda/envs/music_EBT/bin/python eval/plot_thesis_guidance.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
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


def tradeoff(final, tune, out, notes=None):
    """One full-width row per attribute: accuracy against how much the music changed
    (1 - overlap with the same model's unguided output), one line per method through
    its strengths. Error bars only at the operating points, to keep the curves legible."""
    d = json.loads((FIG / 'guidance_heldout' / final).read_text())
    ops = json.loads((FIG / 'guidance_heldout' / tune).read_text())['_operating_points']
    # GPT-2 behaves like Llama under every method; leaving it out halves the clutter.
    systems = sorted((s for s in d if not s.startswith('_') and keep(s) and not s.startswith('gpt2')), key=order)
    attrs = [a for a in ATTR_ORDER if any(a in d[s] for s in systems)]
    fig, axes = plt.subplots(len(attrs), 1, figsize=(WIDTH, 1.75 * len(attrs) + 0.9), squeeze=False,
                             layout='constrained')
    for ax, a in zip(axes[:, 0], attrs):
        xmin, xmax = 1, 0
        for s in systems:
            st = style(s)
            pts = sorted((float(k), v) for k, v in d[s].get(a, {}).items()
                         if k != 'none' and v.get('accuracy') is not None and v.get('oa_unguided') is not None)
            if not pts:
                continue
            xs = [1 - v['oa_unguided'] for _, v in pts]
            ys = [v['accuracy'] for _, v in pts]
            xmin, xmax = min(xmin, min(xs)), max(xmax, max(xs))
            tonic = s == 'ebt/r3'
            ax.plot(xs, ys, color=st['color'], ls=st['ls'], lw=st['lw'] if tonic else 1.2,
                    marker=st['marker'], ms=4 if tonic else 3.2, mfc=st['mfc'], mec=st['color'], mew=1,
                    alpha=1 if tonic else 0.85, zorder=st['z'])
            op = ops.get(a, {}).get(s)
            v = d[s][a].get(op['strength']) if op else None
            if v and v.get('accuracy') is not None:
                x, y = 1 - v['oa_unguided'], v['accuracy']
                ax.errorbar([x], [y], yerr=[[y - v['acc_lo']], [v['acc_hi'] - y]], fmt='none',
                            ecolor=st['color'], elinewidth=1, capsize=2, zorder=st['z'] + 1)
                ax.plot([x], [y], marker='o', ms=10, mfc='none', mec=st['color'], mew=1.6,
                        zorder=st['z'] + 2)
        ax.axhline(0.5, color=INK2, lw=0.9, ls=':')
        ax.set_ylim(0.3, 1.04)
        pad = 0.03 * (xmax - xmin)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_title(ATTR_NAME[a], loc='left')
        ax.set_ylabel('accuracy')
        if notes and a in notes:
            ax.text(0.98, 0.08, notes[a], transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=7.5, color=TONIC, style='italic')
        grid(ax)
    axes[-1][0].set_xlabel('how much the music changed  (1 − overlap with the same model\'s unguided output)')
    handles = []
    for s in systems:
        st = style(s)
        handles.append(Line2D([], [], color=st['color'], ls=st['ls'], lw=st['lw'] if s == 'ebt/r3' else 1.2,
                              marker=st['marker'], ms=4, mfc=st['mfc'], mec=st['color'], label=st['label']))
    handles += [Line2D([], [], ls='none', marker='o', ms=9, mfc='none', mec=INK2, mew=1.4,
                       label='operating point (± 95% CI)'),
                Line2D([], [], color=INK2, ls=':', lw=0.9, label='chance (0.5)')]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, frameon=False,
               columnspacing=1.3, handlelength=2.0, fontsize=7.5)
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
    axes[0][0].set_ylabel('value reached')
    axes[1][0].set_ylabel('how varied the values are\n(0 = all equal, 1 = random)')
    top = [Line2D([], [], color=up_c, marker='^', ms=4, lw=1.8, label='Tonic asked for more (louder / longer / higher)'),
           Line2D([], [], color=down_c, marker='v', ms=4, lw=1.8, label='Tonic asked for less'),
           Line2D([], [], color=INK2, ls=':', lw=1, label='Tonic without guidance'),
           Line2D([], [], color='#eda100', ls='--', lw=1.6, label='if the tokens were picked at random')]
    bottom = [Line2D([], [], color=TONIC, marker='o', ms=4, lw=2, label='Tonic, guided'),
              plt.Rectangle((0, 0), 1, 1, color='#9a9ab4', alpha=0.35, lw=0,
                            label='AR methods at their strongest setting')]
    fig.supxlabel('Tonic guidance strength λ (log scale)', fontsize=8.5)
    fig.legend(handles=top + bottom, loc='outside upper center', ncol=2, frameon=False, columnspacing=1.5)
    fig.savefig(out)
    plt.close(fig)
    print('→', out)


# ── Figure: progress towards the target and dissonance per strength (REMI duration) ──
STEER_JSON = FIG / 'steering_remi_duration.json'
STEER_SYSTEMS = ['ebt/r3', 'llama-ebtrecipe/pplm', 'llama-ebtrecipe/tilt']
STEER_XLABEL = {'ebt/r3': 'strength λ', 'llama-ebtrecipe/pplm': 'step norm',
                'llama-ebtrecipe/tilt': 'tilt fraction $s$'}
STEER_TICKS = {'ebt/r3': [0.005, 0.01, 0.02, 0.05], 'llama-ebtrecipe/pplm': [1, 2, 4, 8, 16, 32],
               'llama-ebtrecipe/tilt': [0.05, 0.1, 0.2, 0.5, 1]}
RANDOM_DISSONANCE = 0.246  # uniformly random notes (synthetic check, 2026-10-05)


def steering_cache():
    """Aggregate the sweep tables once (means and 95% bootstrap CIs over prompts)."""
    if STEER_JSON.exists():
        return json.loads(STEER_JSON.read_text())
    import sys
    sys.path.insert(0, str(ROOT))
    from eval.plot_guidance_sweeps import collect_steering, _boot_mean
    logs = Path.home() / 'orcd/scratch/rebcecca/music_EBT_logs'
    mq, lm, ar = logs / 'music_quality', logs / 'attr_control/listen_midi', logs / 'attr_control/ar_guidance'
    sweeps = [f"{mq}/sweeps/ebt_remi_tables/duration_remi.table.json::{lm}/25280947"] + [
        f"{ar}/{run}/*.table.json::{ar}/{run}/midi::llama-ebtrecipe"
        for run in ('llama-ebtrecipe_tilt_20261019', 'llama-ebtrecipe_pplm_20261104')]
    data, real_mean, real_sd = collect_steering(sweeps, mq / 'reference_remi_256tok/scores.csv')
    out = {'_real_dissonance': [real_mean, real_sd]}
    for sname in STEER_SYSTEMS:
        d = data[(sname, 'duration')]
        lams = sorted(d['progress'])
        out[sname] = {
            'strength': lams,
            'up': [_boot_mean(d['progress'][l]['up']) for l in lams],
            'down': [_boot_mean(d['progress'][l]['down']) for l in lams],
            'dissonance': [_boot_mean(d['diss'][l]) for l in lams],
            'unguided_dissonance': sum(d['unguided']) / len(d['unguided']),
        }
    STEER_JSON.write_text(json.dumps(out, indent=1))
    print('cached', STEER_JSON)
    return out


def steering(out):
    d = steering_cache()
    ops = json.loads((FIG / 'guidance_heldout' / 'tune_remi.json').read_text())['_operating_points']['duration']
    real_mean, real_sd = d['_real_dissonance']
    up_c, down_c = '#d9572b', '#2a78d6'
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.5), sharey='row', layout='constrained',
                             gridspec_kw={'height_ratios': [1.25, 1]})
    for c, sname in enumerate(STEER_SYSTEMS):
        v, st = d[sname], style(sname)
        xs = [float(x) for x in v['strength']]
        op = ops.get(sname, {}).get('strength')

        ax = axes[0][c]
        ax.axhline(1, color=INK2, lw=1, ls='--')
        ax.axhline(0, color=INK2, lw=0.9, ls=':')
        for key, col, mk in (('up', up_c, '^'), ('down', down_c, 'v')):
            m = [p[0] for p in v[key]]
            ax.plot(xs, m, color=col, marker=mk, ms=4, lw=1.8, zorder=3)
            if all(p[1] is not None for p in v[key]):
                ax.fill_between(xs, [p[1] for p in v[key]], [p[2] for p in v[key]],
                                color=col, alpha=0.15, lw=0)
        ax.set_title(st['label'], loc='left', color=st['color'])
        ax.set_ylim(-0.4, 1.9)

        ax = axes[1][c]
        ax.axhspan(max(real_mean - real_sd, 0), real_mean + real_sd, color='#e9e7e1', lw=0)
        ax.axhline(RANDOM_DISSONANCE, color='#eda100', lw=1.4, ls='--')
        ax.axhline(v['unguided_dissonance'], color=INK2, lw=1, ls=':')
        m = [p[0] for p in v['dissonance']]
        ax.plot(xs, m, color=st['color'], marker='o', ms=4, lw=1.8, zorder=3)
        if all(p[1] is not None for p in v['dissonance']):
            ax.fill_between(xs, [p[1] for p in v['dissonance']], [p[2] for p in v['dissonance']],
                            color=st['color'], alpha=0.15, lw=0)
        ax.set_ylim(0, 0.27)
        ax.set_xlabel(STEER_XLABEL[sname] + ' (log scale)')
        for ax in (axes[0][c], axes[1][c]):
            ax.set_xscale('log')
            ax.set_xticks(STEER_TICKS[sname], [f'{t:g}' for t in STEER_TICKS[sname]])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if op:
                ax.axvline(float(op), color='#9a9ab4', lw=1.2, ls='-', alpha=0.7, zorder=1)
            grid(ax)
    axes[0][0].set_ylabel('progress towards the target\n(1 = reached, 0 = no change)')
    axes[1][0].set_ylabel('harsh dissonance\n(share of harsh intervals)')
    handles = [
        Line2D([], [], color=up_c, marker='^', ms=4, lw=1.8, label='longer notes requested'),
        Line2D([], [], color=down_c, marker='v', ms=4, lw=1.8, label='shorter notes requested'),
        Line2D([], [], color=INK2, ls='--', lw=1, label='target reached'),
        Line2D([], [], color='#9a9ab4', lw=1.2, label='operating point'),
        plt.Rectangle((0, 0), 1, 1, color='#e9e7e1', lw=0, label='real music (mean ± 1 SD)'),
        Line2D([], [], color=INK2, ls=':', lw=1, label='same model, unguided'),
        Line2D([], [], color='#eda100', ls='--', lw=1.4, label='uniformly random notes'),
    ]
    fig.legend(handles=handles[:4], loc='outside upper center', ncol=4, frameon=False,
               columnspacing=1.2, handlelength=1.8, fontsize=7.5)
    fig.legend(handles=handles[4:], loc='outside lower center', ncol=3, frameon=False,
               columnspacing=1.2, handlelength=1.8, fontsize=7.5)
    fig.savefig(out)
    plt.close(fig)
    print('→', out)


if __name__ == '__main__':
    tradeoff('final_remi_ebtrecipe.json', 'tune_remi.json', OUT / 'tradeoff_remi.pdf',
             notes={'pitch_register': 'Tonic gains accuracy only by changing the music strongly'})
    tradeoff('final_ant_ar_ebtrecipe.json', 'tune_ant_ar.json', OUT / 'tradeoff_ant.pdf')
    mechanism('remi', OUT / 'mechanism_remi.pdf')
    mechanism('ant', OUT / 'mechanism_ant.pdf')
    steering(OUT / 'steering_remi_duration.pdf')
