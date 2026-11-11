"""
Thesis figure: validation loss vs tokens seen, EBT vs the EBT-recipe baselines.

Left (REMI): logged validation loss over training (REMI runs validate on the
full validation set, so the logged curves are comparable). EBT = valid_final_loss
(last MCMC step), baselines = valid_loss. Runs per model come from the
wandb_run_id.txt files of the checkpoint dirs, merged with "latest run wins",
and isolated post-resume artefacts are dropped (eval/plot_tokens_matched_curves.py).

Right (Anticipation): logged Anticipation validation losses are NOT comparable
across runs (each scores a different slice, diary 2026-10-17), so this panel
uses the shared-window evaluations instead (1,000 random windows, context 512):
several checkpoints of each baseline and the EBT's final checkpoint (its
intermediate checkpoints were deleted).

All runs here share context 512 and the effective batch of their EBT, so
tokens = step x batch x 512.

Usage:
    python eval/plot_val_curves_ebtrecipe.py --out figures/val_curves_ebtrecipe.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.plot_tokens_matched_curves import fetch_curve, run_ids

REMI = {  # name: (checkpoint-dir prefix, W&B project, logged key, tokens per step)
    'Tonic (EBT)': ('ebt-symb-small-remi-s1-job23868523', 'mus_symb_ebt_pretrain', 'valid_final_loss', 256 * 512),
    'Llama (EBT recipe)': ('baseline-llama-small-remi-ebtrecipe-job', 'mus_symb_baseline_pretrain', 'valid_loss', 256 * 512),
    'GPT-2 (EBT recipe)': ('baseline-hf-gpt2-small-remi-ebtrecipe-job', 'mus_symb_baseline_pretrain', 'valid_loss', 256 * 512),
}
TONIC_REMI_RUNS = ['ofef9iwj', '35138ouc', 'euk233f2', 'n2gs5f6c', 'nm7ytdln', 'dxopdzcp', 'txnl0pye', '8wffvj5k']
# Shared-window losses (diary 2026-10-17, job 26152811): step -> loss.
ANT = {
    'Tonic (EBT)': {88800: 1.1024},
    'Llama (EBT recipe)': {24300: 1.2121, 48400: 1.1257, 73000: 1.0744, 88500: 1.0563},
    'GPT-2 (EBT recipe)': {25000: 1.2104, 42100: 1.1412, 59700: 1.1014, 76200: 1.0807, 87900: 1.0704},
}
ANT_TOKENS_PER_STEP = 64 * 512
COLORS = {'Tonic (EBT)': '#2a6fdb', 'Llama (EBT recipe)': '#e8603c', 'GPT-2 (EBT recipe)': '#1aa37a'}
MARKERS = {'Tonic (EBT)': 'o', 'Llama (EBT recipe)': 's', 'GPT-2 (EBT recipe)': '^'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import wandb
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    api = wandb.Api(timeout=120)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, (prefix, project, key, tps) in REMI.items():
        # The EBT lineage's early segments no longer have checkpoint dirs: list its runs
        # explicitly (job21508037 steps 0-8,003, 2026-08-28..09-03, then 8wffvj5k = job23868523).
        ids = TONIC_REMI_RUNS if name.startswith('Tonic') else run_ids(prefix)
        print(f"{name}: {len(ids)} runs")
        steps, vals = fetch_curve(api, project, ids, key)
        if name.startswith('Tonic'):
            m = steps <= 33732   # the checkpoint used in the thesis
            steps, vals = steps[m], vals[m]
        tok = steps * tps / 1e9
        a1.plot(tok, vals, color=COLORS[name], lw=1.6, label=name)
        print(f"   last: step {steps[-1]}  loss {vals[-1]:.4f}  ({tok[-1]:.2f}B tokens)")
    a1.set_xlabel('training tokens (billions)')
    a1.set_ylabel('validation loss (nats per token)')
    a1.set_title('REMI (logged, full validation set)', loc='left', fontsize=10)
    a1.set_ylim(0.6, 1.0)
    a1.legend(frameon=False, fontsize=8)

    for name, pts in ANT.items():
        st = sorted(pts)
        tok = np.array(st) * ANT_TOKENS_PER_STEP / 1e9
        a2.plot(tok, [pts[s] for s in st], color=COLORS[name], marker=MARKERS[name], lw=1.4 if len(st) > 1 else 0,
                ms=6, label=name)
    a2.set_xlabel('training tokens (billions)')
    a2.set_title('Anticipation (1,000 shared windows)', loc='left', fontsize=10)
    a2.legend(frameon=False, fontsize=8)
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=160)
    print(f"→ {out}")


if __name__ == '__main__':
    main()
