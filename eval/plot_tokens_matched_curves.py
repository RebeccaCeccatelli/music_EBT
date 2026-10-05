"""
Validation loss vs *tokens seen* for EBT and the GPT-2/Llama baselines, and
each baseline's loss at the token budgets where the EBT runs stopped.

Why tokens, not steps: the runs differ in effective batch and context length
(docs/training_runs.md), so by the time EBT stopped the baselines had seen
~4-6x more tokens. Comparing final losses alone would mostly measure that.

Runs per model are found from the wandb_run_id.txt files kept in every
checkpoint dir (~/orcd/scratch/.../checkpoints/<run>/). Overlapping steps from
resumes/branches are resolved "latest run wins" (the lineage that was actually
continued). Isolated post-resume validation artifacts are dropped: dips >1.5%
below the median of their ±6 neighbours (the filter used for the 2026-10-05
checkpoint cleanup) and spikes >10% above it, as are the inf sentinels.

For EBT the curve is valid_final_loss (the CE of the last MCMC step, i.e.
the prediction actually sampled from), not valid_loss (the CE averaged over
all MCMC steps). Perplexity is exp() of that loss for every model: EBT's
logged valid_perplexity averages per-batch exp(loss), which reads ~1.5%
higher than exp(mean loss), while the baselines log exp(mean loss).

Caveats printed with the table: the baselines use context 1024 vs EBT's 512
(longer context lowers loss on its own), and per-token loss is not comparable
across tokenizers (vocab 427 vs 55,028), so only compare within a tokenizer.

Usage:
    python eval/plot_tokens_matched_curves.py --out_dir docs/figures/tokens_matched
"""

import argparse
import math
import statistics as st
from pathlib import Path

import numpy as np

ENTITY = "rceccatelli-eth-z-rich"
CKPT_ROOT = Path.home() / "orcd/scratch/rebcecca/music_EBT_logs/checkpoints"

# name: (checkpoint-dir prefix, wandb project, tokens per optimizer step =
# effective batch × context length, from docs/training_runs.md — the wandb run
# configs don't store these)
MODELS = {
    'REMI': {
        'EBT':   ('ebt-symb-small-remi-s1-job',              'mus_symb_ebt_pretrain',      256 * 512),
        'GPT-2': ('baseline-hf-gpt2-small-remi-job',         'mus_symb_baseline_pretrain', 256 * 1024),
        'Llama': ('baseline-llama-small-remi-job',           'mus_symb_baseline_pretrain', 256 * 1024),
    },
    'Anticipation-AT': {
        'EBT':   ('ebt-symb-small-ant-at-full-s1-job',       'mus_symb_ebt_pretrain',      64 * 512),
        'GPT-2': ('baseline-hf-gpt2-small-ant-at-full-job',  'mus_symb_baseline_pretrain', 128 * 1024),
        'Llama': ('baseline-llama-small-ant-at-full-job',    'mus_symb_baseline_pretrain', 128 * 1024),
    },
}


def run_ids(prefix: str) -> list[str]:
    ids = set()
    for d in CKPT_ROOT.glob(prefix + '*'):
        f = d / 'wandb_run_id.txt'
        if f.is_file():
            ids.add(f.read_text().strip())
    return sorted(ids)


def fetch_curve(api, project: str, ids: list[str], key: str) -> tuple[np.ndarray, np.ndarray]:
    """`key` by trainer/global_step, merged over runs (latest run wins)."""
    runs = []
    for rid in ids:
        try:
            runs.append(api.run(f"{ENTITY}/{project}/{rid}"))
        except Exception as e:
            print(f"  ! {project}/{rid}: {type(e).__name__}: {e}")
    runs.sort(key=lambda r: r.created_at)
    by_step = {}
    for r in runs:  # later runs overwrite earlier ones at the same step
        for row in r.scan_history(keys=['trainer/global_step', key], page_size=2000):
            v = row[key]
            v = float(v) if not isinstance(v, str) else float('nan')
            if math.isfinite(v):
                by_step[int(row['trainer/global_step'])] = v
    steps = np.array(sorted(by_step))
    vals = np.array([by_step[s] for s in steps])
    keep = []
    for i in range(len(vals)):
        nb = [vals[j] for j in range(max(0, i - 6), min(len(vals), i + 7)) if j != i]
        med = st.median(nb) if nb else vals[i]
        # Dips: >1.5% below (the checkpoint-cleanup filter). Spikes: >10% above —
        # looser, since loss legitimately falls fast early in training.
        keep.append(0.985 * med <= vals[i] <= 1.10 * med)
    keep = np.array(keep, dtype=bool)
    print(f"    {len(runs)} runs, {len(steps)} val points, {int((~keep).sum())} artifacts dropped")
    return steps[keep], vals[keep]


def loss_at(tokens: np.ndarray, vals: np.ndarray, budget: float) -> float:
    """Smoothed loss at a token budget: median of the readings within ±5% of it."""
    m = (tokens >= 0.95 * budget) & (tokens <= 1.05 * budget)
    if m.any():
        return float(np.median(vals[m]))
    return float('nan') if budget > tokens.max() else float(np.interp(budget, tokens, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='docs/figures/tokens_matched')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import wandb
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    api = wandb.Api(timeout=60)

    for tok, models in MODELS.items():
        curves = {}
        for name, (prefix, project, tps) in models.items():
            ids = run_ids(prefix)
            print(f"{tok} / {name}: {len(ids)} wandb runs from {prefix}*")
            # EBT's valid_loss averages the CE over all MCMC steps; the prediction
            # actually used is the last step's (valid_final_loss), which is the
            # like-for-like counterpart of the baselines' next-token valid_loss.
            key = 'valid_final_loss' if name == 'EBT' else 'valid_loss'
            steps, vals = fetch_curve(api, project, ids, key)
            curves[name] = (steps * tps, vals, steps)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for name, (tokens, vals, _) in curves.items():
            ax.plot(tokens / 1e9, vals, label=name, lw=1.4)
        ebt_end = curves['EBT'][0].max()
        ax.axvline(ebt_end / 1e9, color='grey', ls=':', lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('tokens seen (billions, log)')
        ax.set_ylabel('validation loss (log; EBT: final MCMC step)')
        ax.set_title(f'{tok}: validation loss vs tokens seen')
        ax.legend()
        ax.grid(True, which='both', alpha=0.25)
        fig.tight_layout()
        path = out / f"val_loss_vs_tokens_{tok.lower().replace('-', '_')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        budgets = sorted({0.5e9, 1e9, 2e9, round(ebt_end, -8)})
        print(f"\n{tok} — validation loss at matched token budgets "
              f"(EBT stopped at {ebt_end / 1e9:.2f}B tokens)")
        for label, f in (('loss', lambda x: x), ('perplexity = exp(loss)', math.exp)):
            print(f"  {label}")
            print(f"{'model':8s}" + ''.join(f"{b / 1e9:>9.1f}B" for b in budgets) + f"{'final':>10s}")
            for name, (tokens, vals, steps) in curves.items():
                print(f"{name:8s}" + ''.join(f"{f(loss_at(tokens, vals, b)):10.4f}" for b in budgets)
                      + f"{f(vals[-1]):10.4f}  (step {steps[-1]:,}, {tokens[-1] / 1e9:.1f}B)")
        print(f"→ {path}")
    print("\nCaveats: baselines use context 1024 vs EBT 512 (longer context lowers loss by itself);"
          "\nper-token loss is not comparable across tokenizers — compare within a table only.")


if __name__ == '__main__':
    main()
