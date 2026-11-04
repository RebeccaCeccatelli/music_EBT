"""
Validation loss / perplexity of EBT and the baselines on *identical* windows
at the same context length.

The baselines were trained and validated at context 1024, EBT at 512, and
longer context lowers per-token loss by itself. This evaluates every model on
the same seeded validation windows at 512 tokens (the like-for-like number),
and the baselines also at 1024 (their training setting) to show how much
of their logged advantage was context.

Windows are drawn like the training dataloaders: REMI from a random offset in
the song, Anticipation from the start of the stored sequence. Losses come from
each model's own forward_loss_wrapper(batch, "valid"), i.e. the exact
training-time loss code. For EBT both the first and the last MCMC step are
reported. The last step (final_loss) is the prediction actually sampled from,
and is the counterpart of the baselines' next-token loss. Perplexity =
exp(mean loss over windows); every window has the same number of predicted
tokens, so that is the per-token mean.

Usage:
    python eval/context_matched_loss.py --tokenizer_type REMI \\
        --model EBT=<ckpt> --model GPT-2=<ckpt> --model Llama=<ckpt> \\
        --baseline_long_context 1024 --out results.json
"""

import argparse
import json
import math
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_control.train_density_regressor import load_music_dataset
from inference.mus.infer_ebt import load_checkpoint


def draw_windows(tokenizer_type: str, n: int, max_len: int, seed: int,
                 first_n: bool = False) -> list[list[int]]:
    """n validation windows of max_len tokens; shorter contexts use their prefix.
    first_n: the first n stored Anticipation sequences in file order, i.e. exactly what the
    training-time validation scores (unshuffled loader, limit_val_batches × batch size)."""
    hp = Namespace(context_length=max_len, dataset_name='giga-midi', data_dir=None,
                   tokenizer_type=tokenizer_type)
    ds = load_music_dataset(tokenizer_type, 'validation', hp)
    if first_n:
        assert tokenizer_type.startswith('Anticipation'), 'REMI windows start at random offsets'
        return [ds.get_full_tokens(i)[:max_len] for i in range(n)]
    rng = random.Random(seed)
    windows, tries = [], 0
    while len(windows) < n and tries < 20 * n:
        tries += 1
        toks = ds.get_full_tokens(rng.randrange(len(ds)))
        if len(toks) < max_len:
            continue  # no padding: every window predicts the same number of real tokens
        start = 0 if tokenizer_type.startswith('Anticipation') else rng.randrange(len(toks) - max_len + 1)
        windows.append(toks[start:start + max_len])
    if len(windows) < n:
        print(f"  only {len(windows)} windows of {max_len} tokens found")
    return windows


def predicted_logits(model, input_ids: torch.Tensor, is_ebt: bool) -> torch.Tensor:
    """(B, S, V) logits of the prediction a model samples from: EBT's last MCMC step, else the
    plain next-token output of the baselines (Llama and GPT-2 share this signature)."""
    if is_ebt:
        dists, _ = model(input_ids, return_raw_logits=True, no_randomness=True)
        return dists[-1].reshape(input_ids.shape[0], input_ids.shape[1], -1)
    return model(input_ids, learning=False, return_raw_logits=True)


def per_window_metrics(model, chunk: list[list[int]], is_ebt: bool, device: str) -> dict:
    """Per-window mean CE and top-1/top-5 accuracy of the sampled-from prediction."""
    ids = torch.tensor(chunk, dtype=torch.long, device=device)
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        logp = F.log_softmax(predicted_logits(model, x, is_ebt).float(), dim=-1)
        ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)                   # (B, S)
        top5 = logp.topk(5, dim=-1).indices                                  # (B, S, 5)
        hit1 = (top5[..., 0] == y).float()
        hit5 = (top5 == y.unsqueeze(-1)).any(-1).float()
    keep = (y != model.pad_token_id).float()
    n = keep.sum(1)
    return {'loss': ((ce * keep).sum(1) / n).tolist(), 'top1': ((hit1 * keep).sum(1) / n).tolist(),
            'top5': ((hit5 * keep).sum(1) / n).tolist()}


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    """Mean of b - a over windows with a 95% percentile bootstrap CI (windows resampled jointly)."""
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return {'mean_diff': float(d.mean()), 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'frac_windows_b_worse': float((d > 0).mean()), 'sd_diff': float(d.std(ddof=1))}


def evaluate(model, windows: list[list[int]], ctx: int, batch_size: int, device: str,
             is_ebt: bool = False, per_window: bool = False) -> dict:
    sums, n = {}, 0
    pw = {'loss': [], 'top1': [], 'top5': []}
    for i in range(0, len(windows), batch_size):
        chunk = [w[:ctx] for w in windows[i:i + batch_size]]
        batch = {'input_ids': torch.tensor(chunk, dtype=torch.long, device=device)}
        with torch.no_grad():
            out = model.forward_loss_wrapper(batch, phase='valid')
        for k in ('loss', 'initial_loss', 'final_loss'):
            v = out.get(k)
            if v is not None:
                sums[k] = sums.get(k, 0.0) + float(v) * len(chunk)
        if per_window:
            for k, v in per_window_metrics(model, chunk, is_ebt, device).items():
                pw[k].extend(v)
        n += len(chunk)
    res = {k: v / n for k, v in sums.items()}
    # The prediction used for generation: EBT's last MCMC step, else the plain loss.
    res['pred_loss'] = res.get('final_loss', res['loss'])
    res['perplexity'] = math.exp(res['pred_loss'])
    res['n_windows'] = n
    if per_window:
        # Same loss as the training-time code above, recomputed per window from the logits.
        assert abs(float(np.mean(pw['loss'])) - res['pred_loss']) < 2e-3, (np.mean(pw['loss']), res['pred_loss'])
        res['top1_acc'], res['top5_acc'] = float(np.mean(pw['top1'])), float(np.mean(pw['top5']))
        res['per_window'] = pw
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer_type', required=True)
    ap.add_argument('--model', action='append', required=True, help='NAME=checkpoint_path')
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--baseline_long_context', type=int, default=1024,
                    help='also evaluate non-EBT models at this context (0 = skip)')
    ap.add_argument('--window_len', type=int, default=None,
                    help='length windows are drawn at (default: the longest evaluated context); '
                         '1024 keeps the standard windows when --baseline_long_context 0')
    ap.add_argument('--n_windows', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--first_n', action='store_true',
                    help='Anticipation: use the first --n_windows stored sequences (the training-time '
                         'validation slice) instead of a seeded random draw')
    ap.add_argument('--per_window', action='store_true',
                    help='also store per-window loss/top-1/top-5 and paired bootstrap CIs vs the first model')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_len = args.window_len or max(args.context, args.baseline_long_context)
    windows = draw_windows(args.tokenizer_type, args.n_windows, max_len, args.seed, args.first_n)
    print(f"{len(windows)} validation windows of {max_len} tokens "
          f"({'first in file order' if args.first_n else f'seed {args.seed}'})")

    results = []
    for spec in args.model:
        name, path = spec.split('=', 1)
        model, hp = load_checkpoint(path, device)
        is_ebt = 'ebt' in hp.model_name.lower()
        contexts = [args.context] + ([args.baseline_long_context]
                                     if args.baseline_long_context and not is_ebt else [])
        for ctx in contexts:
            r = evaluate(model, windows, ctx, args.batch_size, device, is_ebt, args.per_window)
            r.update(model=name, context=ctx, checkpoint=path)
            results.append(r)
            extra = (f"  (initial {r['initial_loss']:.4f}, MCMC-avg {r['loss']:.4f})"
                     if 'final_loss' in r else '')
            acc = f"  top1 {r['top1_acc']:.4f}  top5 {r['top5_acc']:.4f}" if 'top1_acc' in r else ''
            print(f"{name:8s} ctx {ctx:5d}: loss {r['pred_loss']:.4f}  ppl {r['perplexity']:.4f}{acc}{extra}")
        del model
        torch.cuda.empty_cache()

    comparisons = []
    if args.per_window:
        ref = results[0]  # first --model at --context (EBT in SET=ebtrecipe)
        for r in results[1:]:
            if r['context'] != ref['context']:
                continue
            c = {'a': ref['model'], 'b': r['model'], 'context': r['context']}
            for k in ('loss', 'top1', 'top5'):
                c[k] = paired_bootstrap(ref['per_window'][k], r['per_window'][k], seed=args.seed)
            comparisons.append(c)
            L, A = c['loss'], c['top1']
            print(f"{r['model']} − {ref['model']} @ {r['context']}: loss {L['mean_diff']:+.4f} "
                  f"[{L['ci95'][0]:+.4f}, {L['ci95'][1]:+.4f}], {r['model']} worse on "
                  f"{100 * L['frac_windows_b_worse']:.0f}% of windows; top1 {A['mean_diff']:+.4f} "
                  f"[{A['ci95'][0]:+.4f}, {A['ci95'][1]:+.4f}]")

    Path(args.out).write_text(json.dumps({'tokenizer_type': args.tokenizer_type,
                                          'seed': args.seed, 'results': results,
                                          'comparisons': comparisons}, indent=2))
    print(f"→ {args.out}")


if __name__ == '__main__':
    main()
