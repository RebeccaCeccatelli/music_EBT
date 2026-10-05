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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_control.train_density_regressor import load_music_dataset
from inference.mus.infer_ebt import load_checkpoint


def draw_windows(tokenizer_type: str, n: int, max_len: int, seed: int) -> list[list[int]]:
    """n validation windows of max_len tokens; shorter contexts use their prefix."""
    hp = Namespace(context_length=max_len, dataset_name='giga-midi', data_dir=None,
                   tokenizer_type=tokenizer_type)
    ds = load_music_dataset(tokenizer_type, 'validation', hp)
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


def evaluate(model, windows: list[list[int]], ctx: int, batch_size: int, device: str) -> dict:
    sums, n = {}, 0
    for i in range(0, len(windows), batch_size):
        chunk = [w[:ctx] for w in windows[i:i + batch_size]]
        batch = {'input_ids': torch.tensor(chunk, dtype=torch.long, device=device)}
        with torch.no_grad():
            out = model.forward_loss_wrapper(batch, phase='valid')
        for k in ('loss', 'initial_loss', 'final_loss'):
            v = out.get(k)
            if v is not None:
                sums[k] = sums.get(k, 0.0) + float(v) * len(chunk)
        n += len(chunk)
    res = {k: v / n for k, v in sums.items()}
    # The prediction used for generation: EBT's last MCMC step, else the plain loss.
    res['pred_loss'] = res.get('final_loss', res['loss'])
    res['perplexity'] = math.exp(res['pred_loss'])
    res['n_windows'] = n
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer_type', required=True)
    ap.add_argument('--model', action='append', required=True, help='NAME=checkpoint_path')
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--baseline_long_context', type=int, default=1024,
                    help='also evaluate non-EBT models at this context (0 = skip)')
    ap.add_argument('--n_windows', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_len = max(args.context, args.baseline_long_context)
    windows = draw_windows(args.tokenizer_type, args.n_windows, max_len, args.seed)
    print(f"{len(windows)} validation windows of {max_len} tokens (seed {args.seed})")

    results = []
    for spec in args.model:
        name, path = spec.split('=', 1)
        model, hp = load_checkpoint(path, device)
        is_ebt = 'ebt' in hp.model_name.lower()
        contexts = [args.context] + ([args.baseline_long_context]
                                     if args.baseline_long_context and not is_ebt else [])
        for ctx in contexts:
            r = evaluate(model, windows, ctx, args.batch_size, device)
            r.update(model=name, context=ctx, checkpoint=path)
            results.append(r)
            extra = (f"  (initial {r['initial_loss']:.4f}, MCMC-avg {r['loss']:.4f})"
                     if 'final_loss' in r else '')
            print(f"{name:8s} ctx {ctx:5d}: loss {r['pred_loss']:.4f}  ppl {r['perplexity']:.4f}{extra}")
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps({'tokenizer_type': args.tokenizer_type,
                                          'seed': args.seed, 'results': results}, indent=2))
    print(f"→ {args.out}")


if __name__ == '__main__':
    main()
