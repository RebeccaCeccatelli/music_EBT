"""
Training FLOPs per token of each model, measured on one real training step
(forward_loss_wrapper(batch, "train") + backward) with
torch's per-op FLOP formulas (torch.utils.flop_counter.flop_registry), at each model's own training
context. Gives the x-axis for compute-matched comparisons
(eval/plot_tokens_matched_curves.py --flops_json).

EBT's training step includes its MCMC refinement and the backward through it,
which is why it costs several times a plain transformer step. The counter
counts matmuls / convolutions / SDPA attention; ops it doesn't know (e.g.
xformers kernels) are missed, so the analytic 6·N reference (N = non-embedding
parameters) is printed next to every measurement as a sanity check.

Usage:
    python eval/train_flops.py --model EBT-REMI=<ckpt>:512 --model Llama-REMI=<ckpt>:1024 \\
        --out train_flops.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.mus.infer_ebt import load_checkpoint
from eval.context_matched_loss import draw_windows


class FlopCounter(TorchDispatchMode):
    """Total FLOPs over the dispatched ops, with torch's own per-op formulas
    (torch.utils.flop_counter.flop_registry). FlopCounterMode itself can't be used:
    its per-module tracker raises "Module hierarchy tracking is wrong" on EBT's
    autograd.grad inside the forward (torch 2.4)."""

    def __init__(self):
        super().__init__()
        self.total = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        f = flop_registry.get(func._overloadpacket)
        if f is not None:
            self.total += f(*args, **kwargs, out_val=out)
        return out


def count_step(model, windows: list[list[int]], device: str) -> float:
    batch = {'input_ids': torch.tensor(windows, dtype=torch.long, device=device)}
    batch_size, ctx = batch['input_ids'].shape
    model.train()
    model.zero_grad(set_to_none=True)
    with FlopCounter() as fc:
        out = model.forward_loss_wrapper(batch, phase='train')
        out['loss'].backward()
    model.zero_grad(set_to_none=True)
    return fc.total / (batch_size * ctx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', action='append', required=True, help='NAME=checkpoint_path:context')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--repeats', type=int, default=3, help='steps averaged (EBT may randomise MCMC steps)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}
    for spec in args.model:
        name, rest = spec.split('=', 1)
        path, ctx = rest.rsplit(':', 1)
        ctx = int(ctx)
        model, hp = load_checkpoint(path, device)
        n_all = sum(p.numel() for p in model.parameters())
        n_emb = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Embedding))
        windows = draw_windows(hp.tokenizer_type, args.batch_size * args.repeats, ctx, seed=0)
        per_tok = [count_step(model, windows[i * args.batch_size:(i + 1) * args.batch_size], device)
                   for i in range(args.repeats)]
        mean = sum(per_tok) / len(per_tok)
        ref = 6 * (n_all - n_emb)
        results[name] = {'checkpoint': path, 'context': ctx, 'model_name': hp.model_name,
                         'params': n_all, 'embedding_params': n_emb,
                         'train_flops_per_token': mean, 'per_step_samples': per_tok,
                         'six_n_per_token': ref}
        print(f"{name:16s} ctx {ctx:5d}  params {n_all / 1e6:6.1f}M  "
              f"train FLOPs/token {mean:.3e}  (6N {ref:.3e}, ratio {mean / ref:.2f})")
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"→ {args.out}")


if __name__ == '__main__':
    main()
