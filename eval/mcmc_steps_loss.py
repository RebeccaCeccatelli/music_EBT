"""
Validation loss of an EBT after each MCMC refinement step, run for more steps
than it was trained with.

Our EBTs were trained with mcmc_num_steps = 2 and ebt_type time_embed, whose
step embedding only has entries for steps 0 and 1, so simply raising
mcmc_num_steps would index past it. Extra steps instead repeat the final
energy landscape (step index 1), the mechanism the EBT code uses for
randomize_mcmc_num_steps with randomize_mcmc_num_steps_final_landscape: K steps
are [0, 1, 1, ..., 1]. Step sizes are the learned ones, no noise.

One forward pass with K steps gives the cross-entropy after every step 1..K
(step 1 starts from the random initial logits). Windows are the seeded
context-matched validation windows (eval/context_matched_loss.py), so step 2
reproduces final_loss there.

Usage:
    python eval/mcmc_steps_loss.py --model EBT-REMI=<ckpt> --max_steps 16 --out mcmc_steps.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.mus.infer_ebt import load_checkpoint
from eval.context_matched_loss import draw_windows


def per_step_loss(model, windows: list[list[int]], batch_size: int, device: str) -> tuple[list, list, list]:
    """Mean CE, mean energy and mean predictive entropy (nats) after each MCMC step, over all windows."""
    ce_sums, en_sums, ent_sums, n = None, None, None, 0
    for i in range(0, len(windows), batch_size):
        ids = torch.tensor(windows[i:i + batch_size], dtype=torch.long, device=device)
        x, targets = ids[:, :-1], ids[:, 1:].reshape(-1)
        dists, energies = model(x, learning=False, return_raw_logits=True, no_randomness=True)
        ce = [float(F.cross_entropy(d.detach().reshape(-1, model.vocab_size), targets,
                                    ignore_index=model.pad_token_id)) for d in dists]
        en = [float(e.detach().mean()) for e in energies]
        keep = targets != model.pad_token_id
        ent = []
        for d in dists:
            logp = F.log_softmax(d.detach().reshape(-1, model.vocab_size)[keep].float(), dim=-1)
            ent.append(float(-(logp.exp() * logp).sum(-1).mean()))
            del logp
        del dists, energies
        if ce_sums is None:
            ce_sums, en_sums, ent_sums = [0.0] * len(ce), [0.0] * len(en), [0.0] * len(ent)
        ce_sums = [s + c * len(ids) for s, c in zip(ce_sums, ce)]
        en_sums = [s + e * len(ids) for s, e in zip(en_sums, en)]
        ent_sums = [s + h * len(ids) for s, h in zip(ent_sums, ent)]
        n += len(ids)
    return [s / n for s in ce_sums], [s / n for s in en_sums], [s / n for s in ent_sums]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', action='append', required=True, help='NAME=checkpoint_path')
    ap.add_argument('--max_steps', type=int, default=16)
    ap.add_argument('--context', type=int, default=512)
    ap.add_argument('--window_len', type=int, default=1024,
                    help='length windows are drawn at (1024 = same windows as context_matched_loss); the first --context tokens are used')
    ap.add_argument('--n_windows', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}
    for spec in args.model:
        name, path = spec.split('=', 1)
        model, hp = load_checkpoint(path, device)
        model.eval()
        trained = model.hparams.get('mcmc_num_steps', 2)
        # K total steps = trained landscapes once each, then the last one repeated.
        model.hparams['randomize_mcmc_num_steps'] = max(args.max_steps - trained, 0)
        model.hparams['randomize_mcmc_num_steps_final_landscape'] = True
        windows = [w[:args.context] for w in
                   draw_windows(hp.tokenizer_type, args.n_windows, args.window_len, args.seed)]
        ce, en, ent = per_step_loss(model, windows, args.batch_size, device)
        results[name] = {'checkpoint': path, 'tokenizer_type': hp.tokenizer_type,
                         'trained_mcmc_steps': trained, 'context': args.context,
                         'n_windows': len(windows), 'loss_per_step': ce, 'energy_per_step': en,
                         'entropy_per_step': ent}
        print(f"{name}  (trained with {trained} steps, {len(windows)} windows)")
        for k, (c, e, h) in enumerate(zip(ce, en, ent), 1):
            print(f"  step {k:2d}: loss {c:.4f}  ppl {math.exp(c):.4f}  energy {e:.4f}  entropy {h:.4f}")
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"→ {args.out}")


if __name__ == '__main__':
    main()
