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


def candidate_selection(model, windows, batch_size, device, n_candidates, seed):
    """Self-verification: N random starts, the trained K steps each, and per position keep the
    candidate with the lowest energy (the EBT as its own verifier). The energy of the final
    prediction is measured by one extra step with step size 0 (energy only, no update).
    Returns CE of: candidate 1 alone, the mean over candidates, the min-energy choice, and the
    oracle (lowest CE, an upper bound on what any selection could gain)."""
    trained = model.hparams.get('mcmc_num_steps', 2)
    model.hparams['randomize_mcmc_num_steps'] = 1
    model.hparams['randomize_mcmc_num_steps_final_landscape'] = True
    model.hparams['extra_step_alpha_scale'] = 0.0
    sums = {'first': 0.0, 'mean': 0.0, 'min_energy': 0.0, 'oracle': 0.0}
    n_tok = 0
    for b, i in enumerate(range(0, len(windows), batch_size)):
        ids = torch.tensor(windows[i:i + batch_size], dtype=torch.long, device=device)
        x, targets = ids[:, :-1], ids[:, 1:].reshape(-1)
        keep = targets != model.pad_token_id
        ces, ens = [], []
        for c in range(n_candidates):
            torch.manual_seed(seed * 100003 + b * 101 + c)
            dists, energies = model(x, learning=False, return_raw_logits=True, no_randomness=True)
            final = dists[trained - 1].detach().reshape(-1, model.vocab_size)
            ce = F.cross_entropy(final.float(), targets, reduction='none', ignore_index=model.pad_token_id)
            ces.append(ce[keep])
            ens.append(energies[trained].detach().reshape(-1)[keep])   # energy of the final prediction
            del dists, energies, final
        ces, ens = torch.stack(ces), torch.stack(ens)                      # (N, T)
        pick = ens.argmin(0)
        sums['first'] += float(ces[0].sum())
        sums['mean'] += float(ces.mean(0).sum())
        sums['min_energy'] += float(ces.gather(0, pick.unsqueeze(0)).sum())
        sums['oracle'] += float(ces.min(0).values.sum())
        n_tok += int(keep.sum())
    model.hparams['extra_step_alpha_scale'] = 1.0
    return {k: v / n_tok for k, v in sums.items()}


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
    ap.add_argument('--alpha_scales', default='1',
                    help='comma list: step size of the steps beyond the trained ones, as a fraction of '
                         'the learned one (1 = original behaviour)')
    ap.add_argument('--n_candidates', type=int, default=0,
                    help='also run self-verification with this many random starts (0 = skip)')
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
        results[name] = {'checkpoint': path, 'tokenizer_type': hp.tokenizer_type,
                         'trained_mcmc_steps': trained, 'context': args.context,
                         'n_windows': len(windows), 'by_alpha_scale': {}}
        print(f"{name}  (trained with {trained} steps, {len(windows)} windows)")
        for scale in [float(a) for a in args.alpha_scales.split(',')]:
            model.hparams['extra_step_alpha_scale'] = scale
            torch.manual_seed(args.seed)
            ce, en, ent = per_step_loss(model, windows, args.batch_size, device)
            r = {'loss_per_step': ce, 'energy_per_step': en, 'entropy_per_step': ent}
            results[name]['by_alpha_scale'][str(scale)] = r
            if scale == 1.0:   # keep the original keys for existing readers
                results[name].update(r)
            print(f"  extra-step size = {scale} x learned")
            for k, (c, e, h) in enumerate(zip(ce, en, ent), 1):
                print(f"    step {k:2d}: loss {c:.4f}  ppl {math.exp(c):.4f}  energy {e:.4f}  entropy {h:.4f}")
        model.hparams['extra_step_alpha_scale'] = 1.0
        if args.n_candidates:
            sel = candidate_selection(model, windows, args.batch_size, device, args.n_candidates, args.seed)
            results[name]['self_verification'] = {'n_candidates': args.n_candidates, **sel}
            print(f"  self-verification, {args.n_candidates} starts: " +
                  ", ".join(f"{k} {v:.4f}" for k, v in sel.items()))
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"→ {args.out}")


if __name__ == '__main__':
    main()
