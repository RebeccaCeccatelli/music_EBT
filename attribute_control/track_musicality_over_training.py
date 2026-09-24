"""
Track model "goodness" across a training run using signals OTHER than
validation loss — loss is confounded by vocabulary size (see
docs/thesis_findings/2026-09-22_anticipation_loss_vocab_normalization.md)
and only measures next-token prediction, not generation quality.

For each given checkpoint (same lineage, different training steps):
  1. Generate SAME-seed unguided samples from the SAME fixed prompts (fair
     comparison across checkpoints — only the model differs).
  2. Score them with the musicality metrics already used for listening
     sweeps: ebt_energy (the model's own self-consistency rating) and
     repetition_ratio (both tokenizer-agnostic). bigram_ll/grammar_violation_rate/
     melodic_interval are REMI-only (see musicality_metrics.score_sample) and
     skipped here for Anticipation's much larger vocabulary.
  3. Also compute energy_improvement directly via the model's own
     forward_loss_wrapper on REAL validation windows (initial vs. final MCMC
     energy, no generation needed) — a training-time diagnostic, applied
     here post-hoc to compare checkpoints on the exact same held-out data.

Usage:
    python attribute_control/track_musicality_over_training.py \\
        --tokenizer_type Anticipation-Arrival-Time \\
        --checkpoints ckpt1.ckpt,ckpt2.ckpt,ckpt3.ckpt \\
        --n_prompts 8 --gen_len 128
"""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))

from inference.mus.infer_ebt import load_checkpoint, load_dataset
from inference.mus.generate_music import generate_music
from attribute_control.musicality_metrics import score_sample

REMI_TOKENIZER_CONFIG = "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json"


def step_from_path(ckpt_path: str) -> int:
    import re
    m = re.search(r'step=(\d+)', ckpt_path)
    return int(m.group(1)) if m else -1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint paths")
    p.add_argument("--tokenizer_type", required=True)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=128)
    p.add_argument("--n_valid_windows", type=int, default=20,
                   help="Real validation windows used for the energy_improvement metric")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb_project", default="mus_symb_attr_control")
    p.add_argument("--wandb_run_name", default=None)
    args = p.parse_args()

    ckpts = sorted(args.checkpoints.split(","), key=step_from_path)
    device = args.device
    print(f"Device: {device}")
    print(f"Checkpoints ({len(ckpts)}), by step: {[step_from_path(c) for c in ckpts]}")

    # Fixed prompts and fixed validation windows — SAME across every
    # checkpoint, drawn once from the first checkpoint's own dataset load
    # (dataset content doesn't depend on which checkpoint, only tokenizer_type).
    import random
    random.seed(args.seed)

    results = []
    dataset = None
    for ckpt_path in ckpts:
        step = step_from_path(ckpt_path)
        print(f"\n=== step {step}: {Path(ckpt_path).name} ===")
        model, hparams = load_checkpoint(ckpt_path, device)
        hparams.device = device
        if dataset is None:
            dataset = load_dataset(hparams, split="validation")
            n = len(dataset)
            prompt_idxs = random.sample(range(n), args.n_prompts)
            valid_idxs = random.sample(range(n), args.n_valid_windows)

        hparams.infer_temp = args.temperature
        hparams.infer_topp = args.top_p
        hparams.infer_max_gen_len = args.gen_len
        hparams.infer_logprobs = False
        hparams.infer_echo = False
        hparams.attribute_target = None
        hparams.lambda_attribute = 0.0
        hparams.attribute_regressor_ckpt = None
        hparams.attribute_regressor_ckpts = None

        # ── Generation-based musicality metrics ────────────────────────────
        energies, reps = [], []
        for idx in prompt_idxs:
            full = dataset.get_full_tokens(idx)
            prompt = full[:args.prompt_len]
            batch = {"input_ids": torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)}
            with torch.no_grad():
                out = generate_music(model, batch, hparams)
            gen = [int(t) for t in out["generation_tokens"][0]]
            m = score_sample(model, gen, device, bigram_table=None, tokenizer_type=args.tokenizer_type)
            energies.append(m["ebt_energy"])
            reps.append(m["repetition_ratio"])
        mean_energy = sum(energies) / len(energies)
        mean_rep = sum(reps) / len(reps)

        # ── Real-data energy_improvement (no generation — teacher-forced) ──
        improvements = []
        for idx in valid_idxs:
            full = dataset.get_full_tokens(idx)
            window = full[:max(args.prompt_len, 32)]
            batch = {"input_ids": torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)}
            with torch.no_grad():
                loss_out = model.forward_loss_wrapper(batch, phase="valid")
            imp = loss_out.get("energy_improvement")
            if imp is not None:
                improvements.append(float(imp))
        mean_improvement = sum(improvements) / len(improvements) if improvements else float("nan")

        print(f"  ebt_energy(generated)={mean_energy:.4f}  repetition_ratio={mean_rep:.4f}  "
              f"energy_improvement(real data)={mean_improvement:.4f}")
        results.append({
            "step": step, "ckpt": ckpt_path,
            "ebt_energy": mean_energy, "repetition_ratio": mean_rep,
            "energy_improvement": mean_improvement,
        })

    print("\n=== Summary (by step) ===")
    print(f"{'step':>8}  {'ebt_energy':>12}  {'repetition':>12}  {'energy_impr':>12}")
    for r in results:
        print(f"{r['step']:>8}  {r['ebt_energy']:>12.4f}  {r['repetition_ratio']:>12.4f}  {r['energy_improvement']:>12.4f}")

    try:
        import wandb
        run_name = args.wandb_run_name or f"musicality-over-training-{args.tokenizer_type}"
        wandb.init(project=args.wandb_project, name=run_name, job_type="training_quality_track")
        table = wandb.Table(columns=["step", "ebt_energy", "repetition_ratio", "energy_improvement", "ckpt"],
                             data=[[r["step"], r["ebt_energy"], r["repetition_ratio"],
                                    r["energy_improvement"], r["ckpt"]] for r in results])
        wandb.log({"musicality_over_training": table})
        for r in results:
            wandb.log({"step": r["step"], "ebt_energy": r["ebt_energy"],
                       "repetition_ratio": r["repetition_ratio"],
                       "energy_improvement": r["energy_improvement"]})
        wandb.finish()
        print(f"\nLogged to wandb: {run_name} (project={args.wandb_project})")
    except Exception as e:
        print(f"(wandb logging skipped: {e})")


if __name__ == "__main__":
    main()
