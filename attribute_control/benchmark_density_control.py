"""
Benchmark for density-controlled generation (R^3-style classifier guidance).

Given a trained NoteDensityRegressor checkpoint, generates short continuations
across a grid of density targets (plus an unguided baseline) for a sample of
validation prompts, measures achieved density with
attribute_control.note_density.compute_density(), and reports whether the
guidance actually steers density in the right direction. Optionally sweeps
--lambdas too, reusing the same loaded model/prompts across lambda values for
a fair, cheap comparison.

The EBT checkpoint is NOT a required argument: it is read from the
regressor checkpoint's own 'ebt_checkpoint' field (attribute_control/
train_density_regressor.py writes this at training time), so a future
regressor can be benchmarked with nothing but its own checkpoint path.
Pass --ebt_checkpoint to override (e.g. to test a regressor against a
*different* EBT checkpoint than the one it was trained on).

Metrics (see `summarize()`), computed per lambda value:
  - per-target mean/std achieved density
  - Pearson r and Spearman rho between target and achieved density
  - MAE(target, achieved)
  - monotonic_prompt_fraction: fraction of prompts where achieved density
    rises (non-strictly) with target
  - correct_direction_fraction: fraction of (prompt, target) pairs where
    achieved density moved the right way relative to that prompt's own
    unguided baseline

Usage:
    python attribute_control/benchmark_density_control.py \
        --regressor_checkpoint <path/to/best.pt> \
        --output_json <path/to/results.json>

    # Lambda sweep (baseline is computed once per prompt and shared):
    python attribute_control/benchmark_density_control.py \
        --regressor_checkpoint <path/to/best.pt> \
        --lambdas 0.5,1,2,4,8

See also job_scripts/mus/attr_control/benchmark_density.sh.
"""

import sys
import os
import json
import random
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timezone

import torch

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.mus.infer_ebt import load_checkpoint, load_dataset
from inference.mus.generate_music import generate_music
from attribute_control.attributes import ATTRIBUTES
from attribute_control.musicality_metrics import (
    build_bigram_logprob_table, score_sample,
)


# ── Stats helpers (no scipy dependency) ─────────────────────────────────────

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def _rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0] * len(vals)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def spearman(xs, ys):
    return pearson(_rank(xs), _rank(ys))


# ── Benchmark ────────────────────────────────────────────────────────────────

def load_model_and_data(args):
    device = args.device
    print(f"Device: {device}")

    reg_ckpt = torch.load(args.regressor_checkpoint, map_location="cpu", weights_only=False)
    ebt_checkpoint = args.ebt_checkpoint or reg_ckpt["ebt_checkpoint"]
    tokenizer_type = args.tokenizer_type or reg_ckpt.get("tokenizer_type", "REMI")
    attribute = args.attribute or reg_ckpt.get("attribute", "density")
    compute_fn = ATTRIBUTES[attribute]
    print(f"Regressor:  {args.regressor_checkpoint}")
    print(f"Attribute:  {attribute}"
          f"{'  (overridden)' if args.attribute else '  (from regressor metadata)'}")
    print(f"EBT ckpt:   {ebt_checkpoint}"
          f"{'  (overridden)' if args.ebt_checkpoint else '  (from regressor metadata)'}")

    model, hparams = load_checkpoint(ebt_checkpoint, device)
    hparams.device = device
    hparams.infer_max_gen_len = args.gen_len
    hparams.infer_temp = args.temperature
    hparams.infer_topp = args.top_p
    hparams.infer_logprobs = False
    hparams.infer_echo = False
    hparams.infer_ebt_advanced = False
    hparams.tokenizer_type = tokenizer_type

    dataset = load_dataset(hparams, split=args.split)
    rng = random.Random(args.seed)
    if args.prompt_indices:
        sample_indices = [int(x) for x in args.prompt_indices.split(",")]
    elif args.prompt_indices_file:
        with open(args.prompt_indices_file) as f:
            pool = json.load(f)
        sample_indices = rng.sample(pool, min(args.n_prompts, len(pool)))
    else:
        sample_indices = rng.sample(range(len(dataset)), min(args.n_prompts, len(dataset)))
    print(f"Prompts ({len(sample_indices)}): {sample_indices}")

    bigram_table = None
    if not args.no_musicality:
        cache_path = str(Path(args.regressor_checkpoint).parent.parent
                          / f"_bigram_table_{tokenizer_type}.npy")
        print(f"Building/loading real-data bigram table ({cache_path})...")
        emb_dim_unused = None
        vocab_size = model.embeddings.weight.shape[0]
        bigram_table = build_bigram_logprob_table(
            dataset, vocab_size, n_songs=3000, seed=0, cache_path=cache_path,
        )

    return (model, hparams, dataset, sample_indices, ebt_checkpoint, tokenizer_type,
            bigram_table, attribute, compute_fn)


def run_sweep(model, hparams, dataset, sample_indices, tokenizer_type, args, lam, bigram_table,
              compute_fn):
    """Runs the target sweep for one lambda value. Baseline (lambda=0, i.e. no
    steering) is prompt-dependent only, so it's recomputed per lambda call but
    is identical in expectation — kept simple rather than cached across calls."""
    device = args.device
    raw = []
    for pi in sample_indices:
        full_tokens = dataset.get_full_tokens(pi)
        prompt = full_tokens[: min(args.prompt_len, len(full_tokens))]
        batch = {
            "input_ids": torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
        }

        # Unguided baseline
        hparams.density_target = None
        hparams.lambda_density = 0.0
        hparams.density_regressor_ckpt = None
        baseline_vals = []
        baseline_music = []
        for _ in range(args.repeats):
            with torch.no_grad():
                out = generate_music(model, batch, hparams)
            gen = out["generation_tokens"][0]
            baseline_vals.append(compute_fn(gen, tokenizer_type))
            if bigram_table is not None:
                baseline_music.append(score_sample(model, gen, device, bigram_table, tokenizer_type))
        baseline = statistics.mean(baseline_vals)
        raw.append({"prompt": pi, "target": None, "lambda": lam, "achieved_mean": baseline,
                     "achieved_all": baseline_vals, "music_all": baseline_music})
        print(f"[lambda={lam}][prompt {pi}] baseline: {baseline:.4f}  (repeats={baseline_vals})"
              f"  music={baseline_music}")

        for tgt in args.targets:
            hparams.density_target = tgt
            hparams.lambda_density = lam
            hparams.density_regressor_ckpt = args.regressor_checkpoint
            vals = []
            music = []
            for _ in range(args.repeats):
                with torch.no_grad():
                    out = generate_music(model, batch, hparams)
                gen = out["generation_tokens"][0]
                vals.append(compute_fn(gen, tokenizer_type))
                if bigram_table is not None:
                    music.append(score_sample(model, gen, device, bigram_table, tokenizer_type))
            mean_val = statistics.mean(vals)
            raw.append({"prompt": pi, "target": tgt, "lambda": lam, "achieved_mean": mean_val,
                         "achieved_all": vals, "baseline": baseline, "music_all": music})
            print(f"[lambda={lam}][prompt {pi}] target={tgt:.3f}  achieved={mean_val:.4f}  "
                  f"(repeats={vals})  music={music}")

    return raw


def _aggregate_music(rows):
    """Mean/std of the three musicality metrics across all repeats in `rows`."""
    samples = [m for r in rows for m in r.get("music_all", [])]
    if not samples:
        return None
    out = {}
    for key in ("ebt_energy", "bigram_ll", "repetition_ratio"):
        vals = [m[key] for m in samples]
        out[key] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }
    return out


def summarize(raw, targets):
    guided = [r for r in raw if r["target"] is not None]
    baseline_rows = [r for r in raw if r["target"] is None]
    music_baseline = _aggregate_music(baseline_rows)
    music_by_target = {str(t): _aggregate_music([r for r in guided if r["target"] == t])
                        for t in targets}

    all_targets = [r["target"] for r in guided for _ in r["achieved_all"]]
    all_achieved = [v for r in guided for v in r["achieved_all"]]

    per_target = {}
    for t in targets:
        vals = [v for r in guided if r["target"] == t for v in r["achieved_all"]]
        per_target[t] = {
            "mean": statistics.mean(vals) if vals else None,
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    mae = (statistics.mean(abs(a - t) for a, t in zip(all_achieved, all_targets))
           if all_achieved else None)

    # Per-prompt monotonicity: does achieved density rise (non-strictly) with target?
    prompts = sorted({r["prompt"] for r in guided})
    monotonic_flags = []
    correct_direction = []
    for p in prompts:
        rows = sorted((r for r in guided if r["prompt"] == p), key=lambda r: r["target"])
        achieved_seq = [r["achieved_mean"] for r in rows]
        target_seq = [r["target"] for r in rows]
        monotonic_flags.append(spearman(target_seq, achieved_seq) > 0)
        baseline = rows[0]["baseline"]
        for r in rows:
            if r["target"] > baseline:
                correct_direction.append(r["achieved_mean"] > baseline)
            elif r["target"] < baseline:
                correct_direction.append(r["achieved_mean"] < baseline)

    return {
        "per_target": {str(k): v for k, v in per_target.items()},
        "pearson_r": pearson(all_targets, all_achieved) if all_achieved else None,
        "spearman_rho": spearman(all_targets, all_achieved) if all_achieved else None,
        "mae": mae,
        "monotonic_prompt_fraction": (
            sum(monotonic_flags) / len(monotonic_flags) if monotonic_flags else None
        ),
        "correct_direction_fraction": (
            sum(correct_direction) / len(correct_direction) if correct_direction else None
        ),
        "n_prompts": len(prompts),
        "music_baseline": music_baseline,
        "music_by_target": music_by_target,
    }


def print_summary(lam, summary):
    print(f"\n--- lambda={lam} ---")
    for t, stats in summary["per_target"].items():
        print(f"  target={t:>6}  achieved_mean={stats['mean']:.4f}  "
              f"std={stats['std']:.4f}  (n={stats['n']})")
    print(f"  Pearson r:                  {summary['pearson_r']:.4f}")
    print(f"  Spearman rho:               {summary['spearman_rho']:.4f}")
    print(f"  MAE(target, achieved):      {summary['mae']:.4f}")
    print(f"  Monotonic-per-prompt frac:  {summary['monotonic_prompt_fraction']:.4f}")
    print(f"  Correct-direction frac:     {summary['correct_direction_fraction']:.4f}")

    mb = summary.get("music_baseline")
    if mb:
        print(f"  --- musicality (baseline, unguided) ---")
        print(f"    ebt_energy={mb['ebt_energy']['mean']:.4f}  "
              f"bigram_ll={mb['bigram_ll']['mean']:.4f}  "
              f"repetition={mb['repetition_ratio']['mean']:.4f}")
        print(f"  --- musicality (guided, delta vs baseline; more negative bigram_ll / "
              f"higher energy / higher repetition = LESS coherent) ---")
        for t, m in summary["music_by_target"].items():
            if not m:
                continue
            de = m["ebt_energy"]["mean"] - mb["ebt_energy"]["mean"]
            db = m["bigram_ll"]["mean"] - mb["bigram_ll"]["mean"]
            dr = m["repetition_ratio"]["mean"] - mb["repetition_ratio"]["mean"]
            print(f"    target={t:>6}  d(energy)={de:+.4f}  d(bigram_ll)={db:+.4f}  "
                  f"d(repetition)={dr:+.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regressor_checkpoint", required=True)
    p.add_argument("--attribute", default=None,
                   choices=[None, "density", "velocity", "duration", "pitch_register", "polyphony", "rhythm", "drum_density", "melodic_interval", "syncopation"],
                   help="Override attribute (default: read from regressor metadata)")
    p.add_argument("--ebt_checkpoint", default=None,
                   help="Override the EBT checkpoint (default: read from regressor metadata)")
    p.add_argument("--tokenizer_type", default=None,
                   help="Override tokenizer type (default: read from regressor metadata)")
    p.add_argument("--targets", type=str, default="0.05,0.10,0.15,0.20,0.25",
                   help="Comma-separated targets to sweep, in the attribute's own units "
                        "(density/velocity/duration are all roughly 0-1 scale)")
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--prompt_indices", type=str, default=None,
                   help="Comma-separated explicit dataset indices to use as prompts "
                        "(overrides --n_prompts random sampling entirely)")
    p.add_argument("--prompt_indices_file", type=str, default=None,
                   help="JSON file of candidate indices to sample --n_prompts from, e.g. a "
                        "curated non-drum pool, instead of the whole dataset")
    p.add_argument("--repeats", type=int, default=3,
                   help="Generations per (prompt, target) to average out sampling noise")
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=128)
    p.add_argument("--lambda_density", type=float, default=1.0,
                   help="Single lambda value (ignored if --lambdas is given)")
    p.add_argument("--lambdas", type=str, default=None,
                   help="Comma-separated lambda values to sweep, e.g. '0.5,1,2,4,8'. "
                        "Reuses the same loaded model/prompts across all values.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", default=None,
                   help="Where to save full results (default: alongside the regressor checkpoint)")
    p.add_argument("--wandb_project", default=None, help="WandB project name (omit to disable)")
    p.add_argument("--wandb_run_name", default=None,
                   help="WandB run name (defaults to '<regressor checkpoint dir>')")
    p.add_argument("--no_musicality", action="store_true",
                   help="Skip the musical-coherence metrics (ebt_energy/bigram_ll/repetition) "
                        "for a faster, density-only run")
    args = p.parse_args()
    args.targets = [float(t) for t in args.targets.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else [args.lambda_density]

    (model, hparams, dataset, sample_indices, ebt_checkpoint, tokenizer_type,
     bigram_table, attribute, compute_fn) = load_model_and_data(args)
    print(f"Targets: {args.targets}   lambdas={lambdas}   repeats={args.repeats}   "
          f"gen_len={args.gen_len}")

    if args.output_json is None:
        suffix = "_lambda_sweep" if len(lambdas) > 1 else ""
        args.output_json = str(
            Path(args.regressor_checkpoint).parent / f"{attribute}_benchmark{suffix}.json"
        )

    meta = {
        "regressor_checkpoint": args.regressor_checkpoint,
        "attribute": attribute,
        "ebt_checkpoint": ebt_checkpoint,
        "tokenizer_type": tokenizer_type,
        "targets": args.targets,
        "lambdas": lambdas,
        "n_prompts": args.n_prompts,
        "repeats": args.repeats,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    use_wandb = _WANDB_AVAILABLE and args.wandb_project
    if use_wandb:
        run_name = args.wandb_run_name or f"benchmark-{Path(args.regressor_checkpoint).parent.name}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            job_type="density_benchmark",
            config=meta,
        )

    # Incremental: write the JSON (and log to wandb) after EVERY lambda, not
    # just at the end — a multi-lambda sweep can run long on a preemptable
    # partition, so a job that gets cut off mid-run still leaves usable,
    # complete-per-lambda results on disk instead of nothing.
    all_raw = []
    summaries_by_lambda = {}
    for lam in lambdas:
        raw = run_sweep(model, hparams, dataset, sample_indices, tokenizer_type, args, lam,
                         bigram_table, compute_fn)
        all_raw.extend(raw)
        summary = summarize(raw, args.targets)
        summaries_by_lambda[str(lam)] = summary
        print_summary(lam, summary)

        result = {"meta": meta, "raw": all_raw, "summary_by_lambda": summaries_by_lambda}
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [checkpoint] saved progress through lambda={lam} -> {args.output_json}")

        if use_wandb:
            baseline_vals = [r["achieved_mean"] for r in all_raw
                              if r["target"] is None and r["lambda"] == lam]
            log_data = {
                "density_benchmark/lambda": lam,
                "density_benchmark/baseline_density_mean": (
                    statistics.mean(baseline_vals) if baseline_vals else None
                ),
                "density_benchmark/pearson_r": summary["pearson_r"],
                "density_benchmark/spearman_rho": summary["spearman_rho"],
                "density_benchmark/mae": summary["mae"],
                "density_benchmark/monotonic_prompt_fraction": summary["monotonic_prompt_fraction"],
                "density_benchmark/correct_direction_fraction": summary["correct_direction_fraction"],
            }
            mb = summary.get("music_baseline")
            if mb:
                log_data["musicality/baseline_ebt_energy"] = mb["ebt_energy"]["mean"]
                log_data["musicality/baseline_bigram_ll"] = mb["bigram_ll"]["mean"]
                log_data["musicality/baseline_repetition"] = mb["repetition_ratio"]["mean"]
                # Averaged across targets: how much guidance at this lambda degrades
                # coherence relative to the unguided baseline, in one number per metric.
                per_t = [m for m in summary["music_by_target"].values() if m]
                if per_t:
                    log_data["musicality/mean_d_ebt_energy"] = statistics.mean(
                        m["ebt_energy"]["mean"] - mb["ebt_energy"]["mean"] for m in per_t)
                    log_data["musicality/mean_d_bigram_ll"] = statistics.mean(
                        m["bigram_ll"]["mean"] - mb["bigram_ll"]["mean"] for m in per_t)
                    log_data["musicality/mean_d_repetition"] = statistics.mean(
                        m["repetition_ratio"]["mean"] - mb["repetition_ratio"]["mean"] for m in per_t)
            wandb.log(log_data)
            per_target_rows = [
                [lam, float(t), stats["mean"], stats["std"], stats["n"]]
                for t, stats in summary["per_target"].items()
            ]
            per_target_table = wandb.Table(
                columns=["lambda", "target", "achieved_mean", "achieved_std", "n"],
                data=per_target_rows,
            )
            wandb.log({
                f"density_benchmark/target_vs_achieved_mean_lambda{lam}": wandb.plot.line(
                    per_target_table, "target", "achieved_mean",
                    title=f"Target vs Mean Achieved Density (lambda={lam})",
                )
            })

    if use_wandb:
        raw_rows = [
            [r["lambda"], r["prompt"], r["target"], v, r["baseline"]]
            for r in all_raw if r["target"] is not None
            for v in r["achieved_all"]
        ]
        raw_table = wandb.Table(
            columns=["lambda", "prompt", "target", "achieved", "baseline"], data=raw_rows
        )
        wandb.log({"density_benchmark/raw_samples": raw_table})

        # Best-lambda summary (by |pearson_r|, since sign flips are informative too)
        best_lambda = max(lambdas, key=lambda l: abs(summaries_by_lambda[str(l)]["pearson_r"]))
        wandb.summary["best_lambda_by_pearson"] = best_lambda
        wandb.summary["best_lambda_pearson_r"] = summaries_by_lambda[str(best_lambda)]["pearson_r"]

        wandb.finish()
        print(f"\nWandB run: {run_name} (project={args.wandb_project})")
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
