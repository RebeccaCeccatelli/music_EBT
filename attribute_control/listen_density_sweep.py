"""
Generate a listenable grid of density-guided samples across (prompt, lambda,
target) combinations and log them to wandb as playable audio, so you can
judge by ear where guidance strength starts corrupting musical quality —
complementing benchmark_density_control.py's proxy metrics with the real thing.

For each of --n_prompts prompts:
  - one unguided baseline (lambda=0) — the "outputs are good" reference point
  - for each lambda in --lambdas:
      for each target in --targets:
          generate, compute achieved density + musicality metrics,
          render prompt+continuation to WAV, log as one row of a wandb.Table
          with an embedded, playable audio column.

All samples for a given prompt use the SAME prompt and generation settings,
so within one prompt you can A/B the baseline against every (lambda, target)
combination directly.

Usage:
    python attribute_control/listen_density_sweep.py \
        --regressor_checkpoint <path/to/best.pt> \
        --wandb_project mus_symb_attr_control

See also job_scripts/mus/attr_control/listen_density_sweep.sh.
"""

import sys
import os
import json
import shutil
import random
import argparse
import tempfile
import statistics
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))

from inference.mus.infer_ebt import load_checkpoint, load_dataset
from inference.mus.generate_music import generate_music
from inference.mus.tokens_to_midi import tokens_to_midi
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
from attribute_control.attributes import ATTRIBUTES
from attribute_control.musicality_metrics import build_bigram_logprob_table, score_sample, segment_scores
from convert_midi_simple import simple_synth

import wandb


CORPUS_STATS_PATH = Path(__file__).parent / "corpus_stats.json"


def load_corpus_stats(attribute: str) -> dict:
    """
    Mean/stdev of `attribute` over a large sample of the training split (see
    compute_corpus_stats.py), used to express values/deltas in standard
    deviations from what real music actually does — raw units are misleading
    across attributes with very different natural spreads (e.g. density's
    corpus stdev is ~0.01, velocity's is ~0.14: the same absolute delta of
    0.03 is a mild nudge for one and a ~3-stdev, near-unrealistic ask for the
    other).
    """
    with open(CORPUS_STATS_PATH) as f:
        stats = json.load(f)
    return stats[attribute]


def zscore(value, mean: float, stdev: float):
    if value is None or stdev == 0:
        return None
    return (value - mean) / stdev


def early_late_scores(tokens, bigram_table):
    """
    First vs. last quarter's bigram_ll/grammar_violation_rate/melodic_interval
    — a whole-sequence average can hide a generation that's fine early on and
    degenerates later (a pattern seen at higher lambda), so this exposes the
    trend directly instead of only reporting one blended number.
    """
    if bigram_table is None:
        return None, None, None, None, None, None
    segs = segment_scores(tokens, bigram_table, n_segments=4)
    early, late = segs[0], segs[-1]
    return (early["bigram_ll"], late["bigram_ll"],
            early["grammar_violation_rate"], late["grammar_violation_rate"],
            early["melodic_interval"], late["melodic_interval"])


def render_wav(tokens, tokenizer, out_dir: Path, name: str) -> str:
    midi_bytes = tokens_to_midi(tokens, tokenizer)
    midi_path = out_dir / f"{name}.mid"
    wav_path = out_dir / f"{name}.wav"
    midi_path.write_bytes(midi_bytes)
    simple_synth(str(midi_path), str(wav_path))
    return str(wav_path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regressor_checkpoint", required=True)
    p.add_argument("--attribute", default=None,
                   choices=[None, "density", "velocity", "duration", "pitch_register", "polyphony", "rhythm", "drum_density", "melodic_interval"],
                   help="Override attribute (default: read from regressor metadata)")
    p.add_argument("--ebt_checkpoint", default=None,
                   help="Override the EBT checkpoint (default: read from regressor metadata)")
    p.add_argument("--tokenizer_type", default=None,
                   help="Override tokenizer type (default: read from regressor metadata)")
    p.add_argument("--n_prompts", type=int, default=5)
    p.add_argument("--prompt_indices", type=str, default=None,
                   help="Comma-separated explicit dataset indices to use as prompts "
                        "(overrides --n_prompts random sampling entirely)")
    p.add_argument("--prompt_indices_file", type=str, default=None,
                   help="JSON file of candidate indices to sample --n_prompts from, e.g. a "
                        "curated non-drum pool, instead of the whole dataset")
    p.add_argument("--targets", type=str, default="0.05,0.10,0.15,0.20,0.25",
                   help="Shared absolute targets for every prompt (ignored if --target_deltas given)")
    p.add_argument("--target_deltas", type=str, default=None,
                   help="Comma-separated offsets (e.g. '-0.04,-0.02,0,0.02,0.04') applied to "
                        "EACH prompt's own measured baseline density instead of shared absolute "
                        "--targets. Not clamped to any 'realistic' range — a dense prompt can be "
                        "pushed denser, a sparse one sparser, following its own natural point.")
    p.add_argument("--target_deltas_std", type=str, default=None,
                   help="Like --target_deltas, but in units of the attribute's corpus standard "
                        "deviation (see attribute_control/corpus_stats.json) instead of raw "
                        "value units — e.g. '-2,-1,1,2' means push 2 stdev below/above baseline. "
                        "Comparable across attributes, unlike raw deltas. Takes priority over "
                        "--target_deltas if both are given.")
    p.add_argument("--baseline_repeats", type=int, default=5,
                   help="Generations to average for each prompt's baseline density reference "
                        "(matters most for --target_deltas, since it anchors every target)")
    p.add_argument("--lambdas", type=str, default="0.25,0.5,1,2,4")
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb_project", default="mus_symb_attr_control")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_musicality", action="store_true")
    p.add_argument("--no_step_gating", action="store_true",
                   help="Disable REMI token-type gating of attribute guidance (old "
                        "behavior: the R³ energy gradient fires on every generation "
                        "step regardless of token type, not just the ones where it "
                        "has a real lever on the target attribute)")
    p.add_argument("--lambda_taper", action="store_true",
                   help="Hold lambda at full strength for --lambda_taper_hold_frac of "
                        "generation, then linearly decay to --lambda_taper_floor (a "
                        "fraction of lambda) by the end — targets the compounding-drift "
                        "failure mode where continued full-strength guidance on an "
                        "increasingly self-generated (not real) context degrades into "
                        "cacophony later in long generations")
    p.add_argument("--lambda_taper_hold_frac", type=float, default=0.4)
    p.add_argument("--lambda_taper_floor", type=float, default=0.2)
    args = p.parse_args()
    targets = [float(t) for t in args.targets.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]
    args.target_deltas = ([float(d) for d in args.target_deltas.split(",")]
                           if args.target_deltas else None)
    args.target_deltas_std = ([float(d) for d in args.target_deltas_std.split(",")]
                               if args.target_deltas_std else None)

    device = args.device
    print(f"Device: {device}")

    reg_ckpt = torch.load(args.regressor_checkpoint, map_location="cpu", weights_only=False)
    ebt_checkpoint = args.ebt_checkpoint or reg_ckpt["ebt_checkpoint"]
    tokenizer_type = args.tokenizer_type or reg_ckpt.get("tokenizer_type", "REMI")
    attribute = args.attribute or reg_ckpt.get("attribute", "density")
    compute_fn = ATTRIBUTES[attribute]
    corpus_stats = load_corpus_stats(attribute)
    corpus_mean, corpus_stdev = corpus_stats["mean"], corpus_stats["stdev"]
    print(f"Regressor:  {args.regressor_checkpoint}")
    print(f"Attribute:  {attribute}"
          f"{'  (overridden)' if args.attribute else '  (from regressor metadata)'}")
    print(f"EBT ckpt:   {ebt_checkpoint}")
    print(f"Corpus:     mean={corpus_mean:.4f}  stdev={corpus_stdev:.4f}  "
          f"(train split, n={corpus_stats['n_samples']})")

    if args.target_deltas_std is not None:
        args.target_deltas = [d * corpus_stdev for d in args.target_deltas_std]
        print(f"Target deltas: {args.target_deltas_std} stdev  "
              f"= {[round(d, 4) for d in args.target_deltas]} raw units")

    model, hparams = load_checkpoint(ebt_checkpoint, device)
    hparams.device = device
    hparams.infer_max_gen_len = args.gen_len
    hparams.infer_temp = args.temperature
    hparams.infer_topp = args.top_p
    hparams.infer_logprobs = False
    hparams.infer_echo = False
    hparams.infer_ebt_advanced = False
    hparams.tokenizer_type = tokenizer_type
    hparams.attr_gate_by_token_type = not args.no_step_gating
    print(f"Step gating: {'off (legacy, fires every step)' if args.no_step_gating else 'on'}")
    hparams.attribute_lambda_taper = args.lambda_taper
    hparams.attribute_lambda_taper_hold_frac = args.lambda_taper_hold_frac
    hparams.attribute_lambda_taper_floor = args.lambda_taper_floor
    if args.lambda_taper:
        print(f"Lambda taper: on (hold {args.lambda_taper_hold_frac:.0%}, "
              f"floor {args.lambda_taper_floor:.0%})")

    tokenizer, vocab_size, pad_token_id = load_tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=getattr(hparams, "tokenizer_config_path", None),
        dataset_name=getattr(hparams, "dataset_name", "giga_midi"),
    )

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
        bigram_table = build_bigram_logprob_table(
            dataset, model.embeddings.weight.shape[0], n_songs=3000, seed=0,
            cache_path=cache_path,
        )

    out_dir = Path(tempfile.mkdtemp(prefix="listen_density_sweep_"))
    print(f"Rendering scratch dir: {out_dir}")

    run_name = args.wandb_run_name or f"listen-{Path(args.regressor_checkpoint).parent.name}"
    wandb.init(project=args.wandb_project, name=run_name, job_type="density_listening_sweep",
               config={
                   "regressor_checkpoint": args.regressor_checkpoint,
                   "attribute": attribute,
                   "corpus_mean": corpus_mean,
                   "corpus_stdev": corpus_stdev,
                   "step_gating": not args.no_step_gating,
                   "ebt_checkpoint": ebt_checkpoint,
                   "tokenizer_type": tokenizer_type,
                   "targets": targets,
                   "lambdas": lambdas,
                   "n_prompts": args.n_prompts,
                   "prompt_len": args.prompt_len,
                   "gen_len": args.gen_len,
                   "temperature": args.temperature,
                   "top_p": args.top_p,
                   "seed": args.seed,
               })

    columns = ["prompt_id", "condition", "lambda", "target", "target_delta", "target_zscore",
               "baseline_value", "achieved_value", "achieved_zscore",
               "ebt_energy", "bigram_ll", "repetition_ratio", "grammar_violation_rate",
               "bigram_ll_early", "bigram_ll_late",
               "grammar_violation_rate_early", "grammar_violation_rate_late",
               "melodic_interval_early", "melodic_interval_late", "audio"]
    table_rows = []

    try:
        for pi in sample_indices:
            full_tokens = dataset.get_full_tokens(pi)
            prompt = full_tokens[: min(args.prompt_len, len(full_tokens))]
            batch = {
                "input_ids": torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
            }

            # ── Ground truth: what the real song actually does next ────────────
            # Not model output at all — the actual continuation from the dataset,
            # same length as gen_len where available. The reference point neither
            # baseline nor guided samples have been compared against so far.
            gt_continuation = full_tokens[len(prompt): len(prompt) + args.gen_len]
            if gt_continuation:
                gt_achieved = compute_fn(gt_continuation, tokenizer_type)
                gt_z = zscore(gt_achieved, corpus_mean, corpus_stdev)
                gt_music = (score_sample(model, gt_continuation, device, bigram_table)
                            if bigram_table is not None else {})
                gt_bll_e, gt_bll_l, gt_gvr_e, gt_gvr_l, gt_mi_e, gt_mi_l = early_late_scores(gt_continuation, bigram_table)
                gt_wav = render_wav(gt_continuation, tokenizer, out_dir, f"p{pi}_ground_truth")
                print(f"[prompt {pi}] ground truth: achieved={gt_achieved:.4f} (z={gt_z:+.2f})  "
                      f"music={gt_music}  "
                      f"early/late melodic_interval={gt_mi_e}/{gt_mi_l}")
                table_rows.append([
                    pi, "ground truth (real song)", None, None, None, None, None,
                    gt_achieved, gt_z,
                    gt_music.get("ebt_energy"), gt_music.get("bigram_ll"),
                    gt_music.get("repetition_ratio"), gt_music.get("grammar_violation_rate"),
                    gt_bll_e, gt_bll_l, gt_gvr_e, gt_gvr_l, gt_mi_e, gt_mi_l,
                    wandb.Audio(gt_wav, caption=f"p{pi} ground truth {attribute}={gt_achieved:.3f}"),
                ])

            # ── Unguided baseline: the "outputs are good" reference point ──────
            # Averaged over --baseline_repeats for a stable per-prompt reference
            # (single-draw baselines can be noisy, especially for prompts that
            # don't commit strongly to one instrument/style) — only matters for
            # --target_deltas mode, where this value anchors every target below,
            # but computed either way for the logged reference point.
            hparams.attribute_target = None
            hparams.lambda_attribute = 0.0
            hparams.attribute_regressor_ckpt = None
            baseline_vals = []
            baseline_gen = None
            for _ in range(args.baseline_repeats):
                with torch.no_grad():
                    out = generate_music(model, batch, hparams)
                gen = out["generation_tokens"][0]
                baseline_vals.append(compute_fn(gen, tokenizer_type))
                if baseline_gen is None:
                    baseline_gen = gen  # log audio for the first draw only
            baseline_avg = statistics.mean(baseline_vals)
            baseline_z = zscore(baseline_avg, corpus_mean, corpus_stdev)
            music = (score_sample(model, baseline_gen, device, bigram_table)
                     if bigram_table is not None else {})
            base_bll_e, base_bll_l, base_gvr_e, base_gvr_l, base_mi_e, base_mi_l = early_late_scores(baseline_gen, bigram_table)
            wav = render_wav(baseline_gen, tokenizer, out_dir, f"p{pi}_baseline")
            print(f"[prompt {pi}] baseline: achieved={baseline_avg:.4f} (z={baseline_z:+.2f})  "
                  f"(repeats={baseline_vals})  music={music}  "
                  f"early/late bigram_ll={base_bll_e}/{base_bll_l}  "
                  f"early/late grammar_violation_rate={base_gvr_e}/{base_gvr_l}  "
                  f"early/late melodic_interval={base_mi_e}/{base_mi_l}")
            table_rows.append([
                pi, "baseline (no guidance)", 0.0, None, None, None,
                baseline_avg, baseline_avg, baseline_z,
                music.get("ebt_energy"), music.get("bigram_ll"), music.get("repetition_ratio"),
                music.get("grammar_violation_rate"),
                base_bll_e, base_bll_l, base_gvr_e, base_gvr_l, base_mi_e, base_mi_l,
                wandb.Audio(wav, caption=f"p{pi} baseline {attribute}={baseline_avg:.3f}"),
            ])

            # Resolve this prompt's target list: either the shared --targets
            # (same absolute values for every prompt), or --target_deltas
            # offsets applied to THIS prompt's own baseline — deliberately not
            # clamped to any "realistic" range, so e.g. an already-dense prompt
            # can be pushed even denser to see what happens.
            if args.target_deltas is not None:
                prompt_targets = [(max(0.0, baseline_avg + d), d) for d in args.target_deltas]
            else:
                prompt_targets = [(t, None) for t in targets]

            # ── Guided grid: same prompt, every (lambda, target) combination ───
            for lam in lambdas:
                for tgt, delta in prompt_targets:
                    hparams.attribute_target = tgt
                    hparams.lambda_attribute = lam
                    hparams.attribute_regressor_ckpt = args.regressor_checkpoint
                    with torch.no_grad():
                        out = generate_music(model, batch, hparams)
                    gen = out["generation_tokens"][0]
                    achieved = compute_fn(gen, tokenizer_type)
                    tgt_z = zscore(tgt, corpus_mean, corpus_stdev)
                    achieved_z = zscore(achieved, corpus_mean, corpus_stdev)
                    music = (score_sample(model, gen, device, bigram_table)
                             if bigram_table is not None else {})
                    bll_e, bll_l, gvr_e, gvr_l, mi_e, mi_l = early_late_scores(gen, bigram_table)
                    name = f"p{pi}_lam{lam}_tgt{tgt:.4f}".replace(".", "_")
                    wav = render_wav(gen, tokenizer, out_dir, name)
                    delta_str = f"  (delta={delta:+.3f})" if delta is not None else ""
                    print(f"[prompt {pi}] lambda={lam}  target={tgt:.4f} (z={tgt_z:+.2f}){delta_str}  "
                          f"achieved={achieved:.4f} (z={achieved_z:+.2f})  music={music}  "
                          f"early/late bigram_ll={bll_e}/{bll_l}  "
                          f"early/late grammar_violation_rate={gvr_e}/{gvr_l}  "
                          f"early/late melodic_interval={mi_e}/{mi_l}")
                    table_rows.append([
                        pi, f"λ={lam}", lam, tgt, delta, tgt_z,
                        baseline_avg, achieved, achieved_z,
                        music.get("ebt_energy"), music.get("bigram_ll"),
                        music.get("repetition_ratio"), music.get("grammar_violation_rate"),
                        bll_e, bll_l, gvr_e, gvr_l, mi_e, mi_l,
                        wandb.Audio(wav, caption=f"p{pi} λ={lam} target={tgt:.3f}"
                                                  f"{delta_str} achieved={achieved:.3f}"),
                    ])

            # Log incrementally after each prompt — long job, keep progress visible/safe.
            table = wandb.Table(columns=columns, data=table_rows)
            wandb.log({"listening_grid": table})
            print(f"[checkpoint] logged {len(table_rows)} samples through prompt {pi}")

        wandb.finish()
        print(f"\nWandB run: {run_name} (project={args.wandb_project})")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
