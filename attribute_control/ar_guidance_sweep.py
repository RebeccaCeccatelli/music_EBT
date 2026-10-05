"""
Plug-and-play attribute-guidance sweep for the frozen AR baselines (GPT-2,
Llama) — the head-to-head counterpart to listen_density_sweep.py's EBT
sweeps (docs/thesis_findings/2026-09-24_remi_guidance_strength_sweeps.md).

Same protocol as the EBT sweeps so the numbers are directly comparable:
same prompt pool, prompt/generation length, sampling settings, per-prompt
baseline (mean of --baseline_repeats unguided draws), targets as offsets of
that baseline in corpus-stdev units, one sample per (prompt, strength,
target), and the same metrics (directional accuracy, MAE vs. target,
bigram_ll, grammar_violation_rate — see aggregate_guidance_sweeps.py).

Methods (attribute_control/ar_guidance.py has the details):
  --method tilt       strength = fraction of the per-step gap to the target
                      closed by exponential tilting (0-1). REMI, velocity/
                      duration/pitch_register only. No regressor needed.
  --method best_of_n  strength = N. One pool of max(N) unguided samples per
                      prompt is drawn once and shared by every target and N
                      (nested: best-of-4 is chosen from the pool's first 4).
  --method pplm       strength = lambda, the L2 norm of the next-token logit
                      perturbation (raw logit units — NOT EBT's lambda scale).
                      Llama only; needs a regressor trained on Llama's
                      embeddings (CHECKPOINT=<llama ckpt> for train_density.sh).

Several attributes can share one run (--attributes): the per-prompt baseline
draws (and best_of_n's pool) are scored for all of them, so they're only
generated once.

Output (--out_dir): one <model>_<method>_<attribute>.table.json per attribute
in the same {"columns", "data"} layout as attribute_control/sweep_tables/
(wandb Table JSON), plus the run config. wandb logging is optional.

Usage:
    python attribute_control/ar_guidance_sweep.py \
        --model_checkpoint <llama_or_gpt2.ckpt> --method tilt \
        --attributes velocity,duration,pitch_register \
        --prompt_indices_file <clean_melodic_prompts.json> --n_prompts 16 \
        --out_dir <dir>

See also job_scripts/mus/attr_control/ar_guidance_sweep.sh.
"""

import sys
import json
import random
import argparse
import statistics
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.mus.infer_ebt import load_checkpoint, load_dataset
from inference.mus.generate_music import generate_music
from attribute_control.attributes import ATTRIBUTES
from attribute_control.ar_guidance import ExpectationTilt
from attribute_control.musicality_metrics import build_bigram_logprob_table, score_sample
from attribute_control.listen_density_sweep import load_corpus_stats, zscore


# Default per-method strength grids, chosen to span "barely moves" to
# "clearly saturated" the same way the EBT sweeps' per-attribute λ grids do.
DEFAULT_STRENGTHS = {
    'tilt': '0.05,0.1,0.2,0.3,0.5,0.75,1.0',
    'best_of_n': '2,4,8,16',
    'pplm': '1,2,4,8,16,32',
}

COLUMNS = ["prompt_id", "condition", "method", "model", "attribute", "lambda",
           "target", "target_delta", "target_zscore",
           "baseline_value", "achieved_value", "achieved_zscore",
           "bigram_ll", "repetition_ratio", "grammar_violation_rate", "n_tilted_steps"]


def generate(model, batch, hparams, logit_processor=None):
    hparams.logit_processor = logit_processor
    try:
        with torch.no_grad():
            out = generate_music(model, batch, hparams)
    finally:
        hparams.logit_processor = None
    return out["generation_tokens"][0]


def set_guidance_off(hparams):
    hparams.attribute_target = None
    hparams.lambda_attribute = 0.0
    hparams.attribute_regressor_ckpt = None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_checkpoint", required=True, help="GPT-2 or Llama baseline checkpoint")
    p.add_argument("--method", required=True, choices=["tilt", "best_of_n", "pplm"])
    p.add_argument("--attributes", default="velocity,duration,pitch_register")
    p.add_argument("--strengths", default=None,
                   help="Comma-separated strength grid (meaning depends on --method; "
                        "defaults in DEFAULT_STRENGTHS)")
    p.add_argument("--regressor_checkpoints", default=None,
                   help="pplm only: comma-separated regressor best.pt paths, one per "
                        "--attributes entry, trained on THIS model's embeddings")
    p.add_argument("--n_prompts", type=int, default=16)
    p.add_argument("--prompt_indices", type=str, default=None)
    p.add_argument("--prompt_indices_file", type=str, default=None)
    p.add_argument("--target_deltas_std", type=str, default="-2,-1,-0.5,0.5,1,2")
    p.add_argument("--baseline_repeats", type=int, default=3)
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="validation")
    p.add_argument("--no_step_gating", action="store_true",
                   help="tilt only: tilt on every step whose distribution puts mass on "
                        "the attribute's tokens, not just the gated ones")
    p.add_argument("--bigram_table", default=None,
                   help="Cached bigram log-prob table (.npy); built from the dataset if missing")
    p.add_argument("--no_musicality", action="store_true")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--wandb_project", default=None, help="Omit to disable wandb logging")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    attributes = [a.strip() for a in args.attributes.split(",") if a.strip()]
    strengths = [float(s) for s in (args.strengths or DEFAULT_STRENGTHS[args.method]).split(",")]
    if args.method == "best_of_n":
        strengths = [int(s) for s in strengths]
    deltas_std = [float(d) for d in args.target_deltas_std.split(",")]
    for a in attributes:
        if a not in ATTRIBUTES:
            raise ValueError(f"Unknown attribute '{a}'")

    regressors = {}
    if args.method == "pplm":
        if not args.regressor_checkpoints:
            raise ValueError("--method pplm needs --regressor_checkpoints (one per attribute)")
        reg_paths = [r.strip() for r in args.regressor_checkpoints.split(",")]
        if len(reg_paths) != len(attributes):
            raise ValueError("--regressor_checkpoints must have one entry per --attributes entry")
        for a, rp in zip(attributes, reg_paths):
            meta = torch.load(rp, map_location="cpu", weights_only=False)
            if meta.get("attribute") != a:
                raise ValueError(f"Regressor {rp} is for '{meta.get('attribute')}', not '{a}'")
            # The regressor's embedding space must be this model's own — a
            # mismatched one silently produces meaningless gradients.
            if Path(meta.get("ebt_checkpoint", "")).resolve() != Path(args.model_checkpoint).resolve():
                print(f"⚠️ Regressor {rp} was trained on {meta.get('ebt_checkpoint')}, "
                      f"not {args.model_checkpoint}")
            regressors[a] = rp

    torch.manual_seed(args.seed)
    device = args.device
    model, hparams = load_checkpoint(args.model_checkpoint, device)
    model_name = hparams.model_name
    if model_name == "ebt":
        raise ValueError("This sweep is for the AR baselines; EBT uses listen_density_sweep.py")
    if args.method == "pplm" and model_name != "baseline_llama_transformer":
        raise ValueError("--method pplm is only implemented for Llama "
                         "(call_model_forward_decode has no GPT-2 gradient branch)")
    tokenizer_type = hparams.tokenizer_type
    if args.method == "tilt" and tokenizer_type != "REMI":
        raise ValueError("--method tilt is REMI-only (its token->value maps are REMI ids)")
    model_slug = {"baseline_llama_transformer": "llama",
                  "baseline_hf_gpt2_transformer": "gpt2"}.get(model_name, model_name)

    hparams.device = device
    hparams.infer_max_gen_len = args.gen_len
    hparams.infer_temp = args.temperature
    hparams.infer_topp = args.top_p
    hparams.infer_logprobs = False
    hparams.infer_echo = False
    hparams.infer_ebt_advanced = False
    hparams.attr_gate_by_token_type = not args.no_step_gating
    set_guidance_off(hparams)

    corpus = {a: load_corpus_stats(a) for a in attributes}
    dataset = load_dataset(hparams, split=args.split)
    rng = random.Random(args.seed)
    if args.prompt_indices:
        prompt_ids = [int(x) for x in args.prompt_indices.split(",")]
    elif args.prompt_indices_file:
        with open(args.prompt_indices_file) as f:
            pool = json.load(f)
        prompt_ids = rng.sample(pool, min(args.n_prompts, len(pool)))
    else:
        prompt_ids = rng.sample(range(len(dataset)), min(args.n_prompts, len(dataset)))
    print(f"Model: {model_name}  method: {args.method}  attributes: {attributes}")
    print(f"Strengths: {strengths}  target deltas (stdev): {deltas_std}")
    print(f"Prompts ({len(prompt_ids)}): {prompt_ids}")

    bigram_table = None
    if not args.no_musicality and tokenizer_type == "REMI":
        # Same table (and so the same bigram_ll scale) as the EBT sweeps when
        # pointed at their cache — the REMI vocab is shared across models.
        bigram_table = build_bigram_logprob_table(
            dataset, model.vocab_size, n_songs=3000, seed=0, cache_path=args.bigram_table)

    def music(gen):
        if bigram_table is None:
            return {}
        return score_sample(model, gen, device, bigram_table, tokenizer_type)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model_slug}_{args.method}.config.json").write_text(json.dumps(
        {**vars(args), "model_name": model_name, "tokenizer_type": tokenizer_type,
         "prompt_ids": prompt_ids, "strengths": strengths}, indent=2))
    rows = {a: [] for a in attributes}

    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, job_type="ar_guidance_sweep",
                   name=args.wandb_run_name or f"ar-{model_slug}-{args.method}",
                   config={**vars(args), "model_name": model_name, "prompt_ids": prompt_ids})

    def row(a, pi, condition, strength, tgt, delta, base, gen, n_tilted=None):
        achieved = ATTRIBUTES[a](gen, tokenizer_type)
        m = music(gen)
        mean, sd = corpus[a]["mean"], corpus[a]["stdev"]
        return [pi, condition, args.method, model_slug, a, strength,
                tgt, delta, zscore(tgt, mean, sd), base, achieved, zscore(achieved, mean, sd),
                m.get("bigram_ll"), m.get("repetition_ratio"), m.get("grammar_violation_rate"),
                n_tilted]

    def save():
        for a in attributes:
            path = out_dir / f"{model_slug}_{args.method}_{a}.table.json"
            path.write_text(json.dumps({"columns": COLUMNS, "data": rows[a]}))
        if args.wandb_project:
            for a in attributes:
                wandb.log({f"grid/{a}": wandb.Table(columns=COLUMNS, data=rows[a])})

    for pi in prompt_ids:
        full_tokens = dataset.get_full_tokens(pi)
        prompt = full_tokens[: min(args.prompt_len, len(full_tokens))]
        batch = {"input_ids": torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)}

        gt = full_tokens[len(prompt): len(prompt) + args.gen_len]
        if gt:
            for a in attributes:
                rows[a].append(row(a, pi, "ground truth (real song)", None, None, None, None, gt))

        # ── Per-prompt unguided baseline (anchors every target) ──────────────
        set_guidance_off(hparams)
        base_gens = [generate(model, batch, hparams) for _ in range(args.baseline_repeats)]
        baselines = {a: statistics.mean(ATTRIBUTES[a](g, tokenizer_type) for g in base_gens)
                     for a in attributes}
        for a in attributes:
            rows[a].append(row(a, pi, "baseline (no guidance)", 0.0, None, None,
                               baselines[a], base_gens[0]))
        print(f"[prompt {pi}] baseline: " +
              "  ".join(f"{a}={baselines[a]:.4f}" for a in attributes))

        # best_of_n: one shared pool per prompt, scored for every attribute.
        pool = ([generate(model, batch, hparams) for _ in range(max(strengths))]
                if args.method == "best_of_n" else None)
        pool_vals = ({a: [ATTRIBUTES[a](g, tokenizer_type) for g in pool] for a in attributes}
                     if pool is not None else None)

        for a in attributes:
            sd = corpus[a]["stdev"]
            for strength in strengths:
                for d_std in deltas_std:
                    delta = d_std * sd
                    tgt = max(0.0, baselines[a] + delta)
                    n_tilted = None
                    if args.method == "tilt":
                        proc = ExpectationTilt(a, tgt, strength, args.temperature,
                                               gate_by_token_type=not args.no_step_gating)
                        gen = generate(model, batch, hparams, logit_processor=proc)
                        n_tilted = proc.n_tilted_steps
                    elif args.method == "best_of_n":
                        cands = pool_vals[a][:strength]
                        best = min(range(len(cands)), key=lambda i: abs(cands[i] - tgt))
                        gen = pool[best]
                    else:  # pplm
                        hparams.attribute_target = tgt
                        hparams.lambda_attribute = strength
                        hparams.attribute_regressor_ckpt = regressors[a]
                        gen = generate(model, batch, hparams)
                        set_guidance_off(hparams)
                    r = row(a, pi, f"{args.method}={strength}", strength, tgt, delta,
                            baselines[a], gen, n_tilted)
                    rows[a].append(r)
                    print(f"[prompt {pi}] {a} {args.method}={strength} delta={d_std:+g}sd "
                          f"target={tgt:.4f} achieved={r[10]:.4f} bigram_ll={r[12]}")
        save()
        print(f"[checkpoint] saved through prompt {pi}")

    if args.wandb_project:
        wandb.finish()
    print(f"\nDone. Tables in {out_dir}")


if __name__ == "__main__":
    main()
