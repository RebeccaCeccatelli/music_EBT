"""
Multi-attribute (compositional) guidance sweep for every method under one
protocol: EBT energy composition (R³'s "Reduce" step, Du et al. 2023) and the
frozen-baseline methods of attribute_control/ar_guidance.py, each steering
several attributes at once.

  --method r3         EBT: the regressor energies are summed inside the MCMC
                      refinement; strength = shared λ.
  --method pplm       Llama: the same summed regressor energy, one normalised
                      gradient step on the next-token logits; strength = the
                      perturbation norm. Regressors on Llama's own embeddings.
  --method tilt       Llama/GPT-2, REMI only: one ExpectationTilt per attribute,
                      chained (each fires only at its own token type, so they
                      never act on the same step); strength = tilt fraction.
  --method best_of_n  any model: N unguided samples, keep the one with the
                      smallest summed |achieved − target| in corpus-SD units;
                      strength = N (nested, from one shared pool per prompt).

Protocol (same as the single-attribute sweeps): per prompt, the baseline is the
mean of --baseline_repeats unguided draws; every combination of signs of the
--target_deltas_std_mag magnitudes gives one joint target (e.g. 2 attributes ×
magnitudes 1,2 → 8 targets: ±1/±1, ±1/∓1, ±2/±2, ±2/∓2); one sample per
(prompt, strength, target).

Output (--out_dir): one <model>_<method>_compose_<attribute>.table.json per
attribute in the usual {columns, data} layout (one row per sample and
attribute, plus a "combo" column), so aggregate_guidance_sweeps.py and
eval/score_sweep_quality.py read them unchanged. Each sample is saved as MIDI
under midi/<attribute>/ for each attribute. A summary with joint metrics (all
attributes moved the requested way in the same sample) is printed and saved as
<model>_<method>_compose.summary.json.

Usage:
    python attribute_control/compose_guidance_sweep.py --model_checkpoint <ckpt> \\
        --method pplm --attributes velocity,duration \\
        --regressor_checkpoints <vel best.pt>,<dur best.pt> --out_dir <dir>
See job_scripts/mus/attr_control/compose_guidance_sweep.sh.
"""

import sys
import json
import random
import argparse
import itertools
import statistics
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.mus.infer_ebt import load_checkpoint, load_dataset
from inference.mus.generate_music import generate_music
from attribute_control.attributes import ATTRIBUTES
from attribute_control.ar_guidance import ExpectationTilt
from attribute_control.sample_midi import save_sample_midi
from attribute_control.ar_guidance_sweep import COLUMNS
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
from attribute_control.musicality_metrics import build_bigram_logprob_table, score_sample
from attribute_control.listen_density_sweep import load_corpus_stats, zscore

DEFAULT_STRENGTHS = {
    "r3": "0.01,0.02,0.03,0.04,0.05",
    "pplm": "1,2,4,8,16,32",
    "tilt": "0.1,0.2,0.3,0.5,0.75,1.0",
    "best_of_n": "2,4,8,16",
}
COMPOSE_COLUMNS = COLUMNS + ["combo"]


class ChainedProcessor:
    """Apply several logit processors in turn (each gates itself)."""

    def __init__(self, processors):
        self.processors = processors

    def __call__(self, context, last_logits):
        for p in self.processors:
            last_logits = p(context, last_logits)
        return last_logits


def guidance_off(hparams):
    hparams.attribute_target = None
    hparams.lambda_attribute = 0.0
    hparams.attribute_regressor_ckpt = None
    hparams.attribute_regressor_ckpts = None
    hparams.attribute_targets = None
    hparams.attribute_weights = None
    hparams.logit_processor = None


def generate(model, batch, hparams):
    with torch.no_grad():
        out = generate_music(model, batch, hparams)
    return [int(t) for t in out["generation_tokens"][0]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_checkpoint", required=True)
    p.add_argument("--method", required=True, choices=list(DEFAULT_STRENGTHS))
    p.add_argument("--attributes", default="velocity,duration")
    p.add_argument("--strengths", default=None)
    p.add_argument("--regressor_checkpoints", default=None,
                   help="r3/pplm: one best.pt per --attributes entry, on THIS model's embeddings")
    p.add_argument("--target_deltas_std_mag", default="1,2",
                   help="Magnitudes in corpus-SD units; every sign combination of each is tested")
    p.add_argument("--n_prompts", type=int, default=16)
    p.add_argument("--prompt_indices", type=str, default=None)
    p.add_argument("--prompt_indices_file", type=str, default=None)
    p.add_argument("--baseline_repeats", type=int, default=3)
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="validation")
    p.add_argument("--bigram_table", default=None)
    p.add_argument("--no_musicality", action="store_true")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    attributes = [a.strip() for a in args.attributes.split(",") if a.strip()]
    if len(attributes) < 2:
        raise ValueError("Composition needs at least two --attributes")
    for a in attributes:
        if a not in ATTRIBUTES:
            raise ValueError(f"Unknown attribute '{a}'")
    strengths = [float(s) for s in (args.strengths or DEFAULT_STRENGTHS[args.method]).split(",")]
    if args.method == "best_of_n":
        strengths = [int(s) for s in strengths]
    mags = [float(m) for m in args.target_deltas_std_mag.split(",")]
    combos = [tuple(m * s for s in signs)
              for m in mags for signs in itertools.product((1, -1), repeat=len(attributes))]

    regressors = []
    if args.method in ("r3", "pplm"):
        if not args.regressor_checkpoints:
            raise ValueError(f"--method {args.method} needs --regressor_checkpoints")
        regressors = [r.strip() for r in args.regressor_checkpoints.split(",")]
        if len(regressors) != len(attributes):
            raise ValueError("--regressor_checkpoints needs one entry per attribute")
        for a, rp in zip(attributes, regressors):
            meta = torch.load(rp, map_location="cpu", weights_only=False)
            if meta.get("attribute") != a:
                raise ValueError(f"Regressor {rp} is for '{meta.get('attribute')}', not '{a}'")
            if Path(meta.get("ebt_checkpoint", "")).resolve() != Path(args.model_checkpoint).resolve():
                print(f"⚠️ Regressor {rp} was trained on {meta.get('ebt_checkpoint')}, "
                      f"not {args.model_checkpoint}")

    torch.manual_seed(args.seed)
    device = args.device
    model, hparams = load_checkpoint(args.model_checkpoint, device)
    model_name = hparams.model_name
    model_slug = {"ebt": "ebt", "baseline_llama_transformer": "llama",
                  "baseline_hf_gpt2_transformer": "gpt2"}.get(model_name, model_name)
    tokenizer_type = hparams.tokenizer_type
    if (args.method == "r3") != (model_name == "ebt"):
        raise ValueError("--method r3 is for EBT checkpoints; baselines use pplm/tilt/best_of_n")
    if args.method == "pplm" and model_name != "baseline_llama_transformer":
        raise ValueError("--method pplm is only implemented for Llama")
    if args.method == "tilt" and tokenizer_type != "REMI":
        raise ValueError("--method tilt is REMI-only")

    hparams.device = device
    hparams.infer_max_gen_len = args.gen_len
    hparams.infer_temp = args.temperature
    hparams.infer_topp = args.top_p
    hparams.infer_logprobs = False
    hparams.infer_echo = False
    hparams.infer_ebt_advanced = False
    hparams.attr_gate_by_token_type = True
    guidance_off(hparams)

    corpus = {a: load_corpus_stats(a) for a in attributes}
    dataset = load_dataset(hparams, split=args.split)
    rng = random.Random(args.seed)
    if args.prompt_indices:
        prompt_ids = [int(x) for x in args.prompt_indices.split(",")]
    elif args.prompt_indices_file:
        pool = json.loads(Path(args.prompt_indices_file).read_text())
        prompt_ids = rng.sample(pool, min(args.n_prompts, len(pool)))
    else:
        prompt_ids = rng.sample(range(len(dataset)), min(args.n_prompts, len(dataset)))
    print(f"Model: {model_name}  method: {args.method}  attributes: {attributes}")
    print(f"Strengths: {strengths}  targets (SD units): {combos}")
    print(f"Prompts ({len(prompt_ids)}): {prompt_ids}")

    bigram_table = None
    if not args.no_musicality and tokenizer_type == "REMI":
        bigram_table = build_bigram_logprob_table(
            dataset, model.vocab_size, n_songs=3000, seed=0, cache_path=args.bigram_table)

    def music(gen):
        return score_sample(model, gen, device, bigram_table, tokenizer_type) if bigram_table is not None else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, _, _ = load_tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=getattr(hparams, "tokenizer_config_path", None),
        dataset_name=getattr(hparams, "dataset_name", "giga_midi"))

    def save_midi(sample_id, prompt, gen, kind="generated", subdirs=None):
        for sub in (subdirs or attributes):
            save_sample_midi(out_dir / "midi" / sub, sample_id, prompt, gen, tokenizer, kind)

    stem = f"{model_slug}_{args.method}_compose"
    (out_dir / f"{stem}.config.json").write_text(json.dumps(
        {**vars(args), "model_name": model_name, "tokenizer_type": tokenizer_type,
         "prompt_ids": prompt_ids, "strengths": strengths, "combos": combos}, indent=2))
    rows = {a: [] for a in attributes}
    joint = []  # one record per guided sample: per-attribute correctness and SD error

    def add_rows(pi, condition, strength, targets, deltas, baselines, gen, sample_id, combo):
        m = music(gen)
        achieved = {}
        for a in attributes:
            mean, sd = corpus[a]["mean"], corpus[a]["stdev"]
            val = ATTRIBUTES[a](gen, tokenizer_type)
            achieved[a] = val
            tgt = targets[a] if targets else None
            rows[a].append([pi, condition, args.method, model_slug, a, strength,
                            tgt, deltas[a] if deltas else None,
                            zscore(tgt, mean, sd) if tgt is not None else None,
                            baselines[a] if baselines else None, val, zscore(val, mean, sd),
                            m.get("bigram_ll"), m.get("repetition_ratio"),
                            m.get("grammar_violation_rate"), None, sample_id,
                            list(combo) if combo else None])
        return achieved

    def save():
        for a in attributes:
            (out_dir / f"{stem}_{a}.table.json").write_text(
                json.dumps({"columns": COMPOSE_COLUMNS, "data": rows[a]}))
        summary = {}
        for s in strengths:
            rs = [r for r in joint if r["strength"] == s]
            if not rs:
                continue
            summary[str(s)] = {
                "n": len(rs),
                "joint_accuracy": statistics.mean(all(r["hit"].values()) for r in rs),
                **{f"accuracy_{a}": statistics.mean(r["hit"][a] for r in rs) for a in attributes},
                **{f"abs_err_sd_{a}": statistics.mean(r["err_sd"][a] for r in rs) for a in attributes},
                "joint_accuracy_conflicting": (statistics.mean(all(r["hit"].values()) for r in rs
                                                               if len({c > 0 for c in r["combo"]}) > 1)
                                               if any(len({c > 0 for c in r["combo"]}) > 1 for r in rs) else None),
            }
        (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    for pi in prompt_ids:
        full = dataset.get_full_tokens(pi)
        prompt = full[: min(args.prompt_len, len(full))]
        batch = {"input_ids": torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)}
        gt = full[len(prompt): len(prompt) + args.gen_len]
        if gt:
            save_midi(f"p{pi}", prompt, gt, kind="ground_truth", subdirs=["ground_truth"])
            add_rows(pi, "ground truth (real song)", None, None, None, None, gt, f"p{pi}", None)

        guidance_off(hparams)
        base_gens = [generate(model, batch, hparams) for _ in range(args.baseline_repeats)]
        baselines = {a: statistics.mean(ATTRIBUTES[a](g, tokenizer_type) for g in base_gens)
                     for a in attributes}
        for k, g in enumerate(base_gens):
            save_midi(f"p{pi}_r{k}", prompt, g, subdirs=["baseline"])
        add_rows(pi, "baseline (no guidance)", 0.0, None, None, baselines, base_gens[0], f"p{pi}_r0", None)
        print(f"[prompt {pi}] baseline: " + "  ".join(f"{a}={baselines[a]:.4f}" for a in attributes))

        pool = pool_vals = None
        if args.method == "best_of_n":
            pool = [generate(model, batch, hparams) for _ in range(max(strengths))]
            pool_vals = [{a: ATTRIBUTES[a](g, tokenizer_type) for a in attributes} for g in pool]

        for strength in strengths:
            for combo in combos:
                deltas = {a: d * corpus[a]["stdev"] for a, d in zip(attributes, combo)}
                targets = {a: max(0.0, baselines[a] + deltas[a]) for a in attributes}
                if args.method == "best_of_n":
                    def dist(v):
                        return sum(abs(v[a] - targets[a]) / corpus[a]["stdev"] for a in attributes)
                    best = min(range(strength), key=lambda i: dist(pool_vals[i]))
                    gen = pool[best]
                else:
                    guidance_off(hparams)
                    if args.method in ("r3", "pplm"):
                        hparams.attribute_regressor_ckpts = regressors
                        hparams.attribute_targets = [targets[a] for a in attributes]
                        hparams.attribute_weights = [1.0] * len(attributes)
                        hparams.lambda_attribute = strength
                    else:  # tilt
                        hparams.logit_processor = ChainedProcessor(
                            [ExpectationTilt(a, targets[a], strength, args.temperature) for a in attributes])
                    gen = generate(model, batch, hparams)
                    guidance_off(hparams)
                cname = "_".join(f"{d:+g}" for d in combo)
                sample_id = f"p{pi}_{args.method}{strength:g}_c{cname}"
                save_midi(sample_id, prompt, gen)
                achieved = add_rows(pi, f"{args.method}={strength} combo={cname}", strength,
                                    targets, deltas, baselines, gen, sample_id, combo)
                hit = {a: (achieved[a] - baselines[a]) * deltas[a] > 0 for a in attributes}
                err = {a: abs(achieved[a] - targets[a]) / corpus[a]["stdev"] for a in attributes}
                joint.append({"strength": strength, "combo": list(combo), "hit": hit, "err_sd": err})
                print(f"[prompt {pi}] {args.method}={strength} combo={cname} " +
                      "  ".join(f"{a}: {achieved[a]:.4f}→{targets[a]:.4f} {'✓' if hit[a] else '✗'}"
                                for a in attributes))
        summary = save()
        print(f"[checkpoint] saved through prompt {pi}")

    summary = save()
    print("\nJoint accuracy by strength (all attributes moved the requested way):")
    for s, m in summary.items():
        conf = m["joint_accuracy_conflicting"]
        print(f"  {args.method}={s:>6}  joint {m['joint_accuracy']:.3f}  "
              f"conflicting-direction {conf:.3f}  " +
              "  ".join(f"{a} {m[f'accuracy_{a}']:.3f}" for a in attributes))
    print(f"\nDone. Tables in {out_dir}")


if __name__ == "__main__":
    main()
