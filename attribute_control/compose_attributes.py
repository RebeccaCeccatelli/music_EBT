"""
Test R3's compositional "Reduce" step (Du et al. 2023): guide generation
toward MULTIPLE attribute targets simultaneously by summing their regressor
energies, rather than steering one attribute at a time as every other tool
in this project does. See inference/mus/generate_music.py's multi-attribute
_attribute_energy_fn for the actual composition mechanism.

For each prompt: one unguided baseline, then one composed-guidance sample
targeting all attributes at once. Reports every attribute's achieved value
against its own target, plus the usual coherence metrics, so a composition
failure (if any) shows up as either a specific attribute not tracking its
target anymore, or a coherence hit beyond what either attribute alone costs.

Usage:
    python attribute_control/compose_attributes.py \
        --regressor_checkpoints <ckpt1>,<ckpt2> \
        --target_deltas_std <-1.5,1.5> \
        --prompt_indices 8484,9938,12623

See also job_scripts/mus/attr_control/compose_attributes.sh.
"""

import sys
import argparse
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))

from inference.mus.infer_ebt import load_checkpoint, load_dataset
from inference.mus.generate_music import generate_music
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
from attribute_control.attributes import ATTRIBUTES
from attribute_control.musicality_metrics import build_bigram_logprob_table, score_sample
from attribute_control.listen_density_sweep import load_corpus_stats, render_wav, zscore

import wandb


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regressor_checkpoints", required=True,
                   help="Comma-separated regressor checkpoint paths, one per attribute to compose")
    p.add_argument("--target_deltas_std", default=None,
                   help="Comma-separated, one per regressor, in units of THAT attribute's own "
                        "corpus stdev, applied to this prompt's own baseline value. Ignored if "
                        "--all_sign_combos is set.")
    p.add_argument("--target_deltas_std_list", default=None,
                   help="Explicit list of delta-tuples to test, one combo per prompt's baseline: "
                        "'d0_0,d0_1;d1_0,d1_1;...' (semicolon-separated combos, comma-separated "
                        "across attributes) — e.g. an intensity-level sweep at hand-picked points "
                        "instead of every sign combination.")
    p.add_argument("--target_deltas_std_mag", default=None,
                   help="Comma-separated MAGNITUDES (one per regressor, always positive) used "
                        "with --all_sign_combos: every +/- combination of these magnitudes is "
                        "tested against the same baseline, e.g. 2 attributes -> 4 combos "
                        "(++, +-, -+, --).")
    p.add_argument("--all_sign_combos", action="store_true",
                   help="Test every +/- sign combination of --target_deltas_std_mag against "
                        "each baseline, instead of a single fixed --target_deltas_std")
    p.add_argument("--weights", default=None,
                   help="Comma-separated relative weights, one per regressor (default: 1.0 each) "
                        "— controls the mix between attributes, not the overall guidance strength")
    p.add_argument("--lam", type=float, default=1.0,
                   help="Overall lambda applied to the combined (summed) attribute gradient, "
                        "same interpretable scale as single-attribute lambda_attribute. Ignored "
                        "if --lams is set.")
    p.add_argument("--lams", default=None,
                   help="Comma-separated lambda values to sweep — every combo (or the single "
                        "--target_deltas_std) is tested at each lambda, against the SAME "
                        "baseline, so both 'which shared lambda is best overall' and 'does each "
                        "direction want a different lambda' can be read off one run.")
    p.add_argument("--prompt_indices", required=True)
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb_project", default="mus_symb_attr_control")
    p.add_argument("--wandb_run_name", default=None)
    args = p.parse_args()

    import itertools

    ckpt_paths = args.regressor_checkpoints.split(",")
    weights = [float(x) for x in args.weights.split(",")] if args.weights else [1.0] * len(ckpt_paths)
    assert len(ckpt_paths) == len(weights), \
        "regressor_checkpoints and weights must have the same length"

    if args.all_sign_combos:
        assert args.target_deltas_std_mag, "--all_sign_combos requires --target_deltas_std_mag"
        mags = [float(x) for x in args.target_deltas_std_mag.split(",")]
        assert len(mags) == len(ckpt_paths), "target_deltas_std_mag must have one value per regressor"
        # Each combo: a tuple of signed deltas, one per attribute, e.g. (+mag0, -mag1).
        delta_combos = [tuple(sign * m for sign, m in zip(signs, mags))
                         for signs in itertools.product([1, -1], repeat=len(mags))]
    elif args.target_deltas_std_list:
        # Explicit, curated list of delta tuples — e.g. an "intensity slider"
        # sweep at matched or independent levels per attribute, without the
        # combinatorial explosion of testing every level x every level.
        # Format: "d0_0,d0_1;d1_0,d1_1;..." (semicolon-separated combos, each
        # comma-separated across attributes).
        delta_combos = [tuple(float(x) for x in combo.split(","))
                         for combo in args.target_deltas_std_list.split(";")]
        for combo in delta_combos:
            assert len(combo) == len(ckpt_paths), \
                "each combo in target_deltas_std_list must have one value per regressor"
    else:
        assert args.target_deltas_std, "Set --target_deltas_std, --all_sign_combos, or --target_deltas_std_list"
        delta_combos = [tuple(float(x) for x in args.target_deltas_std.split(","))]
        assert len(delta_combos[0]) == len(ckpt_paths), \
            "target_deltas_std must have one value per regressor"

    lams = [float(x) for x in args.lams.split(",")] if args.lams else [args.lam]

    device = args.device
    reg_ckpts = [torch.load(cp, map_location="cpu", weights_only=False) for cp in ckpt_paths]
    ebt_checkpoint = reg_ckpts[0]["ebt_checkpoint"]
    tokenizer_type = reg_ckpts[0].get("tokenizer_type", "REMI")
    attr_names = [rc.get("attribute", "density") for rc in reg_ckpts]
    compute_fns = [ATTRIBUTES[n] for n in attr_names]
    corpus_stats = [load_corpus_stats(n) for n in attr_names]

    print(f"Composing attributes: {attr_names}")
    print(f"Weights: {weights}   λ={args.lam}")
    print(f"EBT ckpt: {ebt_checkpoint}")

    model, hparams = load_checkpoint(ebt_checkpoint, device)
    hparams.device = device
    hparams.infer_max_gen_len = args.gen_len
    hparams.infer_temp = args.temperature
    hparams.infer_topp = args.top_p
    hparams.infer_logprobs = False
    hparams.infer_echo = False
    hparams.infer_ebt_advanced = False
    hparams.tokenizer_type = tokenizer_type
    hparams.attr_gate_by_token_type = True

    tokenizer, vocab_size, pad_token_id = load_tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=getattr(hparams, "tokenizer_config_path", None),
        dataset_name=getattr(hparams, "dataset_name", "giga_midi"),
    )
    dataset = load_dataset(hparams, split=args.split)
    bigram_table = build_bigram_logprob_table(
        dataset, model.embeddings.weight.shape[0], n_songs=3000, seed=0,
        cache_path=str(Path(ckpt_paths[0]).parent.parent / f"_bigram_table_{tokenizer_type}.npy"),
    )

    out_dir = Path(tempfile.mkdtemp(prefix="compose_attributes_"))
    run_name = args.wandb_run_name or f"compose-{'_'.join(attr_names)}"
    wandb.init(project=args.wandb_project, name=run_name, job_type="compose_attributes",
               config=vars(args))
    columns = ["prompt_id", "condition"]
    for n in attr_names:
        columns += [f"{n}_achieved", f"{n}_target", f"{n}_zscore"]
    columns += ["bigram_ll", "repetition_ratio", "grammar_violation_rate", "audio"]
    table_rows = []

    prompt_indices = [int(x) for x in args.prompt_indices.split(",")]
    for pi in prompt_indices:
        full_tokens = dataset.get_full_tokens(pi)
        prompt = full_tokens[: min(args.prompt_len, len(full_tokens))]
        batch = {"input_ids": torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)}

        # ── Unguided baseline ────────────────────────────────────────────────
        hparams.attribute_regressor_ckpts = None
        hparams.attribute_targets = None
        hparams.lambda_attribute = 0.0
        with torch.no_grad():
            out = generate_music(model, batch, hparams)
        base_gen = out["generation_tokens"][0]
        base_vals = [fn(base_gen, tokenizer_type) for fn in compute_fns]
        base_music = score_sample(model, base_gen, device, bigram_table)
        base_wav = render_wav(base_gen, tokenizer, out_dir, f"p{pi}_baseline")
        print(f"[prompt {pi}] baseline: " +
              "  ".join(f"{n}={v:.4f}" for n, v in zip(attr_names, base_vals)) +
              f"  music={base_music}")
        row = [pi, "baseline"]
        for n, v, cs in zip(attr_names, base_vals, corpus_stats):
            row += [v, None, zscore(v, cs["mean"], cs["stdev"])]
        row += [base_music.get("bigram_ll"), base_music.get("repetition_ratio"),
                base_music.get("grammar_violation_rate"),
                wandb.Audio(base_wav, caption=f"p{pi} baseline")]
        table_rows.append(row)

        # ── Composed guidance: one run per (sign combo, lambda) pair, all
        # attributes at once, all against this SAME baseline ──────────────────
        for combo_idx, deltas_std in enumerate(delta_combos):
            # Targets: THIS prompt's own baseline + delta_std * that attribute's
            # own corpus stdev (clamped >=0, matching listen_density_sweep.py's convention).
            targets = [max(0.0, bv + d * cs["stdev"])
                       for bv, d, cs in zip(base_vals, deltas_std, corpus_stats)]
            combo_label = "/".join(f"{n}{d:+.2f}std" for n, d in zip(attr_names, deltas_std))

            for lam in lams:
                hparams.attribute_regressor_ckpts = ckpt_paths
                hparams.attribute_targets = targets
                hparams.attribute_weights = weights
                hparams.lambda_attribute = lam
                with torch.no_grad():
                    out = generate_music(model, batch, hparams)
                gen = out["generation_tokens"][0]
                vals = [fn(gen, tokenizer_type) for fn in compute_fns]
                music = score_sample(model, gen, device, bigram_table)
                name = f"p{pi}_{combo_label.replace('/', '_')}_lam{lam}".replace(".", "_")
                wav = render_wav(gen, tokenizer, out_dir, name)
                print(f"[prompt {pi}] composed [{combo_label}] (λ={lam}): " +
                      "  ".join(f"{n}={v:.4f} target={t:.4f}"
                                for n, v, t in zip(attr_names, vals, targets)) +
                      f"  music={music}")
                row = [pi, f"composed [{combo_label}] λ={lam}"]
                for n, v, t, cs in zip(attr_names, vals, targets, corpus_stats):
                    row += [v, t, zscore(v, cs["mean"], cs["stdev"])]
                row += [music.get("bigram_ll"), music.get("repetition_ratio"),
                        music.get("grammar_violation_rate"),
                        wandb.Audio(wav, caption=f"p{pi} [{combo_label}] λ={lam}")]
                table_rows.append(row)

        table = wandb.Table(columns=columns, data=table_rows)
        wandb.log({"compose_comparison": table})
        print(f"[checkpoint] logged through prompt {pi}")

    wandb.finish()
    print(f"\nWandB run: {run_name} (project={args.wandb_project})")


if __name__ == "__main__":
    main()
