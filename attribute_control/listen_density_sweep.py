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
import shutil
import random
import argparse
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))

from inference.mus.infer_ebt import load_checkpoint, load_dataset
from inference.mus.generate_music import generate_music
from inference.mus.tokens_to_midi import tokens_to_midi
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
from attribute_control.note_density import compute_density
from attribute_control.musicality_metrics import build_bigram_logprob_table, score_sample
from convert_midi_simple import simple_synth

import wandb


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
    p.add_argument("--ebt_checkpoint", default=None,
                   help="Override the EBT checkpoint (default: read from regressor metadata)")
    p.add_argument("--tokenizer_type", default=None,
                   help="Override tokenizer type (default: read from regressor metadata)")
    p.add_argument("--n_prompts", type=int, default=5)
    p.add_argument("--targets", type=str, default="0.05,0.10,0.15,0.20,0.25")
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
    args = p.parse_args()
    targets = [float(t) for t in args.targets.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]

    device = args.device
    print(f"Device: {device}")

    reg_ckpt = torch.load(args.regressor_checkpoint, map_location="cpu", weights_only=False)
    ebt_checkpoint = args.ebt_checkpoint or reg_ckpt["ebt_checkpoint"]
    tokenizer_type = args.tokenizer_type or reg_ckpt.get("tokenizer_type", "REMI")
    print(f"Regressor:  {args.regressor_checkpoint}")
    print(f"EBT ckpt:   {ebt_checkpoint}")

    model, hparams = load_checkpoint(ebt_checkpoint, device)
    hparams.device = device
    hparams.infer_max_gen_len = args.gen_len
    hparams.infer_temp = args.temperature
    hparams.infer_topp = args.top_p
    hparams.infer_logprobs = False
    hparams.infer_echo = False
    hparams.infer_ebt_advanced = False
    hparams.tokenizer_type = tokenizer_type

    tokenizer, vocab_size, pad_token_id = load_tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=getattr(hparams, "tokenizer_config_path", None),
        dataset_name=getattr(hparams, "dataset_name", "giga_midi"),
    )

    dataset = load_dataset(hparams, split=args.split)
    rng = random.Random(args.seed)
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

    columns = ["prompt_id", "condition", "lambda", "target", "achieved_density",
               "ebt_energy", "bigram_ll", "repetition_ratio", "audio"]
    table_rows = []

    try:
        for pi in sample_indices:
            full_tokens = dataset.get_full_tokens(pi)
            prompt = full_tokens[: min(args.prompt_len, len(full_tokens))]
            batch = {
                "input_ids": torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
            }

            # ── Unguided baseline: the "outputs are good" reference point ──────
            hparams.density_target = None
            hparams.lambda_density = 0.0
            hparams.density_regressor_ckpt = None
            with torch.no_grad():
                out = generate_music(model, batch, hparams)
            gen = out["generation_tokens"][0]
            achieved = compute_density(gen, tokenizer_type)
            music = score_sample(model, gen, device, bigram_table) if bigram_table is not None else {}
            wav = render_wav(prompt + gen, tokenizer, out_dir, f"p{pi}_baseline")
            print(f"[prompt {pi}] baseline: achieved={achieved:.4f}  music={music}")
            table_rows.append([
                pi, "baseline (no guidance)", 0.0, None, achieved,
                music.get("ebt_energy"), music.get("bigram_ll"), music.get("repetition_ratio"),
                wandb.Audio(wav, caption=f"p{pi} baseline density={achieved:.3f}"),
            ])

            # ── Guided grid: same prompt, every (lambda, target) combination ───
            for lam in lambdas:
                for tgt in targets:
                    hparams.density_target = tgt
                    hparams.lambda_density = lam
                    hparams.density_regressor_ckpt = args.regressor_checkpoint
                    with torch.no_grad():
                        out = generate_music(model, batch, hparams)
                    gen = out["generation_tokens"][0]
                    achieved = compute_density(gen, tokenizer_type)
                    music = (score_sample(model, gen, device, bigram_table)
                             if bigram_table is not None else {})
                    name = f"p{pi}_lam{lam}_tgt{tgt}".replace(".", "_")
                    wav = render_wav(prompt + gen, tokenizer, out_dir, name)
                    print(f"[prompt {pi}] lambda={lam}  target={tgt:.3f}  "
                          f"achieved={achieved:.4f}  music={music}")
                    table_rows.append([
                        pi, f"λ={lam}", lam, tgt, achieved,
                        music.get("ebt_energy"), music.get("bigram_ll"),
                        music.get("repetition_ratio"),
                        wandb.Audio(wav, caption=f"p{pi} λ={lam} target={tgt:.2f} "
                                                  f"achieved={achieved:.3f}"),
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
