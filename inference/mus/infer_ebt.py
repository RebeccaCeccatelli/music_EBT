#!/usr/bin/env python3
"""
EBT Symbolic Music Inference Script

Loads a checkpoint, autoregressively generates continuations using the
energy-based forward pass, converts to MIDI/WAV, and logs to wandb.

Usage:
    python inference/mus/infer_ebt.py \
        --checkpoint <path_to_ckpt> \
        --num_samples 5 \
        --generation_length 256 \
        --output_dir ./ebt_outputs
"""

import torch
import argparse
import os
import json
from pathlib import Path
from argparse import Namespace
from typing import Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from base_model_trainer import ModelTrainer
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
from data.mus.symbolic.dataloaders.giga_midi_miditok_dataset import GigaMIDIMiditokDataset
from data.mus.symbolic.dataloaders.custom_music_dataloader import CustomMusicDataset
from inference.mus.generate_music import generate_music


_ANTICIPATION_DIR = {
    'Anticipation-Arrival-Time': 'anticipation',
    'Anticipation-Vanilla': 'anticipation-vanilla',
    'Anticipation-Interarrival': 'anticipation-interarrival',
}


def load_checkpoint(ckpt_path: str, device: str) -> Tuple[object, Namespace]:
    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams_dict = checkpoint.get('hyper_parameters', {})
    hparams = Namespace(**hparams_dict) if isinstance(hparams_dict, dict) else hparams_dict

    hparams.device = device
    model_trainer = ModelTrainer(hparams)
    model_trainer.load_state_dict(checkpoint['state_dict'])
    model_trainer = model_trainer.to(device)
    model_trainer.eval()
    model_trainer.model.eval()

    print(f"✅ Checkpoint loaded")
    print(f"   Model:     {hparams.model_name}")
    print(f"   Size:      {hparams.model_size}")
    print(f"   Tokenizer: {hparams.tokenizer_type}")
    return model_trainer.model, hparams


def load_dataset(hparams: Namespace, split: str = "test"):
    tokenizer_type = getattr(hparams, 'tokenizer_type', 'REMI')
    if tokenizer_type in _ANTICIPATION_DIR:
        return CustomMusicDataset(
            hparams,
            dataset_name='giga-midi',
            split=split,
            tokenizer_type=_ANTICIPATION_DIR[tokenizer_type],
        )
    return GigaMIDIMiditokDataset(hparams, split=split)


def main():
    parser = argparse.ArgumentParser(description="EBT Symbolic Music Inference")
    parser.add_argument("--checkpoint",        type=str,   required=True)
    parser.add_argument("--num_samples",       type=int,   default=5)
    parser.add_argument("--generation_length", type=int,   default=256, help="Tokens to generate after the prompt")
    parser.add_argument("--prompt_length",     type=int,   default=128, help="Prompt tokens")
    parser.add_argument("--temperature",       type=float, default=0.7)
    parser.add_argument("--top_p",             type=float, default=0.9)
    parser.add_argument("--mcmc_num_steps",    type=int,   default=None, help="Override MCMC refinement steps (default: use value from checkpoint)")
    parser.add_argument("--ebt_advanced",      action="store_true", help="Use multi-sample energy-based selection")
    parser.add_argument("--use_test_split",    action="store_true", help="Use test split instead of validation")
    parser.add_argument("--seed",              type=int,   default=None, help="Random seed for song selection (random each run if not set)")
    parser.add_argument("--output_dir",        type=str,   default="./ebt_inference")
    parser.add_argument("--device",            type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir     = Path(args.output_dir)
    midi_dir       = output_dir / "midi"
    wav_dir        = output_dir / "wav"
    tokens_dir     = output_dir / "tokens"
    piano_rolls_dir = output_dir / "piano_rolls"
    for d in (midi_dir, wav_dir, tokens_dir, piano_rolls_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model, hparams = load_checkpoint(args.checkpoint, args.device)

    # ── Inference hparams ─────────────────────────────────────────────────────
    if args.mcmc_num_steps is not None:
        print(f"Overriding mcmc_num_steps: {hparams.mcmc_num_steps} → {args.mcmc_num_steps}")
        hparams.mcmc_num_steps = args.mcmc_num_steps
    hparams.infer_ebt_advanced = args.ebt_advanced
    hparams.infer_max_gen_len  = args.generation_length
    hparams.infer_temp         = args.temperature
    hparams.infer_topp         = args.top_p
    hparams.infer_logprobs     = False
    hparams.infer_echo         = False

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer, vocab_size, pad_token_id = load_tokenizer(
        tokenizer_type=hparams.tokenizer_type,
        tokenizer_config_path=getattr(hparams, 'tokenizer_config_path', None),
        dataset_name=getattr(hparams, 'dataset_name', 'giga_midi'),
    )
    print(f"✅ Tokenizer: {hparams.tokenizer_type} (vocab_size={vocab_size})")

    # ── Load dataset ──────────────────────────────────────────────────────────
    split = "test" if args.use_test_split else "validation"
    dataset = load_dataset(hparams, split=split)
    print(f"✅ Dataset: {len(dataset)} songs ({split} split)")

    # ── Random song selection ─────────────────────────────────────────────────
    import random
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    num_samples = min(args.num_samples, len(dataset))
    sample_indices = rng.sample(range(len(dataset)), num_samples)
    print(f"Random seed: {seed}")
    print(f"Sample indices: {sample_indices}")

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\nGenerating {num_samples} samples "
          f"(temp={args.temperature}, top_p={args.top_p}, "
          f"ebt_advanced={args.ebt_advanced})...")

    all_generations = []

    for i, dataset_idx in enumerate(sample_indices):
        print(f"  Sample {i+1}/{num_samples} (dataset idx {dataset_idx})...", end="", flush=True)
        try:
            full_tokens = dataset[dataset_idx]['input_ids'].tolist()
            # Strip padding
            while full_tokens and full_tokens[-1] == pad_token_id:
                full_tokens.pop()

            prompt_len = min(args.prompt_length, len(full_tokens))
            prompt = full_tokens[:prompt_len]

            batch = {
                'input_ids': torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)
            }

            with torch.no_grad():
                result = generate_music(model, batch, hparams)

            generated = result['generation_tokens'][0]
            all_generations.append({
                'sample_idx':        i,
                'prompt_tokens':     prompt,
                'ground_truth_tokens': full_tokens,
                'generated_tokens':  prompt + generated,
            })
            print(f" ✅ ({len(generated)} new tokens)")

        except Exception as e:
            print(f" ❌ {e}")
            all_generations.append({
                'sample_idx':        i,
                'prompt_tokens':     [],
                'ground_truth_tokens': [],
                'generated_tokens':  [],
                'error':             str(e),
            })

    # ── Save generations JSONL ─────────────────────────────────────────────────
    jsonl_path = tokens_dir / "generations.jsonl"
    with open(jsonl_path, 'w') as f:
        for entry in all_generations:
            f.write(json.dumps(entry) + '\n')
    print(f"\n✅ Tokens saved: {jsonl_path}")

    # ── Convert to MIDI ───────────────────────────────────────────────────────
    print("\nConverting to MIDI...")
    from inference.mus.tokens_to_midi import tokens_to_midi

    for entry in all_generations:
        idx = entry['sample_idx']
        for kind, tokens in [
            ("prompt",       entry['prompt_tokens']),
            ("ground_truth", entry['ground_truth_tokens']),
            ("generated",    entry['generated_tokens']),
        ]:
            if not tokens:
                continue
            try:
                midi_bytes = tokens_to_midi(tokens, tokenizer)
                path = midi_dir / f"sample_{idx:03d}_{kind}.mid"
                path.write_bytes(midi_bytes)
                print(f"  ✅ {path.name}")
            except Exception as e:
                print(f"  ❌ sample_{idx:03d}_{kind}: {e}")

    # ── Synthesize WAV ────────────────────────────────────────────────────────
    print("\nSynthesizing WAV files...")
    from convert_midi_simple import simple_synth

    for midi_file in sorted(midi_dir.glob("*.mid")):
        wav_file = wav_dir / midi_file.with_suffix(".wav").name
        try:
            simple_synth(str(midi_file), str(wav_file))
            print(f"  ✅ {wav_file.name}")
        except Exception as e:
            print(f"  ❌ {midi_file.name}: {e}")

    # ── Generate piano rolls ──────────────────────────────────────────────────
    print("\nGenerating piano rolls...")
    from inference.mus.tokens_to_piano_roll import midi_to_piano_roll, save_piano_roll_image

    for midi_file in sorted(midi_dir.glob("*.mid")):
        png_path = piano_rolls_dir / midi_file.with_suffix(".png").name
        try:
            piano_roll = midi_to_piano_roll(midi_file)
            save_piano_roll_image(piano_roll, png_path, title=midi_file.stem)
            print(f"  ✅ {png_path.name}")
        except Exception as e:
            print(f"  ❌ {midi_file.name}: {e}")

    print(f"\n{'='*50}")
    print(f"✅ Inference complete")
    print(f"   MIDI:        {midi_dir}")
    print(f"   WAV:         {wav_dir}")
    print(f"   Piano rolls: {piano_rolls_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
