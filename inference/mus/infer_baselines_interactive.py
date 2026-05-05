#!/usr/bin/env python3
"""
Interactive inference script for symbolic music baselines.

Evaluates trained baseline models (HF GPT2, Llama) by:
1. Loading a checkpoint
2. Sampling from test set
3. Generating continuations
4. Synthesizing to audio for listening

Usage:
    python inference/mus/infer_baselines_interactive.py \
        --checkpoint <path_to_ckpt> \
        --model_name baseline_hf_gpt2_transformer \
        --num_samples 3 \
        --output_dir ./inference_outputs
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from argparse import Namespace

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference.mus.generate_music import generate_remi
from inference.mus.tokens_to_midi import tokens_to_midi
from data.mus.symbolic.dataloaders.giga_midi_miditok_dataset import GigaMIDIMiditokDataset
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
from base_model_trainer import ModelTrainer


def load_checkpoint(ckpt_path: str, model_name: str, device: str) -> Tuple[object, Dict]:
    """
    Load checkpoint and return model + hparams.

    Args:
        ckpt_path: Path to Lightning checkpoint
        model_name: Name of model architecture
        device: Device to load model on

    Returns:
        (model, hparams) tuple
    """
    print(f"Loading checkpoint: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract hparams
    hparams_dict = checkpoint.get('hyper_parameters', {})
    if isinstance(hparams_dict, dict):
        hparams = Namespace(**hparams_dict)
    else:
        hparams = hparams_dict

    # Override model_name if specified
    hparams.model_name = model_name
    hparams.device = device

    # Create model trainer and load state dict
    model_trainer = ModelTrainer(hparams)
    model_trainer.load_state_dict(checkpoint['state_dict'])

    # Set to eval mode and move to device
    model_trainer = model_trainer.to(device)
    model_trainer.eval()
    model_trainer.model.eval()

    print(f"✅ Checkpoint loaded successfully")
    print(f"   Model: {hparams.model_name}")
    print(f"   Model size: {hparams.model_size}")
    print(f"   Tokenizer: {hparams.tokenizer_type}")

    return model_trainer.model, hparams


def prepare_dataset(hparams: Namespace, split: str = "validation") -> GigaMIDIMiditokDataset:
    """
    Load music dataset.

    Args:
        hparams: Hyperparameters with context_length
        split: "train", "validation", or "test"

    Returns:
        Dataset object
    """
    print(f"Loading {split} dataset...")
    dataset = GigaMIDIMiditokDataset(hparams, split=split)
    print(f"✅ Dataset loaded: {len(dataset)} songs")
    return dataset


def sample_and_prepare_prompt(
    dataset: GigaMIDIMiditokDataset,
    sample_idx: int,
    prompt_length: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Sample from dataset and extract prompt.

    Args:
        dataset: Music dataset
        sample_idx: Index in dataset
        prompt_length: Number of tokens for prompt
        device: Device to place tensors on

    Returns:
        (prompt_tokens, full_tokens, actual_prompt_len) tuple
    """
    sample = dataset[sample_idx]
    full_tokens = sample['input_ids']  # (context_length,)

    # Extract prompt (first prompt_length tokens)
    prompt_tokens = full_tokens[:prompt_length].unsqueeze(0).to(device)  # (1, prompt_length)

    return prompt_tokens, full_tokens, prompt_length


def generate_continuation(
    model: object,
    hparams: Namespace,
    prompt_tokens: torch.Tensor,
    generation_length: int
) -> List[int]:
    """
    Generate token continuation from prompt.

    Args:
        model: Trained model
        hparams: Hyperparameters with inference settings
        prompt_tokens: Prompt token tensor (1, prompt_len)
        generation_length: Number of tokens to generate

    Returns:
        List of generated token IDs
    """
    # Prepare batch
    batch = {'input_ids': prompt_tokens}

    # Set generation hyperparams
    hparams.infer_max_gen_len = generation_length
    hparams.infer_temp = 0.7
    hparams.infer_topp = 0.9
    hparams.infer_logprobs = False
    hparams.infer_echo = False

    # Generate
    print(f"Generating {generation_length} tokens...")
    with torch.no_grad():
        outputs = generate_remi(model, batch, hparams)

    generated_tokens = outputs['generation_tokens'][0]  # First (and only) sample
    print(f"✅ Generated {len(generated_tokens)} tokens")

    return generated_tokens


def tokens_to_files(
    tokens: List[int],
    tokenizer: object,
    midi_path: str,
    wav_path: str,
    description: str = "generated"
) -> bool:
    """
    Convert tokens to MIDI and optionally to WAV.

    Args:
        tokens: List of token IDs
        tokenizer: Tokenizer object with decode method
        midi_path: Where to save MIDI file
        wav_path: Where to save WAV file
        description: Description for logging

    Returns:
        True if successful, False otherwise
    """
    try:
        # Tokens to MIDI (returns bytes)
        midi_bytes = tokens_to_midi(tokens, tokenizer)
        with open(midi_path, 'wb') as f:
            f.write(midi_bytes)
        print(f"✅ Saved {description} MIDI: {midi_path}")

        # Try to convert MIDI to WAV using midi2audio
        try:
            from midi2audio import FluidSynth
            fs = FluidSynth()
            fs.midi_to_audio(midi_path, wav_path)
            print(f"✅ Synthesized WAV: {wav_path}")
        except ImportError:
            print(f"⚠️  midi2audio not available, skipping WAV synthesis")
            print(f"   (install via: pip install midi2audio)")
        except Exception as e:
            print(f"⚠️  WAV synthesis failed: {e}")
            print(f"   Saved MIDI file only")

        return True

    except Exception as e:
        print(f"❌ Failed to convert tokens to MIDI: {e}")
        return False


def run_single_sample_inference(
    model: object,
    hparams: Namespace,
    tokenizer: object,
    dataset: GigaMIDIMiditokDataset,
    sample_idx: int,
    prompt_length: int,
    generation_length: int,
    output_dir: str,
    device: str
) -> bool:
    """
    Run inference on a single sample.

    Args:
        model: Trained model
        hparams: Hyperparameters
        tokenizer: Tokenizer object
        dataset: Music dataset
        sample_idx: Index in dataset
        prompt_length: Tokens for prompt
        generation_length: Tokens to generate
        output_dir: Output directory
        device: Device for computation

    Returns:
        True if successful
    """
    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}")
    print(f"{'='*70}")

    try:
        # Prepare prompt
        prompt_tokens, full_tokens, _ = sample_and_prepare_prompt(
            dataset, sample_idx, prompt_length, device
        )

        # Generate continuation
        generated_tokens = generate_continuation(
            model, hparams, prompt_tokens, generation_length
        )

        # Prepare full sequence with continuation
        full_with_continuation = list(full_tokens.cpu().numpy())[:prompt_length] + generated_tokens

        # Save outputs
        base_name = f"sample_{sample_idx}"

        # Ground truth (full original)
        print("\nSaving ground truth...")
        tokens_to_files(
            list(full_tokens.cpu().numpy()),
            tokenizer,
            f"{output_dir}/{base_name}_ground_truth.mid",
            f"{output_dir}/{base_name}_ground_truth.wav",
            "ground truth"
        )

        # Prompt only
        print("Saving prompt...")
        tokens_to_files(
            list(prompt_tokens[0].cpu().numpy()),
            tokenizer,
            f"{output_dir}/{base_name}_prompt.mid",
            f"{output_dir}/{base_name}_prompt.wav",
            "prompt"
        )

        # Generated continuation only
        print("Saving generated continuation...")
        tokens_to_files(
            generated_tokens,
            tokenizer,
            f"{output_dir}/{base_name}_generated.mid",
            f"{output_dir}/{base_name}_generated.wav",
            "generated"
        )

        # Save generated tokens to JSON
        tokens_dir = Path(output_dir).parent / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        tokens_file = tokens_dir / f"{base_name}_generated.json"
        with open(tokens_file, 'w') as f:
            import json
            json.dump(generated_tokens, f)

        # Full sequence with continuation
        print("Saving full sequence with continuation...")
        tokens_to_files(
            full_with_continuation,
            tokenizer,
            f"{output_dir}/{base_name}_full_with_continuation.mid",
            f"{output_dir}/{base_name}_full_with_continuation.wav",
            "full with continuation"
        )

        return True

    except Exception as e:
        print(f"❌ Error processing sample {sample_idx}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Interactive inference for symbolic music baselines"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="baseline_hf_gpt2_transformer",
        choices=["baseline_hf_gpt2_transformer", "baseline_llama_transformer"],
        help="Model architecture"
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=128,
        help="Number of tokens for prompt context"
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=256,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of test samples to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_outputs",
        help="Output directory for MIDI and WAV files"
    )
    parser.add_argument(
        "--use_test_split",
        action="store_true",
        help="Use test split instead of validation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for computation"
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load checkpoint
    model, hparams = load_checkpoint(args.checkpoint, args.model_name, device)

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer, vocab_size, pad_token_id = load_tokenizer(
        tokenizer_type=hparams.tokenizer_type,
        tokenizer_config_path=getattr(hparams, 'tokenizer_config_path', None),
        dataset_name=getattr(hparams, 'dataset_name', 'giga_midi')
    )
    print(f"✅ Tokenizer loaded: {hparams.tokenizer_type} ({vocab_size} vocab)")

    # Load dataset
    split = "test" if args.use_test_split else "validation"
    dataset = prepare_dataset(hparams, split=split)

    # Run inference on samples
    successful = 0
    failed = 0

    for i in range(min(args.num_samples, len(dataset))):
        if run_single_sample_inference(
            model, hparams, tokenizer, dataset,
            i, args.prompt_length, args.generation_length,
            str(output_dir), device
        ):
            successful += 1
        else:
            failed += 1

    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"✅ Successful: {successful}/{args.num_samples}")
    print(f"❌ Failed: {failed}/{args.num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"\nTo listen to generated audio:")
    print(f"  ffplay {output_dir}/sample_0_generated.wav")

    # Save metadata
    metadata = {
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "prompt_length": args.prompt_length,
        "generation_length": args.generation_length,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_samples_requested": args.num_samples,
        "num_samples_successful": successful,
        "dataset_split": split,
        "device": device,
    }

    metadata_path = output_dir / "inference_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
