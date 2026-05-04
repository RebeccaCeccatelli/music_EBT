"""
Convert generated REMI tokens from inference to MIDI files.

Usage:
    python inference/mus/tokens_to_midi.py \
        --generations_file logs/inference/MUS_SYMB/giga-midi/baseline-symb-xxs-prod0.0008_2026-04-22_18-17-39_/last/generations.jsonl \
        --output_dir ./generated_midis \
        --tokenizer_type REMI
"""

import json
import argparse
import os
from pathlib import Path
from typing import List
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer


def tokens_to_midi(token_list: List[int], tokenizer) -> bytes:
    """
    Convert a list of token IDs to a MIDI file (bytes).
    
    Args:
        token_list: List of integer token IDs
        tokenizer: REMI tokenizer instance with decode() method
    
    Returns:
        MIDI file as bytes
    """
    # miditok tokenizer.decode() expects a list of token objects or token IDs
    # Convert token IDs to miditok tokens if needed
    midi_obj = tokenizer.decode(token_list)
    
    # Convert MIDI object to bytes
    if hasattr(midi_obj, 'save'):
        # If it's a pretty_midi or music21 object
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            midi_obj.write(f.name)
            with open(f.name, 'rb') as mf:
                midi_bytes = mf.read()
            os.remove(f.name)
            return midi_bytes
    else:
        # If it already returns bytes
        return midi_obj


def main():
    parser = argparse.ArgumentParser(description="Convert generated tokens to MIDI files")
    parser.add_argument(
        "--generations_file",
        type=str,
        required=True,
        help="Path to generations.jsonl file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_midis",
        help="Output directory for MIDI files"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="REMI",
        choices=["REMI", "Octuple", "CPWord", "MuMIDI"],
        help="Tokenizer type used in training"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="giga-midi",
        help="Dataset name for tokenizer"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Max number of MIDI files to generate (for testing)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Load tokenizer
    print(f"Loading {args.tokenizer_type} tokenizer...")
    tokenizer, vocab_size, pad_token_id = load_tokenizer(
        tokenizer_type=args.tokenizer_type,
        dataset_name=args.dataset_name
    )
    print(f"Tokenizer loaded: vocab_size={vocab_size}, pad_token_id={pad_token_id}")
    
    # Process generations file
    print(f"\nReading generations from: {args.generations_file}")
    
    if not os.path.exists(args.generations_file):
        print(f"ERROR: File not found: {args.generations_file}")
        return
    
    converted_count = 0
    error_count = 0
    
    with open(args.generations_file, 'r') as f:
        for i, line in enumerate(f):
            if args.max_files and converted_count >= args.max_files:
                break
                
            try:
                data = json.loads(line.strip())
                sample_idx = data.get('sample_idx', i)
                tokens = data.get('generated_tokens', [])
                
                if not tokens:
                    print(f"Sample {sample_idx}: No tokens found, skipping")
                    error_count += 1
                    continue
                
                # Convert tokens to MIDI
                print(f"Converting sample {sample_idx} ({len(tokens)} tokens)...", end=" ")
                midi_bytes = tokens_to_midi(tokens, tokenizer)
                
                # Save MIDI file
                output_file = output_path / f"sample_{sample_idx:04d}.mid"
                with open(output_file, 'wb') as mf:
                    mf.write(midi_bytes)
                
                print(f"✓ Saved to {output_file.name}")
                converted_count += 1
                
            except Exception as e:
                print(f"\nERROR processing line {i}: {e}")
                error_count += 1
                continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Successfully converted: {converted_count} samples")
    print(f"  Errors: {error_count} samples")
    print(f"  Output directory: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
