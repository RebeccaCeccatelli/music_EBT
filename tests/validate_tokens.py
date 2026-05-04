#!/usr/bin/env python3
"""
Validate generated token files for correctness and consistency.
"""
import os
import sys
from collections import Counter
from pathlib import Path
from argparse import ArgumentParser

def get_vocab_sizes():
    """Get vocab sizes for different tokenizers"""
    from tokenization.anticipation.anticipation.vocab_selector import VOCAB_SIZE, MIDI_VOCAB_SIZE
    return {
        "anticipation_arrival": VOCAB_SIZE,
        "anticipation_interarrival": MIDI_VOCAB_SIZE,
        "miditok": 2000,  # Approximate; actual varies with config
    }

def validate_token_file(filepath, tokenizer_name=None):
    """Validate a single token file"""
    if not os.path.exists(filepath):
        return {"status": "missing", "error": f"File not found: {filepath}"}
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return {"status": "empty", "error": "File is empty"}
        
        # Parse tokens
        all_tokens = []
        token_counts = Counter()
        line_lengths = []
        sample_lines = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            tokens = line.split()
            line_lengths.append(len(tokens))
            all_tokens.extend(tokens)
            
            # Keep first 3 lines as samples
            if len(sample_lines) < 3:
                sample_lines.append(tokens[:20])  # First 20 tokens
            
            try:
                for token in tokens:
                    token_id = int(token)
                    token_counts[token_id] += 1
            except ValueError as e:
                return {
                    "status": "invalid",
                    "error": f"Non-integer token at line {line_num}: '{token}'"
                }
        
        # Statistics
        unique_tokens = len(token_counts)
        max_token = max(token_counts.keys()) if token_counts else 0
        min_token = min(token_counts.keys()) if token_counts else 0
        avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0
        
        result = {
            "status": "valid",
            "file_size_mb": os.path.getsize(filepath) / (1024 * 1024),
            "num_lines": len(lines),
            "num_tokens": len(all_tokens),
            "unique_tokens": unique_tokens,
            "token_range": f"{min_token}-{max_token}",
            "max_token_id": max_token,
            "min_token_id": min_token,
            "avg_tokens_per_line": avg_line_length,
            "top_5_tokens": token_counts.most_common(5),
            "sample_tokens": sample_lines,
        }
        
        # Validation checks specific to tokenizer type
        if tokenizer_name == "anticipation":
            # For anticipation, check expected vocab size ranges
            # Vanilla: ~5000, Non-vanilla: ~6000+
            if max_token < 4000:
                result["warning"] = "Token IDs lower than expected for anticipation tokenizer"
        elif tokenizer_name == "anticipation-vanilla":
            # Vanilla should have similar range to regular, but exact vocab differs
            if max_token < 4000:
                result["warning"] = "Token IDs lower than expected for anticipation-vanilla tokenizer"
        elif tokenizer_name == "miditok":
            # MidiTok typically uses smaller vocab (REMI)
            if max_token > 5000:
                result["warning"] = "Token IDs higher than typical for MidiTok (REMI) tokenizer"
        
        return result
    
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    parser = ArgumentParser(description="Validate tokenized MIDI files")
    parser.add_argument("dataset", help="Dataset name (e.g., jordan-progrock-dataset)")
    parser.add_argument("--storage", help="Remote storage path", default=None)
    args = parser.parse_args()
    
    # Determine storage path
    if args.storage:
        storage_path = args.storage
    elif os.getenv("REMOTE_DATA_STORAGE"):
        storage_path = os.getenv("REMOTE_DATA_STORAGE")
    else:
        storage_path = "/home/rebcecca/orcd/pool/music_datasets"
    
    dataset_path = os.path.join(storage_path, args.dataset)
    tokens_path = os.path.join(dataset_path, "tokens")
    
    if not os.path.exists(tokens_path):
        print(f"❌ Tokens directory not found: {tokens_path}")
        sys.exit(1)
    
    print(f"\n📊 TOKENIZATION VALIDATION REPORT\n")
    print(f"Dataset: {args.dataset}")
    print(f"Storage: {storage_path}\n")
    
    # Check each tokenizer
    tokenizers = {
        "Anticipation (Arrival-Time)": "anticipation",
        "Anticipation (Vanilla)": "anticipation-vanilla", 
        "MidiTok": "miditok",
    }
    
    all_results = {}
    
    for name, folder in tokenizers.items():
        folder_path = os.path.join(tokens_path, folder)
        if not os.path.exists(folder_path):
            print(f"⏭️  {name}: Folder not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"🎵 {name}")
        print(f"{'='*60}")
        
        # Special handling for MidiTok (uses .json files, not .txt)
        if folder == "miditok":
            print(f"\n📁 MidiTok uses individual .json files per MIDI:")
            json_files = list(Path(folder_path).rglob("*.json"))
            json_files = [f for f in json_files if f.name != "tokenizer.json"]  # Exclude tokenizer config
            
            if not json_files:
                print(f"  ❌ No tokenized .json files found")
            else:
                print(f"  ✅ Found {len(json_files)} tokenized files")
                total_size = sum(f.stat().st_size for f in json_files) / (1024 * 1024)
                print(f"  📦 Total Size: {total_size:.2f} MB")
                print(f"  📁 Distribution:")
                
                # Show files by split
                for split in ['train', 'validation', 'test']:
                    split_files = [f for f in json_files if f.parent.name == split or (split == 'test' and 'test' in f.name.lower())]
                    if split_files:
                        split_size = sum(f.stat().st_size for f in split_files) / (1024 * 1024)
                        print(f"     {split.capitalize()}: {len(split_files)} files ({split_size:.2f} MB)")
            
            folder_results["json"] = {"status": "valid", "count": len(json_files)}
            continue
        
        # Standard handling for Anticipation (uses aggregated .txt files)
        splits_to_try = [
            ["Train", "Test", "Validation"],      # Capitalized (GigaMIDI style)
            ["train", "test", "validation"],      # Lowercase (custom style)
        ]
        
        splits = None
        for option in splits_to_try:
            # Check if any of these splits exist
            if any(os.path.exists(os.path.join(folder_path, f"tokenized-events-{s}.txt")) for s in option):
                splits = option
                break
        
        if splits is None:
            splits = ["Train", "Test", "Validation"]  # fallback
        
        folder_results = {}
        
        for split in splits:
            filepath = os.path.join(folder_path, f"tokenized-events-{split}.txt")
            result = validate_token_file(filepath, tokenizer_name=folder)
            folder_results[split] = result
            
            status_icon = "✅" if result["status"] == "valid" else "❌"
            print(f"\n{status_icon} {split}:")
            
            if result["status"] == "valid":
                print(f"  📦 File Size: {result['file_size_mb']:.2f} MB")
                print(f"  📝 Lines: {result['num_lines']:,}")
                print(f"  🎯 Total Tokens: {result['num_tokens']:,}")
                print(f"  🔤 Unique Tokens: {result['unique_tokens']}")
                print(f"  📈 Token Range: {result['token_range']}")
                print(f"  ⏱️  Avg Tokens/Line: {result['avg_tokens_per_line']:.1f}")
                print(f"  🔝 Top 5 Tokens: {result['top_5_tokens']}")
                
                # Show sample tokens
                print(f"  📋 Sample Tokens (first 20 from 3 lines):")
                for i, sample in enumerate(result['sample_tokens'], 1):
                    print(f"     Line {i}: {' '.join(sample)}")
                
                if "warning" in result:
                    print(f"  ⚠️  {result['warning']}")
            else:
                print(f"  ⚠️  {result['error']}")
        
        all_results[name] = folder_results
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 SUMMARY")
    print(f"{'='*60}\n")
    
    valid_count = 0
    total_count = 0
    
    for tokenizer, splits in all_results.items():
        for split, result in splits.items():
            total_count += 1
            if result["status"] == "valid":
                valid_count += 1
    
    print(f"✅ Valid Files: {valid_count}/{total_count}")
    
    if valid_count == total_count:
        print(f"🎉 All tokenized files are valid and ready for training!\n")
        return 0
    else:
        print(f"⚠️  Some files have issues. Review above for details.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
