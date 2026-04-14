"""
Shared tokenizer loading utilities for symbolic and neural music models.

Supports:
- miditok: REMI, Octuple, CPWord, MuMIDI
- Anticipation: Vanilla, Interarrival, Arrival-Time variants
"""

from typing import Tuple, Any


def load_tokenizer(
    tokenizer_type: str,
    tokenizer_config_path: str = None,
    dataset_name: str = 'giga-midi',
    use_vanilla: bool = False
) -> Tuple[Any, int, int]:
    """
    Load a tokenizer based on type and return metadata.
    
    Args:
        tokenizer_type: One of:
            Miditok: "REMI", "Octuple", "CPWord", "MuMIDI"
            Anticipation: "Anticipation-Vanilla", "Anticipation-Interarrival", "Anticipation-Arrival-Time", or "Anticipation"
        tokenizer_config_path: Optional path to tokenizer config file (for miditok)
        dataset_name: Dataset name for Anticipation tokenizer (default: 'giga-midi')
        use_vanilla: If True and tokenizer_type is "Anticipation", use vanilla vocabulary (no control block).
                    Ignored if tokenizer_type specifies a variant like "Anticipation-Vanilla"
    
    Returns:
        Tuple of (tokenizer_object, vocab_size, pad_token_id)
    
    Raises:
        ValueError: If tokenizer_type is unknown or invalid
    """
    
    if tokenizer_type.startswith('Anticipation'):
        return _load_anticipation_tokenizer(tokenizer_type, dataset_name, use_vanilla)
    else:
        return _load_miditok_tokenizer(tokenizer_type, tokenizer_config_path)


def _load_anticipation_tokenizer(
    tokenizer_type: str,
    dataset_name: str,
    use_vanilla: bool = False
) -> Tuple[Any, int, int]:
    """
    Load anticipation tokenizer with specified variant.
    
    Variants:
    - Vanilla: No control block (simplified, smaller vocab)
    - Interarrival: MIDI-like interarrival time encoding
    - Arrival-Time: Absolute time encoding (supports infilling)
    
    Args:
        tokenizer_type: "Anticipation-Vanilla", "Anticipation-Interarrival", "Anticipation-Arrival-Time", or "Anticipation"
        dataset_name: Dataset name for the tokenizer
        use_vanilla: If True and type is just "Anticipation", use vanilla variant
    """
    
    # Determine variant from tokenizer_type
    if tokenizer_type == 'Anticipation':
        # Use the use_vanilla flag to determine variant
        variant = 'Vanilla' if use_vanilla else 'Arrival-Time'
    else:
        # Extract variant from full type like "Anticipation-Vanilla"
        variant = tokenizer_type.split('-', 1)[1]
    
    # Parse variant
    if variant == 'Vanilla':
        interarrival = False
        use_vanilla = True
    elif variant == 'Interarrival':
        interarrival = True
        use_vanilla = False
    elif variant == 'Arrival-Time':
        interarrival = False
        use_vanilla = False
    else:
        raise ValueError(
            f"Unknown Anticipation variant: {variant}. "
            f"Use: Vanilla, Interarrival, or Arrival-Time"
        )
    
    try:
        from data.mus.symbolic.tokenization.anticipation_tokenizer import AnticipationTokenizer
        from dataloaders.constants import DatasetType
    except ImportError as e:
        raise ImportError(
            f"Failed to import Anticipation tokenizer. "
            f"Ensure you're in the music-EBT workspace. Error: {e}"
        )
    
    # Instantiate tokenizer
    tokenizer = AnticipationTokenizer(
        dataset_name=dataset_name,
        dataset_type=DatasetType.GIGA_MIDI,
        interarrival=interarrival,
        use_vanilla=use_vanilla,
        use_wandb=False  # Don't log during training initialization
    )
    
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.get_pad_token_id()
    
    return tokenizer, vocab_size, pad_token_id


def _load_miditok_tokenizer(
    tokenizer_type: str,
    tokenizer_config_path: str = None
) -> Tuple[Any, int, int]:
    """
    Load miditok tokenizer (REMI, Octuple, CPWord, MuMIDI).
    
    Args:
        tokenizer_type: One of "REMI", "Octuple", "CPWord", "MuMIDI"
        tokenizer_config_path: Optional path to config file
    
    Returns:
        Tuple of (tokenizer, vocab_size, pad_token_id)
    """
    
    try:
        from miditok import REMI, Octuple, CPWord, MuMIDI
    except ImportError:
        raise ImportError("miditok not installed. Install with: pip install miditok")
    
    # Map tokenizer names to classes
    tokenizer_classes = {
        'REMI': REMI,
        'Octuple': Octuple,
        'CPWord': CPWord,
        'MuMIDI': MuMIDI,
    }
    
    if tokenizer_type not in tokenizer_classes:
        available = ', '.join(tokenizer_classes.keys())
        raise ValueError(
            f"Unknown miditok tokenizer: {tokenizer_type}. "
            f"Available: {available}"
        )
    
    # Instantiate tokenizer
    tokenizer_class = tokenizer_classes[tokenizer_type]
    try:
        if tokenizer_config_path:
            tokenizer = tokenizer_class(params=tokenizer_config_path)
        else:
            tokenizer = tokenizer_class()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize {tokenizer_type} tokenizer. "
            f"Config path: {tokenizer_config_path}. Error: {e}"
        )
    
    vocab_size = len(tokenizer.vocab)
    pad_token_id = tokenizer.pad_token_id
    
    return tokenizer, vocab_size, pad_token_id


def get_supported_tokenizers() -> dict:
    """
    Return dict of supported tokenizer types and their descriptions.
    Useful for validation and help messages.
    """
    return {
        'REMI': 'miditok (default token representation)',
        'Octuple': 'miditok (parallel tracks)',
        'CPWord': 'miditok (compound words)',
        'MuMIDI': 'miditok (MuMIDI representation)',
        'Anticipation-Vanilla': 'Anticipation (simplified, no control block)',
        'Anticipation-Interarrival': 'Anticipation (MIDI-like interarrival)',
        'Anticipation-Arrival-Time': 'Anticipation (absolute time, supports infilling)',
    }


def verify_tokens(dataset_name: str = "giga-midi") -> dict:
    """
    Verify that tokenization is complete for a dataset.
    Checks for presence and size of tokenized-events files.
    
    Args:
        dataset_name: Dataset name (default: 'giga-midi')
    
    Returns:
        dict: Status for each split ('train', 'validation', 'test')
    """
    import os
    from pathlib import Path
    from data.mus.symbolic.path_utils import get_dataset_path
    
    root_dir = get_dataset_path(dataset_name)
    token_dir = os.path.join(root_dir, "tokens", "anticipation")
    
    print(f"🔍 Checking tokenization for: {dataset_name}")
    print(f"   Token directory: {token_dir}")
    print()
    
    splits = ['train', 'validation', 'test']
    results = {}
    all_ok = True
    total_size_mb = 0
    
    for split in splits:
        token_file = os.path.join(token_dir, f"tokenized-events-{split}.txt")
        
        if os.path.exists(token_file):
            size_bytes = os.path.getsize(token_file)
            size_mb = size_bytes / (1024 * 1024)
            total_size_mb += size_mb
            
            # Count lines (rough estimate of number of tokenized sequences)
            try:
                with open(token_file, 'r') as f:
                    num_lines = sum(1 for _ in f)
            except:
                num_lines = "?"
            
            if size_bytes > 0:
                results[split] = "✅"
                status = f"✅ READY   | {size_mb:>8.2f} MB | ~{num_lines} sequences"
            else:
                results[split] = "⚠️"
                all_ok = False
                status = f"⚠️  EMPTY   | {size_mb:>8.2f} MB | (0 bytes)"
        else:
            results[split] = "❌"
            all_ok = False
            status = f"❌ MISSING | File not found"
        
        print(f"  {split:12} {status}")
    
    print()
    print(f"📊 Total tokenized data: {total_size_mb:.2f} MB")
    
    # Check for miditok REMI tokenizer
    remi_tokenizer = os.path.join(token_dir, "..", "miditok", "tokenizer.json")
    if os.path.exists(remi_tokenizer):
        print(f"✅ REMI tokenizer found: {remi_tokenizer}")
    else:
        print(f"⚠️  REMI tokenizer not found (may not be needed)")
    
    print()
    if all_ok:
        print("✨ Tokenization is complete! Ready for training.")
    else:
        print("⚠️  Tokenization is incomplete. Run:")
        print(f"   cd data/mus/symbolic/tokenization")
        print(f"   python -m anticipation_tokenizer {dataset_name}")
    
    return results


def verify_midi_source(dataset_name: str = "giga-midi") -> bool:
    """
    Check if raw MIDI files are present before tokenization.
    
    Args:
        dataset_name: Dataset name (default: 'giga-midi')
    
    Returns:
        bool: True if all splits have MIDI files, False otherwise
    """
    import os
    from pathlib import Path
    from data.mus.symbolic.path_utils import get_dataset_path
    
    root_dir = get_dataset_path(dataset_name)
    midi_dir = os.path.join(root_dir, "midi")
    
    if not os.path.exists(midi_dir):
        print(f"❌ MIDI directory not found: {midi_dir}")
        print(f"   Download GigaMIDI first:")
        print(f"   python -m data.mus.symbolic.dataloaders.giga_midi_dataloader")
        return False
    
    splits = ['train', 'validation', 'test']
    print(f"\n📁 Raw MIDI status in: {midi_dir}")
    
    all_midi_found = True
    for split in splits:
        split_path = os.path.join(midi_dir, split)
        
        if os.path.isdir(split_path):
            # Count MIDI files recursively
            midi_files = list(Path(split_path).rglob("*.mid")) + list(Path(split_path).rglob("*.midi"))
            midi_files = [f for f in midi_files if not f.name.startswith("._")]  # Skip Mac junk
            
            if midi_files:
                print(f"  {split:12} ✅ {len(midi_files)} MIDI files found")
            else:
                print(f"  {split:12} ⚠️  No MIDI files found")
                all_midi_found = False
        else:
            print(f"  {split:12} ❌ Directory missing")
            all_midi_found = False
    
    return all_midi_found
