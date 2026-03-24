import sys
import os
import shutil
import logging
from pathlib import Path
from miditok import REMI, TokenizerConfig
from path_utils import get_dataset_path
from dataloaders.constants import DatasetType

class MidiTokTokenizer:
    def __init__(self, dataset_name: str, dataset_type: DatasetType, encoding="REMI"):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.root_datadir = get_dataset_path(dataset_name)
        self.token_base_dir = os.path.join(self.root_datadir, "tokens", "miditok")
        
        # Define quarantine directory for corrupted files
        self.quarantine_dir = os.path.join(self.root_datadir, "quarantined_midis")
        
        config = TokenizerConfig(num_velocities=32, use_chords=True, use_programs=True)
        self.tokenizer = REMI(config)

    def _get_midi_files(self, search_path):
        p = Path(search_path)
        all_files = list(p.rglob("*.mid")) + list(p.rglob("*.midi"))
        
        # Filter out hidden macOS metadata files
        valid_files = [f for f in all_files if not f.name.startswith("._")]
        
        return valid_files

    def _safe_tokenize_loop(self, input_dir, output_dir):
        """
        Processes files individually to handle corruption without crashing 
        the whole script and allows for resuming partial runs.
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.quarantine_dir, exist_ok=True)
        
        files = self._get_midi_files(input_dir)
        if not files:
            print(f"⚠️ No MIDI files found in {input_dir}")
            return

        print(f"🚀 Processing {len(files)} files into {output_dir}...")
        
        count_success = 0
        count_skipped = 0
        count_quarantine = 0

        for midi_path in files:
            # Determine relative path to maintain folder structure in tokens/
            try:
                rel_path = midi_path.relative_to(input_dir)
            except ValueError:
                # Fallback if input_dir is not a parent
                rel_path = Path(midi_path.name)

            token_save_path = Path(output_dir) / rel_path.with_suffix(".json")

            # 1. Skip if already processed
            if token_save_path.exists():
                count_skipped += 1
                continue

            token_save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # 2. Tokenize and Save
                # .encode() is the modern replacement for .midi_to_tokens()
                tokens = self.tokenizer.encode(str(midi_path))
                self.tokenizer.save_tokens(tokens, str(token_save_path))
                count_success += 1
                
                # Print progress occasionally
                if count_success % 500 == 0:
                    print(f"  Processed {count_success} files...")

            except (RuntimeError, Exception) as e:
                # 3. Quarantine corrupted files
                count_quarantine += 1
                print(f"\n🚨 Corruption detected at index {count_success + count_skipped + count_quarantine}: {midi_path.name}")
                
                dest_path = Path(self.quarantine_dir) / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    shutil.move(str(midi_path), str(dest_path))
                    print(f"   📦 Moved to: {dest_path}")
                except Exception as move_err:
                    print(f"   ❌ Failed to move file: {move_err}")

        print(f"📊 Results for this batch: {count_success} new, {count_skipped} skipped, {count_quarantine} quarantined.")

    def tokenize(self):
        print(f"\n=== [MidiTok] Processing {self.dataset_name} ===")
        os.makedirs(self.token_base_dir, exist_ok=True)

        splits = ['train', 'validation', 'test']
        # Check if the dataset already has split folders
        has_splits = any(os.path.isdir(os.path.join(self.root_datadir, s)) for s in splits)

        if has_splits:
            for split in splits:
                split_path = os.path.join(self.root_datadir, split)
                if not os.path.isdir(split_path): continue
                
                output_dir = os.path.join(self.token_base_dir, split)
                print(f"\n📂 Split: {split}")
                self._safe_tokenize_loop(split_path, output_dir)
        else:
            midi_input_dir = os.path.join(self.root_datadir, "midi")
            search_path = midi_input_dir if os.path.isdir(midi_input_dir) else self.root_datadir
            
            print(f"\n📂 Flat dataset structure detected.")
            self._safe_tokenize_loop(search_path, self.token_base_dir)

        # Save tokenizer configuration
        tokenizer_json_path = os.path.join(self.token_base_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_json_path)
        print(f"\n✨ Done! Tokenizer config saved to: {tokenizer_json_path}")

    def run_full_pipeline(self):
        self.tokenize()

if __name__ == "__main__":
    # 1. Validation & Help Message
    if len(sys.argv) < 2:
        options = ", ".join(DatasetType.list())
        print(f"Usage: python3 -m tokenization.miditok_tokenizer [{options}] [dataset_name_if_custom]")
        sys.exit(1)

    # 2. Safe Enum Conversion
    try:
        mode = DatasetType(sys.argv[1].lower())
    except ValueError:
        print(f"❌ Error: '{sys.argv[1]}' is not a valid DatasetType.")
        print(f"Valid options are: {DatasetType.list()}")
        sys.exit(1)

    # 3. Resolve Dataset Name based on Enum
    if mode == DatasetType.LAKH:
        # Changed from "lakh-clean" to "lakh-full" to match your new download
        dataset_name = "lakh-full"
    elif mode == DatasetType.GIGA_MIDI:
        dataset_name = "giga-midi"
    elif mode == DatasetType.CUSTOM:
        if len(sys.argv) < 3:
            print("❌ Error: CUSTOM mode requires a dataset name.")
            sys.exit(1)
        dataset_name = sys.argv[2]
    else:
        dataset_name = mode.value 

    # 4. Initialize and run
    # Note: For LMD-Full, REMI is a great choice, but ensure your RAM is 
    # sufficient if you plan to aggregate these tokens later.
    master = MidiTokTokenizer(dataset_name=dataset_name, dataset_type=mode)
    master.run_full_pipeline()