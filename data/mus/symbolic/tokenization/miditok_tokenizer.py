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

    def _pre_cleanup(self):
        """
        Deletes hidden macOS metadata files (._*) and __MACOSX folders 
        to prevent them from being treated as valid MIDI files.
        """
        print(f"🧹 Starting pre-tokenization cleanup in: {self.root_datadir}")
        deleted_files = 0
        deleted_dirs = 0

        # walk topdown=False to delete files inside folders before the folders themselves
        for root, dirs, files in os.walk(self.root_datadir, topdown=False):
            # 1. Remove ._ metadata files and .DS_Store
            for name in files:
                if name.startswith("._") or name == ".DS_Store":
                    file_path = os.path.join(root, name)
                    try:
                        os.remove(file_path)
                        deleted_files += 1
                    except Exception as e:
                        print(f"   ❌ Failed to delete {name}: {e}")

            # 2. Remove __MACOSX internal folders, in case there are any
            for name in dirs:
                if name == "__MACOSX":
                    dir_path = os.path.join(root, name)
                    try:
                        shutil.rmtree(dir_path)
                        deleted_dirs += 1
                    except Exception as e:
                        print(f"   ❌ Failed to delete directory {name}: {e}")

        if deleted_files > 0 or deleted_dirs > 0:
            print(f"✅ Cleanup complete: Removed {deleted_files} junk files and {deleted_dirs} junk directories.")
        else:
            print("✨ Dataset was already clean. No junk files found.")

    def _get_midi_files(self, search_path):
        p = Path(search_path)
        all_files = list(p.rglob("*.mid")) + list(p.rglob("*.midi"))
        
        # Secondary safety filter just in case
        valid_files = [f for f in all_files if not f.name.startswith("._")]
        
        return valid_files

    def _safe_tokenize_loop(self, input_dir, output_dir):
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
            try:
                rel_path = midi_path.relative_to(input_dir)
            except ValueError:
                rel_path = Path(midi_path.name)

            token_save_path = Path(output_dir) / rel_path.with_suffix(".json")

            if token_save_path.exists():
                count_skipped += 1
                continue

            token_save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                tokens = self.tokenizer.encode(str(midi_path))
                self.tokenizer.save_tokens(tokens, str(token_save_path))
                count_success += 1
                if count_success % 1000 == 0:
                    print(f"  Processed {count_success} files...")

            except (RuntimeError, Exception) as e:
                count_quarantine += 1
                dest_path = Path(self.quarantine_dir) / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    shutil.move(str(midi_path), str(dest_path))
                except Exception:
                    pass

        print(f"📊 Results: {count_success} new, {count_skipped} skipped, {count_quarantine} quarantined.")

    def _cleanup(self):
        """Removes the quarantine directory if it is empty."""
        if os.path.exists(self.quarantine_dir):
            if not any(Path(self.quarantine_dir).iterdir()):
                print(f"🧹 Cleaning up: Quarantine directory is empty, removing it.")
                os.rmdir(self.quarantine_dir)
            else:
                print(f"⚠️ Note: {self.quarantine_dir} contains files. Check them for actual corruption.")

    def tokenize(self):
        # RUN CLEANUP FIRST
        self._pre_cleanup()

        print(f"\n=== [MidiTok] Processing {self.dataset_name} ===")
        os.makedirs(self.token_base_dir, exist_ok=True)

        splits = ['train', 'validation', 'test']
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
            self._safe_tokenize_loop(search_path, self.token_base_dir)

        tokenizer_json_path = os.path.join(self.token_base_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_json_path)
        
        # POST CLEANUP
        self._cleanup()

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
    if mode == DatasetType.GIGA_MIDI:
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