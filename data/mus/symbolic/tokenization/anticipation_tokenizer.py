import sys
import os
import shutil
from datetime import datetime
import wandb
from argparse import Namespace
from pathlib import Path
from miditok import REMI, TokenizerConfig 
from path_utils import get_dataset_path, get_subset_path
from dataloaders.constants import DatasetType

# Import the main logic from your sub-modules
from tokenization.anticipation.train.midi_preprocess import main as preprocess_main
from tokenization.anticipation.finetune.tokenize_custom import main as custom_main
from tokenization.anticipation.train.tokenize_gigaMIDI import main as giga_main

class AnticipationTokenizer:
    def __init__(self, dataset_name: str, dataset_type: DatasetType, augment: int = 1, interarrival: bool = False, use_wandb = True):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.augment = augment
        self.interarrival = interarrival
        self.use_wandb = use_wandb
        
        if self.use_wandb:
            # Dynamically naming the project based on the dataset (e.g., GigaMIDI-Tokenization)
            # We capitalize it for a cleaner look on the dashboard
            project_formatted = f"{self.dataset_name.replace('-', ' ').title().replace(' ', '-')}-Tokenization"
            
            wandb.init(
                project=project_formatted,
                name=f"anticipation-{datetime.now().strftime('%m%d-%H%M')}",
                settings=wandb.Settings(console="wrap_raw"), 
                config={
                    "dataset": dataset_name,
                    "pipeline": "Anticipation",
                    "augment": self.augment,
                    "interarrival": self.interarrival
                }
            )
        
        # 1. Resolve MIDI source directory
        self.midi_dir = get_subset_path(self.dataset_name, "midi")
        
        # 2. Define Token destination and Quarantine
        self.token_dir = os.path.join(get_dataset_path(self.dataset_name), "tokens", "anticipation")
        self.quarantine_dir = os.path.join(get_dataset_path(self.dataset_name), "quarantined_midis")
        
        os.makedirs(self.token_dir, exist_ok=True)
        os.makedirs(self.quarantine_dir, exist_ok=True)
        
        # Initialize a basic tokenizer just for the "Mirror Validation" check
        self.validator = REMI(TokenizerConfig())

    def _pre_cleanup_and_validate(self):
        """
        1. Physically deletes Mac metadata (._*) and __MACOSX.
        2. Dry-runs every MIDI to catch corruptions BEFORE the main script crashes.
        """
        print(f"🧹 [Pre-Flight] Cleaning and Validating: {self.midi_dir}")
        deleted_junk = 0
        quarantined_files = 0
        
        # Get all potential files
        all_paths = list(Path(self.midi_dir).rglob("*.mid")) + list(Path(self.midi_dir).rglob("*.midi"))
        
        for midi_path in all_paths:
            # A. Remove Mac System Junk Immediately
            if midi_path.name.startswith("._") or midi_path.name == ".DS_Store":
                try:
                    os.remove(midi_path)
                    deleted_junk += 1
                except: pass
                continue

            # B. Mirror Check: Try to encode. If it fails, it's 'corrupted' for our purposes.
            try:
                # We don't need the tokens, just the 'True' or 'False' on whether it opens
                _ = self.validator.encode(str(midi_path))
            except Exception:
                quarantined_files += 1
                # Move to quarantine while preserving structure
                try:
                    rel_path = midi_path.relative_to(self.midi_dir)
                    dest_path = Path(self.quarantine_dir) / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(midi_path), str(dest_path))
                    print(f"🚨 Quarantined corrupted file: {midi_path.name}")
                except: pass

        print(f"✅ Cleanup results: {deleted_junk} junk files deleted, {quarantined_files} actual corruptions quarantined.")
        return quarantined_files

    def preprocess(self, add_drum: bool = False):
        """Cleans and standardizes MIDI files, with automatic skip detection."""
        # Always clean and validate first
        new_quarantines = self._pre_cleanup_and_validate()

        sentinel_file = os.path.join(self.midi_dir, ".preprocessed_done")

        # If we already did this AND no new files were just quarantined, we can skip
        if os.path.exists(sentinel_file) and new_quarantines == 0:
            print(f"✨ Preprocessing already completed. Skipping...")
            return

        print(f"\n--- [1/2] Preprocessing MIDI files in: {self.midi_dir} ---")
        preproc_args = Namespace(dir=self.midi_dir, add_drum=add_drum)
        
        preprocess_main(preproc_args)

        with open(sentinel_file, 'w') as f:
            f.write(f"Completed on: {datetime.now()}")
        print(f"✅ Preprocessing marker created: {sentinel_file}")

    def tokenize(self):
        """Converts preprocessed MIDIs into text-based event tokens."""
        print(f"\n--- [2/2] Tokenizing results into: {self.token_dir} ---")
        
        # SAFETY: If the train file is tiny (under 100MB), it's corrupted from the last fail.
        # We delete it so it starts fresh without the Mac ghosts.
        train_file = os.path.join(self.token_dir, "tokenized-events-train.txt")
        if os.path.exists(train_file):
            size_mb = os.path.getsize(train_file) / (1024 * 1024)
            if size_mb < 100:
                print(f"🗑️ Deleting suspiciously small train file ({size_mb:.2f}MB) to restart fresh.")
                os.remove(train_file)
        
        token_args = Namespace(
            datadir=self.midi_dir,     
            outdir=self.token_dir,     
            dataset_name=self.dataset_name,
            augment=self.augment,
            interarrival=self.interarrival,
        )

        if self.dataset_type == DatasetType.GIGA_MIDI:
            print(f"Using GigaMIDI-specific tokenization...")
            giga_main(token_args)
        elif self.dataset_type == DatasetType.CUSTOM:
            print(f"Using simple folder-based logic...")
            custom_main(token_args)
        
        if self.use_wandb:
            wandb.finish()

    def run_full_pipeline(self, add_drum: bool = False):
        print(f"=== Starting Anticipation Pipeline for {self.dataset_name.upper()} ===")
        self.preprocess(add_drum=add_drum)
        self.tokenize()
        print(f"=== Finished. Data is in: {self.token_dir} ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 -m tokenization.anticipation_tokenizer [{', '.join(DatasetType.list())}]")
        sys.exit(1)

    try:
        mode = DatasetType(sys.argv[1].lower())
    except ValueError:
        print(f"❌ Error: '{sys.argv[1]}' is not a valid dataset type.")
        sys.exit(1)
    
    # Defaults for GigaMIDI
    if mode == DatasetType.GIGA_MIDI:
        dataset_name = "giga-midi"
        augment = 1
        interarrival = True
        add_drum = False
    elif mode == DatasetType.CUSTOM:
        dataset_name = sys.argv[2] if len(sys.argv) > 2 else "custom"
        augment = 1       
        interarrival = True
        add_drum = True
    else:
        print(f"⚠️ Mode {mode} logic is not defined.")
        sys.exit(1)

    master = AnticipationTokenizer(
        dataset_name=dataset_name, 
        dataset_type=mode,
        augment=augment, 
        interarrival=interarrival
    )
    
    master.run_full_pipeline(add_drum=add_drum)