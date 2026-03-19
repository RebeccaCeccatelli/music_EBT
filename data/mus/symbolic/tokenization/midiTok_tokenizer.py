import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from constants import DatasetType, get_dataset_path

class MidiTokTokenizer:
    def __init__(self, dataset_type: DatasetType, dataset_name: str, encoding="REMI"):
        """
        Args:
            dataset_type: Enum (DatasetType.CUSTOM or DatasetType.LAKH)
            dataset_name: The string name of the folder (e.g., 'jordan-progrock')
            encoding: MidiTok strategy (REMI, TSD, etc.)
        """
        self.dataset_type = DatasetType(dataset_type)
        self.dataset_name = dataset_name
        self.root_datadir = get_dataset_path(dataset_name)
        
        # Initialize MidiTok Configuration
        # (Using standard REMI settings; can be customized)
        config = TokenizerConfig(num_velocities=32, use_chords=True, use_programs=True)
        self.tokenizer = REMI(config)

    def _get_midi_files(self, search_path):
        """Helper to find all MIDI files in a specific path."""
        p = Path(search_path)
        return list(p.rglob("*.mid")) + list(p.rglob("*.midi"))

    def tokenize(self):
        print(f"\n=== [MidiTok] Processing {self.dataset_name} ({self.dataset_type.value}) ===")
        
        # Determine if we have subfolders (Lakh-style) or a flat folder (Custom)
        splits = ['train', 'validation', 'test']
        has_splits = any(os.path.isdir(os.path.join(self.root_datadir, s)) for s in splits)

        if has_splits:
            # Loop through each split and tokenize separately
            for split in splits:
                split_path = os.path.join(self.root_datadir, split)
                if not os.path.isdir(split_path):
                    continue
                
                output_dir = os.path.join(self.root_datadir, f"miditok_{split}")
                files = self._get_midi_files(split_path)
                
                print(f"Tokenizing {len(files)} files for split: {split}...")
                self.tokenizer.tokenize_midi_dataset(files, output_dir)
        else:
            # Process as a single custom dataset
            output_dir = os.path.join(self.root_datadir, "miditok_output")
            files = self._get_midi_files(self.root_datadir)
            
            print(f"Tokenizing {len(files)} files from {self.root_datadir}...")
            self.tokenizer.tokenize_midi_dataset(files, output_dir)

        # Save the tokenizer JSON so you can reconstruct the music later
        self.tokenizer.save_params(os.path.join(self.root_datadir, "tokenizer.json"))
        print(f"Done! Tokenizer config saved to {self.root_datadir}/tokenizer.json")

    def run_full_pipeline(self):
        """Entry point mirroring the AnticipationTokenizer interface."""
        self.tokenize()

if __name__ == "__main__":
    # --- Example 1: Jordan Prog-Rock (Custom) ---
    jordan_tokenizer = MidiTokTokenizer(
        dataset_type=DatasetType.CUSTOM,
        dataset_name="jordan-progrock-dataset"
    )
    jordan_tokenizer.run_full_pipeline()