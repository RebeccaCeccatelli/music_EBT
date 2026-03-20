import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from tokenization.constants import DatasetType, get_dataset_path

class MidiTokTokenizer:
    def __init__(self, dataset_type: DatasetType, dataset_name: str, encoding="REMI"):
        """
        Args:
            dataset_type: Enum (DatasetType.CUSTOM or DatasetType.LAKH)
            dataset_name: The string name of the folder (e.g., 'jordan-progrock-dataset')
            encoding: MidiTok strategy (REMI, TSD, etc.)
        """
        self.dataset_type = DatasetType(dataset_type)
        self.dataset_name = dataset_name
        self.root_datadir = get_dataset_path(dataset_name)
        
        # Define the base output directory for MidiTok
        self.token_base_dir = os.path.join(self.root_datadir, "tokens", "miditok")
        
        # Initialize MidiTok Configuration
        config = TokenizerConfig(num_velocities=32, use_chords=True, use_programs=True)
        
        # Future-proofing: could switch strategy based on 'encoding' arg
        if encoding.upper() == "REMI":
            self.tokenizer = REMI(config)
        else:
            # Fallback to REMI or add other strategies here
            self.tokenizer = REMI(config)

    def _get_midi_files(self, search_path):
        """Helper to find all MIDI files in a specific path."""
        p = Path(search_path)
        return list(p.rglob("*.mid")) + list(p.rglob("*.midi"))

    def tokenize(self):
        print(f"\n=== [MidiTok] Processing {self.dataset_name} ({self.dataset_type.value}) ===")
        
        # Ensure the base tokens/miditok directory exists
        os.makedirs(self.token_base_dir, exist_ok=True)

        # Check for standard data splits
        splits = ['train', 'validation', 'test']
        has_splits = any(os.path.isdir(os.path.join(self.root_datadir, s)) for s in splits)

        if has_splits:
            # Loop through each split and tokenize into subfolders within tokens/miditok/
            for split in splits:
                split_path = os.path.join(self.root_datadir, split)
                if not os.path.isdir(split_path):
                    continue
                
                output_dir = os.path.join(self.token_base_dir, split)
                os.makedirs(output_dir, exist_ok=True)
                
                files = self._get_midi_files(split_path)
                
                if files:
                    print(f"Tokenizing {len(files)} files for split: {split}...")
                    self.tokenizer.tokenize_midi_dataset(files, output_dir)
                else:
                    print(f"No MIDI files found in {split_path}")
        else:
            # Process as a single custom dataset (flat structure)
            # Input usually comes from the 'midi' subfolder or root
            midi_input_dir = os.path.join(self.root_datadir, "midi")
            
            # If 'midi' folder doesn't exist, fallback to root_datadir
            search_path = midi_input_dir if os.path.isdir(midi_input_dir) else self.root_datadir
            
            files = self._get_midi_files(search_path)
            
            if files:
                print(f"Tokenizing {len(files)} files into {self.token_base_dir}...")
                self.tokenizer.tokenize_midi_dataset(files, self.token_base_dir)
            else:
                print(f"Error: No MIDI files found in {search_path}")

        # Save the tokenizer JSON inside the miditok folder
        tokenizer_json_path = os.path.join(self.token_base_dir, "tokenizer.json")
        self.tokenizer.save_params(tokenizer_json_path)
        print(f"Done! Tokenizer config saved to: {tokenizer_json_path}")

    def run_full_pipeline(self):
        """Entry point mirroring the AnticipationTokenizer interface."""
        self.tokenize()

if __name__ == "__main__":
    # Example: Jordan Prog-Rock (Custom)
    # This will output to: path/to/jordan-progrock-dataset/tokens/miditok/
    jordan_tokenizer = MidiTokTokenizer(
        dataset_type=DatasetType.CUSTOM,
        dataset_name="jordan-progrock-dataset"
    )
    jordan_tokenizer.run_full_pipeline()