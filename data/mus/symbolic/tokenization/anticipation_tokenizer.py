import sys
import os
from argparse import Namespace
from pathlib import Path
from path_utils import get_dataset_path, get_subset_path
from dataloaders.constants import DatasetType

# Import the main logic from your sub-modules
from tokenization.anticipation.train.midi_preprocess import main as preprocess_main
from tokenization.anticipation.finetune.tokenize_custom import main as custom_main
from tokenization.anticipation.train.tokenize_gigaMIDI import main as giga_main

class AnticipationTokenizer:
    def __init__(self, dataset_name: str, dataset_type: DatasetType, augment: int = 1, interarrival: bool = False):
        self.dataset_name = dataset_name
        self.dataset_type = DatasetType(dataset_type)
        
        # 1. Resolve MIDI source directory (e.g., datasets/giga-midi/midi)
        self.midi_dir = get_subset_path(self.dataset_name, "midi")
        
        # 2. Define and CREATE the token destination (e.g., datasets/giga-midi/tokens/anticipation)
        # This prevents the FileNotFoundError during multiprocessing
        self.token_dir = os.path.join(get_dataset_path(self.dataset_name), "tokens", "anticipation")
        os.makedirs(self.token_dir, exist_ok=True)
        
        self.augment = augment
        self.interarrival = interarrival

    def preprocess(self, add_drum: bool = False):
        """Cleans and standardizes MIDI files, with automatic skip detection."""
        # Define the path for our "success marker"
        sentinel_file = os.path.join(self.midi_dir, ".preprocessed_done")

        if os.path.exists(sentinel_file):
            print(f"✨ Preprocessing already completed (marker found at {sentinel_file}). Skipping...")
            return

        print(f"\n--- [1/2] Preprocessing MIDI files in: {self.midi_dir} ---")
        preproc_args = Namespace(dir=self.midi_dir, add_drum=add_drum)
        
        # Run the actual preprocessing logic
        preprocess_main(preproc_args)

        # After successful completion, create the sentinel file
        with open(sentinel_file, 'w') as f:
            f.write(f"Completed on: {os.popen('date').read()}")
        print(f"✅ Preprocessing marker created: {sentinel_file}")

    def run_full_pipeline(self, add_drum: bool = False):
        """Convenience method to run both stages back-to-back."""
        print(f"=== Starting Full Pipeline for {self.dataset_name.upper()} ===")
        # Preprocess will now decide internally whether to run or skip
        self.preprocess(add_drum=add_drum)
        self.tokenize()
        print(f"=== Finished. Data is in: {self.token_dir} ===")

    def tokenize(self):
        """Converts preprocessed MIDIs into text-based event tokens."""
        print(f"\n--- [2/2] Tokenizing results into: {self.token_dir} ---")
        
        token_args = Namespace(
            datadir=self.midi_dir,     
            outdir=self.token_dir,     
            dataset_name=self.dataset_name,
            augment=self.augment,
            interarrival=self.interarrival,
        )

        # Different datasets assume different folder structures
        if self.dataset_type == DatasetType.GIGA_MIDI:
            print(f"Using GigaMIDI-specific tokenization for {self.dataset_type.value}")
            giga_main(token_args)
        elif self.dataset_type == DatasetType.CUSTOM:
            print(f"Using simple folder-based logic for {self.dataset_type.value}")
            custom_main(token_args)

if __name__ == "__main__":
    # 1. Basic validation: Check if a dataset type was provided (e.g. giga_midi)
    if len(sys.argv) < 2:
        print(f"Usage: python3 -m tokenization.anticipation_tokenizer [{', '.join(DatasetType.list())}] [optional_custom_name]")
        sys.exit(1)

    # 2. Convert input string to Enum safely
    try:
        # sys.argv[1].lower() matches the Enum values like "giga-midi" or "custom"
        mode = DatasetType(sys.argv[1].lower())
    except ValueError:
        print(f"❌ Error: '{sys.argv[1]}' is not a valid dataset type.")
        print(f"Valid options: {DatasetType.list()}")
        sys.exit(1)
    
    if mode == DatasetType.GIGA_MIDI:
        dataset_name = "giga-midi"
        augment = 1
        interarrival = True
        add_drum = False
        
    elif mode == DatasetType.CUSTOM:
        if len(sys.argv) < 3:
            print("❌ Error: CUSTOM mode requires a dataset name as the second argument.")
            sys.exit(1)
        dataset_name = sys.argv[2]
        augment = 1       
        interarrival = True
        add_drum = True
        
    else:
        print(f"⚠️ Mode {mode} logic is not defined for the Anticipation pipeline.")
        sys.exit(1)

    # 4. Initialize the Tokenizer with the correct paths and settings
    master = AnticipationTokenizer(
        dataset_name=dataset_name, 
        dataset_type=mode,
        augment=augment, 
        interarrival=interarrival
    )
    
    # 5. Execute the pipeline
    # This will: 
    #   a) Preprocess the MIDI files (cleaning/quantizing)
    #   b) Tokenize them into the 'anticipation' format (Control/Time/Note events)
    master.run_full_pipeline(add_drum=add_drum)