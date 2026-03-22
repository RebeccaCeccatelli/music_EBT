import sys
import os
from argparse import Namespace
from pathlib import Path
from path_utils import get_dataset_path, get_subset_path
from dataloaders.constants import DatasetType

# Import the main logic from your sub-modules
from tokenization.anticipation.train.midi_preprocess import main as preprocess_main
from tokenization.anticipation.finetune.tokenize_custom import main as custom_main
from tokenization.anticipation.train.tokenize_lakh import main as lakh_main

class AnticipationTokenizer:
    def __init__(self, dataset_name: str, dataset_type: DatasetType, augment: int = 1, interarrival: bool = False):
        self.dataset_name = dataset_name
        self.dataset_type = DatasetType(dataset_type)
        
        # 1. Resolve MIDI source directory (e.g., datasets/lakh-midi-clean/midi)
        self.midi_dir = get_subset_path(self.dataset_name, "midi")
        
        # 2. Define and CREATE the token destination (e.g., datasets/lakh-midi-clean/tokens/anticipation)
        # This prevents the FileNotFoundError during multiprocessing
        self.token_dir = os.path.join(get_dataset_path(self.dataset_name), "tokens", "anticipation")
        os.makedirs(self.token_dir, exist_ok=True)
        
        self.augment = augment
        self.interarrival = interarrival

    def preprocess(self, add_drum: bool = False):
        """Cleans and standardizes MIDI files before tokenization."""
        print(f"\n--- [1/2] Preprocessing MIDI files in: {self.midi_dir} ---")
        # Namespace mimics command line arguments for the underlying script
        preproc_args = Namespace(dir=self.midi_dir, add_drum=add_drum)
        preprocess_main(preproc_args)

    def tokenize(self):
        """Converts preprocessed MIDIs into text-based event tokens."""
        print(f"\n--- [2/2] Tokenizing results into: {self.token_dir} ---")
        
        token_args = Namespace(
            midi_dir=self.midi_dir,    # Source of preprocessed files
            token_dir=self.token_dir,  # Destination for .txt tokens
            dataset_name=self.dataset_name,
            augment=self.augment,
            interarrival=self.interarrival
        )

        # Lakh uses a specific hash-based splitting logic, Custom uses a simpler folder-based one
        if self.dataset_type in [DatasetType.LAKH, DatasetType.GIGA_MIDI]:
            print(f"Using hash-based splitting logic for {self.dataset_type.value}")
            lakh_main(token_args)
        elif self.dataset_type == DatasetType.CUSTOM:
            print(f"Using simple folder-based logic for {self.dataset_type.value}")
            custom_main(token_args)

    def run_full_pipeline(self, add_drum: bool = False):
        """Convenience method to run both stages back-to-back."""
        print(f"=== Starting Full Pipeline for {self.dataset_name.upper()} ===")
        self.preprocess(add_drum=add_drum)
        self.tokenize()
        print(f"=== Finished. Data is in: {self.token_dir} ===")

if __name__ == "__main__":
    # 1. Basic validation
    if len(sys.argv) < 2:
        print(f"Usage: python3 -m ... [{', '.join(DatasetType.list())}] [dataset_name_if_custom]")
        sys.exit(1)

    # 2. Convert input string to Enum safely
    try:
        # sys.argv[1].lower() matches the Enum values like "lakh" or "custom"
        mode = DatasetType(sys.argv[1].lower())
    except ValueError:
        print(f"❌ Error: '{sys.argv[1]}' is not a valid dataset type.")
        print(f"Valid options: {DatasetType.list()}")
        sys.exit(1)

    # 3. Set parameters based on the Enum
    if mode == DatasetType.LAKH:
        dataset_name = "lakh-midi-clean"
        augment = 1
    
    elif mode == DatasetType.GIGA_MIDI:
        dataset_name = "giga-midi"
        augment = 1
        
    elif mode == DatasetType.CUSTOM:
        if len(sys.argv) < 3:
            print("❌ Error: CUSTOM mode requires a dataset name.")
            sys.exit(1)
        dataset_name = sys.argv[2]
        augment = 10 
        
    else:
        # This handles MAESTRO or others that are not yet implemented
        print(f"⚠️ Mode {mode} logic is not yet implemented.")
        sys.exit(1)

    # 4. Pass the Enum object directly
    master = AnticipationTokenizer(
        dataset_name=dataset_name, 
        dataset_type=mode,
        augment=augment, 
        interarrival=False
    )
    
    master.run_full_pipeline(add_drum=True)