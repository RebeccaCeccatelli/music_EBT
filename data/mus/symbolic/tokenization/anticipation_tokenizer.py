import sys
import os
from argparse import Namespace
from pathlib import Path
from path_utils import get_dataset_path, get_subset_path
from tokenization.constants import DatasetType

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
        if self.dataset_type == DatasetType.LAKH:
            lakh_main(token_args)
        else:
            custom_main(token_args)

    def run_full_pipeline(self, add_drum: bool = False):
        """Convenience method to run both stages back-to-back."""
        print(f"=== Starting Full Pipeline for {self.dataset_name.upper()} ===")
        self.preprocess(add_drum=add_drum)
        self.tokenize()
        print(f"=== Finished. Data is in: {self.token_dir} ===")

if __name__ == "__main__":
    # Expecting: python3 -m ...anticipation_tokenizer [MODE] [OPTIONAL_NAME]
    # Example 1: python3 -m ... LAKH
    # Example 2: python3 -m ... CUSTOM jordan-progrock-dataset
    
    if len(sys.argv) < 2:
        print("Usage: python3 -m ... [LAKH | GIGAMIDI | CUSTOM] [dataset_name_if_custom]")
        sys.exit(1)

    mode = sys.argv[1].upper()
    
    if mode == "LAKH":
        dataset_name = "lakh-midi-clean"
        dataset_type = DatasetType.LAKH
        augment = 1
    elif mode == "GIGAMIDI":
        dataset_name = "giga-midi" # Ensure this matches your folder name
        dataset_type = DatasetType.LAKH # Giga usually follows Lakh structure
        augment = 1
    elif mode == "CUSTOM":
        if len(sys.argv) < 3:
            print("Error: CUSTOM mode requires a dataset name (e.g., jordan-progrock-dataset)")
            sys.exit(1)
        dataset_name = sys.argv[2]
        dataset_type = DatasetType.CUSTOM
        augment = 10 # Default higher augmentation for custom sets
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    master = AnticipationTokenizer(
        dataset_name=dataset_name, 
        dataset_type=dataset_type, 
        augment=augment, 
        interarrival=False
    )
    
    master.run_full_pipeline(add_drum=True)