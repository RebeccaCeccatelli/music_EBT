import os
from argparse import Namespace

from constants import DatasetType, get_dataset_path

# Import the main logic from existing scripts
from anticipation.train.midi_preprocess import main as preprocess_main
from anticipation.finetune.tokenize_custom import main as custom_main
from anticipation.train.tokenize_lakh import main as lakh_main

class AnticipationTokenizer:
    def __init__(self, dataset_name: DatasetType, datadir: str, augment: int = 1, interarrival: bool = False):
        """
        Args:
            dataset_name: DatasetType enum instance (e.g., DatasetType.CUSTOM)
            datadir: Path to the specific dataset folder
            augment: -k factor (default 1)
            interarrival: Use -i encoding (default False)
        """
        # Ensure we are working with the Enum type
        self.dataset_name = DatasetType(dataset_name)
        self.datadir = datadir
        self.augment = augment
        self.interarrival = interarrival

    def preprocess(self, add_drum: bool = False):
        print(f"\n--- [1/2] Preprocessing MIDI files in: {self.datadir} ---")
        
        preproc_args = Namespace(
            dir=self.datadir,
            add_drum=add_drum
        )
        preprocess_main(preproc_args)

    def tokenize(self):
        print(f"\n--- [2/2] Tokenizing into Anticipation format ---")
        
        if self.dataset_name == DatasetType.LAKH:
            # Lakh expects a specific folder structure (train/val/test)
            # and relies on anticipation.config for split names.
            lakh_args = Namespace(
                datadir=self.datadir,
                augment=self.augment,
                interarrival=self.interarrival
            )
            print(f"Calling Lakh Tokenizer logic on {self.datadir}...")
            lakh_main(lakh_args)
        else:
            # Default to custom tokenization logic
            token_args = Namespace(
                datadir=self.datadir,
                augment=self.augment,
                interarrival=self.interarrival
            )
            custom_main(token_args)

    def run_full_pipeline(self, add_drum: bool = False):
        """Runs Preprocessing followed by Tokenization."""
        # .upper() works because DatasetType inherits from str
        print(f"=== Starting Full Pipeline for {self.dataset_name.upper()} ===")
        
        # Step 1: Preprocess (MIDI -> Compound TXT)
        self.preprocess(add_drum=add_drum)
        
        # Step 2: Tokenize (Compound TXT -> Anticipation Tokens)
        self.tokenize()
        
        print("=== Pipeline Finished Successfully ===")

if __name__ == "__main__":
    # 1. Define your dataset name as a string (or use the Enum directly)
    MY_DATASET_NAME = "jordan-progrock-dataset"
    
    # 2. Use your new path helper
    PATH = get_dataset_path(MY_DATASET_NAME)
    
    print(f"Targeting Absolute Path: {PATH}")
    
    # 3. Instantiate using the Enum
    master = AnticipationTokenizer(
        dataset_name=DatasetType.CUSTOM, 
        datadir=PATH, 
        augment=10, 
        interarrival=False
    )
    
    master.run_full_pipeline(add_drum=True)