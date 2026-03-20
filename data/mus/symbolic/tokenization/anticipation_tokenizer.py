import os
from argparse import Namespace
from tokenization.constants import DatasetType, get_subset_path, get_dataset_path

# Import the main logic from your sub-modules
from tokenization.anticipation.train.midi_preprocess import main as preprocess_main
from tokenization.anticipation.finetune.tokenize_custom import main as custom_main
from tokenization.anticipation.train.tokenize_lakh import main as lakh_main

class AnticipationTokenizer:
    def __init__(self, dataset_name: str, dataset_type: DatasetType, augment: int = 1, interarrival: bool = False):
        self.dataset_name = dataset_name
        self.dataset_type = DatasetType(dataset_type)
        
        # Resolve clean subfolders
        self.midi_dir = get_subset_path(self.dataset_name, "midi")
        self.token_dir = os.path.join(get_dataset_path(self.dataset_name), "tokens", "anticipation")

        # Ensure the directory exists before any workers try to write to it
        os.makedirs(self.token_dir, exist_ok=True)
        
        self.augment = augment
        self.interarrival = interarrival

    def preprocess(self, add_drum: bool = False):
        print(f"\n--- [1/2] Preprocessing MIDI files in: {self.midi_dir} ---")
        preproc_args = Namespace(dir=self.midi_dir, add_drum=add_drum)
        preprocess_main(preproc_args)

    def tokenize(self):
        print(f"\n--- [2/2] Tokenizing results into: {self.token_dir} ---")
        
        token_args = Namespace(
            midi_dir=self.midi_dir,    # Input for worker
            token_dir=self.token_dir,  # Output for worker
            dataset_name=self.dataset_name,
            augment=self.augment,
            interarrival=self.interarrival
        )

        if self.dataset_type == DatasetType.LAKH:
            lakh_main(token_args)
        else:
            custom_main(token_args)

    def run_full_pipeline(self, add_drum: bool = False):
        print(f"=== Starting Full Pipeline for {self.dataset_name.upper()} ===")
        self.preprocess(add_drum=add_drum)
        self.tokenize()
        print(f"=== Finished. Data is in: {self.token_dir} ===")

if __name__ == "__main__":
    # Example usage for your Jordan dataset
    master = AnticipationTokenizer(
        dataset_name="jordan-progrock-dataset", 
        dataset_type=DatasetType.CUSTOM, 
        augment=10, 
        interarrival=False
    )
    master.run_full_pipeline(add_drum=True)