import os
from argparse import Namespace

# Import the main logic from existing scripts
from anticipation.train.midi_preprocess import main as preprocess_main
from anticipation.finetune.tokenize_custom import main as custom_main
from anticipation.train.tokenize_lakh import main as lakh_main

class AnticipationTokenizer:
    def __init__(self, dataset_name, datadir, augment=1, interarrival=False):
        """
        Args:
            dataset_name: 'custom', 'lakh', etc.
            datadir: Path to the specific dataset folder (e.g., ./finetune/MY_DATA)
            augment: -k factor (default 1)
            interarrival: Use -i encoding (default False)
        """
        self.dataset_name = dataset_name.lower()
        self.datadir = datadir
        self.augment = augment
        self.interarrival = interarrival

    def run_full_pipeline(self, add_drum=False):
        """Runs Preprocessing followed by Tokenization."""
        print(f"=== Starting Full Pipeline for {self.dataset_name.upper()} ===")
        
        # Step 1: Preprocess (MIDI -> Compound TXT)
        self.preprocess(add_drum=add_drum)
        
        # Step 2: Tokenize (Compound TXT -> Anticipation Tokens)
        self.tokenize()
        
        print("=== Pipeline Finished Successfully ===")

    def preprocess(self, add_drum=False):
        print(f"\n--- [1/2] Preprocessing MIDI files in: {self.datadir} ---")
        # Mocking args for midi-preprocess.py
        # Note: midi-preprocess uses 'dir' as the argument name
        preproc_args = Namespace(
            dir=self.datadir,
            add_drum=add_drum
        )
        preprocess_main(preproc_args)

    def tokenize(self):
        print(f"\n--- [2/2] Tokenizing into Anticipation format ---")
        
        if self.dataset_name == "lakh":
            # Call lakh_main(lakh_args) here when implemented
            print("Lakh logic would go here.") #TODO
        else:
            # Mocking args for tokenize-custom.py
            # Note: tokenize-custom uses 'datadir' as the argument name
            token_args = Namespace(
                datadir=self.datadir,
                augment=self.augment,
                interarrival=self.interarrival
            )
            custom_main(token_args)

if __name__ == "__main__":
    # Get the absolute path to the directory where THIS script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    MY_DATASET = "jordan-progrock-dataset"
    
    # ADD "anticipation" to the join list here:
    PATH = os.path.join(BASE_DIR, "anticipation", "finetune", MY_DATASET)
    
    print(f"Targeting Absolute Path: {PATH}")
    
    master = AnticipationTokenizer(
        dataset_name="custom", 
        datadir=PATH, 
        augment=10, 
        interarrival=False
    )
    
    master.run_full_pipeline(add_drum=True)