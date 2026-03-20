import kagglehub
import shutil
import os
from pathlib import Path
from tokenization.constants import get_dataset_path, get_subset_path

class LakhDataset:
    def __init__(self):
        """
        Initializes the Lakh Dataset handler. 
        Organizes data into a 'midi' subfolder to stay consistent with the pipeline.
        """
        self.dataset_name = "lakh-midi-clean"
        
        # The 'Mother' path for the whole dataset
        self.root_path = Path(get_dataset_path(self.dataset_name))
        
        # The specific folder where raw MIDI files will live
        self.midi_path = Path(get_subset_path(self.dataset_name, "midi"))
        
    def is_installed(self) -> bool:
        """Checks if the midi directory exists and isn't empty."""
        return self.midi_path.exists() and any(self.midi_path.iterdir())

    def download(self, force=False):
        """
        Downloads the dataset from Kaggle and moves it into the 'midi' subfolder.
        """
        if self.is_installed() and not force:
            print(f"✅ Lakh MIDI files already exist at: {self.midi_path}")
            return

        print(f"🚀 Starting download for {self.dataset_name}...")
        
        # Download to Kaggle's temporary cache
        tmp_path = kagglehub.dataset_download("imsparsh/lakh-midi-clean")
        
        print(f"📦 Moving dataset from cache to: {self.midi_path}")
        
        # If the midi_path exists but we are forcing a download, clean it first
        if self.midi_path.exists():
            shutil.rmtree(self.midi_path)
            
        # Move the downloaded folder to be our 'midi' folder
        shutil.move(tmp_path, self.midi_path)
        
        # Create the tokens folder now so it's ready for the Master Script
        get_subset_path(self.dataset_name, "tokens")
        
        print(f"✨ Lakh Dataset successfully downloaded to {self.midi_path}")

    def get_stats(self):
        """Verify the download."""
        if not self.is_installed():
            print("Dataset not found. Run .download() first.")
            return
        
        file_count = sum(1 for _ in self.midi_path.rglob('*.mid*'))
        print(f"📊 Dataset Stats: {file_count} MIDI files found in {self.midi_path}")

if __name__ == "__main__":
    lmd = LakhDataset()
    lmd.download()
    lmd.get_stats()