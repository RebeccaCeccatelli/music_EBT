import kagglehub
import shutil
import os
from pathlib import Path
from path_utils import get_dataset_path, get_subset_path
from dataloaders.constants import DatasetType

class LakhDataset:
    def __init__(
        self, 
        kaggle_id: str = "imsparsh/lakh-midi-clean",
        version_suffix: str = "clean"
    ):
        self.kaggle_id = kaggle_id
        
        # Create a unique name like "lakh-clean" or "lakh-matched"
        self.dataset_name = f"{DatasetType.LAKH.value}-{version_suffix}"
        
        self.root_path = Path(get_dataset_path(self.dataset_name))
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
        tmp_path = kagglehub.dataset_download(self.kaggle_id)
        
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