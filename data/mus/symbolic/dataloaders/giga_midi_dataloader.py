import os
import zipfile
import shutil
from huggingface_hub import hf_hub_download
from path_utils import get_dataset_path
from dataloaders.constants import DatasetType

class GigaMIDIDataset:
    def __init__(self):
        self.dataset_type = DatasetType.GIGA_MIDI
        self.dataset_name = "giga-midi"
        self.repo_id = "Metacreation/GigaMIDI"
        self.zip_filename = "Final_GigaMIDI_V2.0_Final.zip"
        
        self.root_path = get_dataset_path(self.dataset_name)
        self.midi_path = os.path.join(self.root_path, "midi")
        os.makedirs(self.root_path, exist_ok=True)

    def download_and_extract(self):
        """Downloads the V2.0 ZIP and handles nested extraction."""
        
        # 1. Verification Check
        # We check for 'training' folder specifically because that's where the actual MIDIs live
        final_check_path = os.path.join(self.midi_path, "training")
        if os.path.exists(final_check_path) and any(os.scandir(final_check_path)):
            print(f"✅ GigaMIDI (unzipped) already exists in {self.midi_path}. Skipping.")
            return

        # 2. Download from Hugging Face
        print(f"🚀 [1/3] Downloading {self.zip_filename} from Hugging Face...")
        local_zip_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.zip_filename,
            repo_type="dataset",
            local_dir=self.root_path,
            token=os.getenv("HUGGING_FACE_HUB_TOKEN")
        )

        # 3. First Extraction (The Wrapper ZIP)
        print(f"📦 [2/3] Extracting Wrapper ZIP to {self.midi_path}...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.midi_path)

        # 4. Nested Extraction (The Split ZIPs)
        # Based on your ls, they are in: midi/Final_GigaMIDI_V1.1_Final/
        nested_dir = os.path.join(self.midi_path, "Final_GigaMIDI_V1.1_Final")
        
        if os.path.exists(nested_dir):
            print(f"📦 [3/3] Extracting Nested Split ZIPs...")
            for split_zip in ["training-V1.1-80%.zip", "validation-V1.1-10%.zip", "test-V1.1-10%.zip"]:
                zip_path = os.path.join(nested_dir, split_zip)
                if os.path.exists(zip_path):
                    print(f"   解压 {split_zip}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # Extract directly into self.midi_path to flatten the structure
                        zip_ref.extractall(self.midi_path)
            
            # 5. Cleanup Nested Folder to save space and avoid path confusion
            print("🧹 Cleaning up temporary nested ZIPs...")
            shutil.rmtree(nested_dir)
            if os.path.exists(os.path.join(self.midi_path, "__MACOSX")):
                shutil.rmtree(os.path.join(self.midi_path, "__MACOSX"))
        
        print(f"✨ GigaMIDI structure ready at: {self.midi_path}")

if __name__ == "__main__":
    loader = GigaMIDIDataset()
    loader.download_and_extract()