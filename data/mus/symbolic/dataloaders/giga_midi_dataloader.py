import os
import zipfile
import shutil
from huggingface_hub import hf_hub_download
from data.mus.symbolic.path_utils import get_dataset_path
from data.mus.symbolic.dataloaders.constants import DatasetType

class GigaMIDIDataset:
    def __init__(self):
        self.dataset_type = DatasetType.GIGA_MIDI
        self.dataset_name = "giga-midi"
        self.repo_id = "Metacreation/GigaMIDI"
        self.zip_filename = "Final_GigaMIDI_V2.0_Final.zip"
        
        self.root_path = get_dataset_path(self.dataset_name)
        self.midi_path = os.path.join(self.root_path, "midi")
        os.makedirs(self.midi_path, exist_ok=True)
    
    def download_and_extract(self):
        # 1. Define Mappings (ZIP name in the HF repo : Final Folder Name)
        splits = {
            "training-V1.1-80%.zip": "train",
            "validation-V1.1-10%.zip": "validation",
            "test-V1.1-10%.zip": "test"
        }

        # Check if MIDI files exist (don't just check if the folder exists)
        def has_midi(path):
            for root, _, files in os.walk(path):
                if any(f.lower().endswith(('.mid', '.midi')) for f in files):
                    return True
            return False

        if all(os.path.exists(os.path.join(self.midi_path, s)) and has_midi(os.path.join(self.midi_path, s)) for s in splits.values()):
            print("✅ All splits already extracted and contain MIDI. Skipping.")
            return

        # 2. Download Main ZIP (GigaMIDI V2.0 Wrapper)
        local_zip_path = os.path.join(self.root_path, self.zip_filename)
        if not os.path.exists(local_zip_path):
            print(f"🚀 Downloading {self.zip_filename}...")
            local_zip_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.zip_filename,
                repo_type="dataset",
                local_dir=self.root_path,
                token=os.getenv("HUGGING_FACE_HUB_TOKEN")
            )
        else:
            print(f"📦 Main ZIP already exists. Skipping download.")

        # 3. Extract Wrapper (This yields the 3 split ZIPs)
        # We extract this into a temporary "raw" folder to keep things clean
        raw_extract_path = os.path.join(self.midi_path, "raw_zips")
        if not any(os.path.exists(os.path.join(raw_extract_path, z)) for z in splits.keys()):
            print("📦 Extracting main wrapper ZIP...")
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_extract_path)

        # 4. Deep Extraction Logic
        for zip_name, final_name in splits.items():
            target_path = os.path.join(self.midi_path, final_name)
            
            if os.path.exists(target_path) and has_midi(target_path):
                print(f"⏩ {final_name} already processed. Skipping.")
                continue

            os.makedirs(target_path, exist_ok=True)
            
            # Find the split zip (it might be in raw_extract_path or a subfolder)
            split_zip_path = None
            for root, _, files in os.walk(raw_extract_path):
                if zip_name in files:
                    split_zip_path = os.path.join(root, zip_name)
                    break
            
            if not split_zip_path:
                print(f"❌ Error: Could not find {zip_name} in the extracted files.")
                continue

            print(f"📂 Processing split: {final_name}...")
            temp_split_dir = os.path.join(self.midi_path, f"temp_{final_name}")
            
            # Step A: Extract the split zip (e.g., training-V1.1-80%.zip)
            with zipfile.ZipFile(split_zip_path, 'r') as z:
                # Only extract files that don't start with ._ or __MACOSX
                member_list = [m for m in z.namelist() if "__MACOSX" not in m and not os.path.basename(m).startswith("._")]
                z.extractall(temp_split_dir, members=member_list)

            # Step B: Find and extract any nested ZIPs (the "all-instruments-with-drums.zip")
            for root, _, files in os.walk(temp_split_dir):
                for file in files:
                    if file.endswith(".zip") and "all-instruments-with-drums" in file:
                        inner_zip_path = os.path.join(root, file)
                        print(f"  📦 Found nested ZIP: {file}. Extracting to {final_name}...")
                        with zipfile.ZipFile(inner_zip_path, 'r') as iz:
                            iz.extractall(target_path)
                    
                    # Also catch raw MIDIs if they weren't zipped (common in test/val)
                    elif file.lower().endswith(('.mid', '.midi')) and "all-instruments-with-drums" in root:
                        shutil.copy(os.path.join(root, file), os.path.join(target_path, file))

            # Clean up the temporary split directory
            shutil.rmtree(temp_split_dir)
            print(f"✅ Finished extracting {final_name}")

        # 5. Final Cleanup
        if os.path.exists(raw_extract_path):
            shutil.rmtree(raw_extract_path)
        
        macosx_dir = os.path.join(self.midi_path, "__MACOSX")
        if os.path.exists(macosx_dir):
            shutil.rmtree(macosx_dir)
        
        print(f"✨ GigaMIDI is fully extracted in {self.midi_path}")

if __name__ == "__main__":
    loader = GigaMIDIDataset()
    loader.download_and_extract()