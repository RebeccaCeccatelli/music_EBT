import sys
import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from path_utils import get_dataset_path
from dataloaders.constants import DatasetType

class MidiTokTokenizer:
    def __init__(self, dataset_name: str, dataset_type: DatasetType, encoding="REMI"):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.root_datadir = get_dataset_path(dataset_name)
        self.token_base_dir = os.path.join(self.root_datadir, "tokens", "miditok")
        
        config = TokenizerConfig(num_velocities=32, use_chords=True, use_programs=True)
        self.tokenizer = REMI(config)
        
        config = TokenizerConfig(num_velocities=32, use_chords=True, use_programs=True)
        self.tokenizer = REMI(config)

    def _get_midi_files(self, search_path):
        p = Path(search_path)
        return list(p.rglob("*.mid")) + list(p.rglob("*.midi"))

    def tokenize(self):
        print(f"\n=== [MidiTok] Processing {self.dataset_name} ===")
        os.makedirs(self.token_base_dir, exist_ok=True)

        splits = ['train', 'validation', 'test']
        has_splits = any(os.path.isdir(os.path.join(self.root_datadir, s)) for s in splits)

        if has_splits:
            for split in splits:
                split_path = os.path.join(self.root_datadir, split)
                if not os.path.isdir(split_path): continue
                
                output_dir = os.path.join(self.token_base_dir, split)
                os.makedirs(output_dir, exist_ok=True)
                
                files = self._get_midi_files(split_path)
                if files:
                    print(f"Tokenizing {len(files)} files for split: {split}...")
                    self.tokenizer.tokenize_dataset(files, output_dir)
        else:
            midi_input_dir = os.path.join(self.root_datadir, "midi")
            search_path = midi_input_dir if os.path.isdir(midi_input_dir) else self.root_datadir
            
            files = self._get_midi_files(search_path)
            if files:
                print(f"Tokenizing {len(files)} files into {self.token_base_dir}...")
                self.tokenizer.tokenize_dataset(files, self.token_base_dir)

        tokenizer_json_path = os.path.join(self.token_base_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_json_path)
        print(f"Done! Tokenizer config saved to: {tokenizer_json_path}")

    def run_full_pipeline(self):
        self.tokenize()

if __name__ == "__main__":
    # 1. Validation & Help Message
    if len(sys.argv) < 2:
        options = ", ".join(DatasetType.list())
        print(f"Usage: python3 -m tokenization.miditok_tokenizer [{options}] [dataset_name_if_custom]")
        sys.exit(1)

    # 2. Safe Enum Conversion
    try:
        mode = DatasetType(sys.argv[1].lower())
    except ValueError:
        print(f"❌ Error: '{sys.argv[1]}' is not a valid DatasetType.")
        print(f"Valid options are: {DatasetType.list()}")
        sys.exit(1)

    # 3. Resolve Dataset Name based on Enum
    if mode == DatasetType.LAKH:
        dataset_name = "lakh-midi-clean"
    elif mode == DatasetType.GIGA_MIDI:
        dataset_name = "giga-midi"
    elif mode == DatasetType.CUSTOM:
        if len(sys.argv) < 3:
            print("❌ Error: CUSTOM mode requires a dataset name (e.g., jordan-progrock-dataset)")
            sys.exit(1)
        dataset_name = sys.argv[2]
    else:
        # For MAESTRO or others added to Enum but not logic
        dataset_name = mode.value 

    # 4. Initialize and run
    master = MidiTokTokenizer(dataset_name=dataset_name, dataset_type=mode)
    master.run_full_pipeline()
