import os
from enum import Enum

class DatasetType(str, Enum):
    CUSTOM = "custom"
    LAKH = "lakh"
    MAESTRO = "maestro"
    GIGA_MIDI = "gigaMIDI"
    POP909 = "pop909"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

# This file is in /symbolic/datasets/constants.py
# We want the root to be /symbolic/
DATASETS_DIR = os.path.dirname(os.path.abspath(__file__)) 
SYMBOLIC_ROOT = os.path.dirname(DATASETS_DIR)

# Adjust this path based on where your 'anticipation' folder actually lives.
# If 'anticipation' is inside 'symbolic', use this:
FINETUNE_DIR = os.path.join(SYMBOLIC_ROOT, "tokenization", "anticipation", "finetune")

def get_dataset_path(dataset_name: str) -> str:
    print(DATASETS_DIR)
    print(SYMBOLIC_ROOT)
    print(FINETUNE_DIR)
    return os.path.join(FINETUNE_DIR, dataset_name)