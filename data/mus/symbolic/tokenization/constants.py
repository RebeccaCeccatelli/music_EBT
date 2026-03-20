import os
from pathlib import Path
from enum import Enum

# 1. Resolve Repo Root (assumes constants.py is in /tokenization/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_DATA = REPO_ROOT / "datasets" 

def get_data_root() -> Path:
    """Determine the 'Mother' directory based on Environment Variables."""
    remote_mother_dir = os.getenv("MUSIC_DATA_ROOT")
    
    if remote_mother_dir:
        path = Path(remote_mother_dir)
        mode = "REMOTE/CUSTOM"
    else:
        path = DEFAULT_LOCAL_DATA
        mode = "LOCAL/DEFAULT"

    print(f"--- [DATA CONFIG] Mode: {mode} ---")
    print(f"--- [DATA CONFIG] Mother Directory: {path} ---\n")
    return path

# Global constant for the project
MUSIC_DATA_ROOT = get_data_root()

def get_dataset_path(dataset_name: str) -> str:
    """Returns the absolute path to a specific dataset folder."""
    path = MUSIC_DATA_ROOT / dataset_name
    os.makedirs(path, exist_ok=True)
    return str(path)

def get_subset_path(dataset_name: str, subset: str) -> str:
    """Returns path to 'midi' or 'tokens' subfolders."""
    base_path = Path(get_dataset_path(dataset_name))
    target_path = base_path / subset
    os.makedirs(target_path, exist_ok=True)
    return str(target_path)

class DatasetType(str, Enum):
    CUSTOM = "custom"
    LAKH = "lakh"
    MAESTRO = "maestro"
    GIGA_MIDI = "gigaMIDI"

    @classmethod
    def list(cls):
        return [c.value for c in cls]