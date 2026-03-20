import os
from pathlib import Path
from enum import Enum

# 1. Resolve Repo Root 
# FILE_PATH is /.../music_EBT/data/mus/symbolic/path_utils.py
FILE_PATH = Path(__file__).resolve()

# DEFAULT_LOCAL_DATA will be /.../music_EBT/data/mus/symbolic/datasets
# This matches the 'datasets' folder visible in your sidebar
DEFAULT_LOCAL_DATA = FILE_PATH.parent / "datasets" 

def get_data_root() -> Path:
    """Determine the 'Mother' directory based on Environment Variables."""
    remote_mother_dir = os.getenv("REMOTE_DATA_STORAGE")
    
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
# This runs once upon the first import of path_utils
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