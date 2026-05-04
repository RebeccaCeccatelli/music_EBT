import json
import os
import random
import torch
from torch.utils.data import Dataset


class GigaMIDIMiditokDataset(Dataset):
    """
    Dataset for GigaMIDI pre-tokenized miditok JSON files.

    Each file contains {"ids": [token_id, ...]} for one song.
    __getitem__ samples a random context_length window from the song,
    so each epoch sees a different slice of each song (data augmentation).
    Init only lists files — no file I/O at startup.

    Directory layout:
      train:      {base}/{split}/all-instruments-with-drums/{shard}/*.json
      validation: {base}/{split}/*.json
      test:       {base}/{split}/*.json
    """

    TOKENIZER_CONFIG_PATH = (
        "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json"
    )

    def __init__(self, hparams, split="train"):
        self.context_length = hparams.context_length
        base = os.getenv(
            "CUSTOM_STORAGE_PATH", "/home/rebcecca/orcd/pool/music_datasets"
        )
        split_dir = os.path.join(base, "giga-midi", "tokens", "miditok", split)

        self.files = self._find_json_files(split_dir)
        if not self.files:
            raise FileNotFoundError(f"No JSON token files found under {split_dir}")

        print(
            f"GigaMIDIMiditokDataset [{split}]: {len(self.files)} songs, "
            f"context_length={self.context_length}"
        )

    def _find_json_files(self, split_dir):
        files = []
        for root, _, fnames in os.walk(split_dir):
            for fname in fnames:
                if fname.endswith(".json"):
                    files.append(os.path.join(root, fname))
        return sorted(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx]) as f:
            ids = json.load(f)["ids"]

        if len(ids) <= self.context_length:
            # Pad short sequences with zeros (pad token id)
            ids = ids + [0] * (self.context_length - len(ids) + 1)

        # Random window so each epoch sees different slices
        max_start = len(ids) - self.context_length
        start = random.randint(0, max_start)
        chunk = ids[start : start + self.context_length]
        return {"input_ids": torch.tensor(chunk, dtype=torch.long)}
