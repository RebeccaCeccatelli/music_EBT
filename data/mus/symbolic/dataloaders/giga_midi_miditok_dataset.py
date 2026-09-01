import json
import os
import random
import torch
from torch.utils.data import Dataset


class GigaMIDIMiditokDataset(Dataset):
    """
    Dataset for GigaMIDI pre-tokenized miditok JSON files.

    Each file contains {"ids": [token_id, ...]} for one song.
    __getitem__ samples a context_length window from the song. For the train
    split this window is randomized per call (data augmentation — each epoch
    sees a different slice). For validation/test it's a fixed, per-song
    deterministic window (seeded by song index) instead, so valid_loss means
    the same thing across checkpoints/epochs/resumes — an unseeded random
    window there previously made validation content change from one check to
    the next, confounding genuine model-quality changes with which slice of
    each song happened to get sampled.
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
        self.split = split
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

    def get_full_tokens(self, idx):
        """Return the full token list for song `idx` from the beginning of the song.

        Useful for inference when you want a prompt from the song's actual start
        rather than the random window __getitem__ samples for data augmentation.
        """
        with open(self.files[idx]) as f:
            return json.load(f)["ids"]

    def __getitem__(self, idx):
        ids = self.get_full_tokens(idx)

        if len(ids) <= self.context_length:
            # Pad short sequences with zeros (pad token id)
            ids = ids + [0] * (self.context_length - len(ids) + 1)

        max_start = len(ids) - self.context_length
        if self.split == "train":
            # Random window so each epoch sees different slices (augmentation).
            start = random.randint(0, max_start)
        else:
            # Fixed per-song window, independent of global RNG state/epoch/
            # worker — so this exact slice is scored the same way every time.
            start = random.Random(idx).randint(0, max_start)
        chunk = ids[start : start + self.context_length]
        return {"input_ids": torch.tensor(chunk, dtype=torch.long)}
