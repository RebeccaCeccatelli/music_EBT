import torch
from torch.utils.data import Dataset
import os
import numpy as np


class CustomMusicDataset(Dataset):
    """
    Lazy-loading dataset for pre-tokenized music sequences.

    Reads a large flat text file (one space-separated sequence per line) using
    pre-computed byte offsets, so each DataLoader worker seeks directly to its
    assigned lines without loading the full file into memory.

    The byte-offset index is built once on first use and cached alongside the
    token file (<file>.idx.npy). Subsequent runs skip the scan entirely.

    Directory layout expected:
        {base}/{dataset_name}/tokens/{tokenizer_type}/
            tokenized-events-train.txt
            tokenized-events-validation.txt
            tokenized-events-test.txt
    """

    def __init__(self, hparams, dataset_name, split='train', tokenizer_type='anticipation', pad_token_id=0):
        self.context_length = hparams.context_length
        self.pad_token_id = pad_token_id

        base_data_path = os.getenv('CUSTOM_STORAGE_PATH', '/home/rebcecca/orcd/pool/music_datasets')
        self.tokens_dir = os.path.join(base_data_path, dataset_name, "tokens", tokenizer_type)
        self.token_file = os.path.join(self.tokens_dir, f"tokenized-events-{split}.txt")

        if not os.path.exists(self.token_file):
            raise FileNotFoundError(
                f"Token file not found: {self.token_file}\n"
                f"Expected structure:\n"
                f"  {self.tokens_dir}/\n"
                f"    ├── tokenized-events-train.txt\n"
                f"    ├── tokenized-events-validation.txt\n"
                f"    └── tokenized-events-test.txt"
            )

        # Build/load offset index in the main process before workers are forked.
        self.offsets = self._get_offsets()
        # File handle is None here; each worker opens its own handle lazily in __getitem__.
        self._file = None

        print(f"✅ Indexed {len(self.offsets)} sequences from {dataset_name}/{tokenizer_type}/{split}")

    # ------------------------------------------------------------------
    # Offset index
    # ------------------------------------------------------------------

    def _get_offsets(self):
        """Return a numpy array of byte offsets, one per non-empty line.

        Builds the index by scanning the file for newlines (fast — no integer
        parsing) and saves it as a .npy file next to the token file so the scan
        only happens once.
        """
        idx_path = self.token_file + '.idx.npy'
        token_mtime = os.path.getmtime(self.token_file)

        if os.path.exists(idx_path) and os.path.getmtime(idx_path) >= token_mtime:
            return np.load(idx_path)

        print(f"Building line index for {os.path.basename(self.token_file)} (first run only) ...")
        offsets = []
        with open(self.token_file, 'rb') as f:
            offset = 0
            for line in f:
                if line.strip():
                    offsets.append(offset)
                offset += len(line)

        offsets = np.array(offsets, dtype=np.int64)
        np.save(idx_path, offsets)
        print(f"  → {len(offsets)} sequences indexed, cached to {os.path.basename(idx_path)}")
        return offsets

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.offsets)

    def _get_file(self):
        """Return a per-worker file handle, opening it on first access after fork."""
        if self._file is None:
            self._file = open(self.token_file, 'rb')
        return self._file

    def __getitem__(self, idx):
        f = self._get_file()
        f.seek(int(self.offsets[idx]))
        line = f.readline().decode('utf-8').strip()
        tokens = list(map(int, line.split()))

        if len(tokens) < self.context_length:
            tokens = tokens + [self.pad_token_id] * (self.context_length - len(tokens))
        elif len(tokens) > self.context_length:
            tokens = tokens[:self.context_length]

        return {"input_ids": torch.tensor(tokens, dtype=torch.long)}
