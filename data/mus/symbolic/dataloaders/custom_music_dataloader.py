import torch
from torch.utils.data import Dataset
import os

class CustomMusicDataset(Dataset):
    """
    Generic dataloader for custom tokenized music datasets.
    
    Expects structure:
    /home/rebcecca/orcd/pool/music_datasets/{dataset_name}/
    └── tokens/
        ├── anticipation/
        │   ├── tokenized-events-train.txt
        │   ├── tokenized-events-validation.txt
        │   └── tokenized-events-test.txt
        ├── anticipation-vanilla/
        │   └── (same structure)
        └── miditok/
            └── (same structure)
    
    Each line in token files contains space-separated token IDs.
    """
    
    def __init__(self, hparams, dataset_name, split='train', tokenizer_type='anticipation'):
        """
        Args:
            hparams: hyperparameters containing context_length
            dataset_name: name of dataset (e.g., 'jordan-progrock-dataset', 'giga-midi')
            split: 'train', 'validation', or 'test'
            tokenizer_type: 'anticipation', 'anticipation-vanilla', or 'miditok'
        """
        self.hparams = hparams
        self.dataset_name = dataset_name
        self.split = split
        self.context_length = hparams.context_length
        self.tokenizer_type = tokenizer_type
        
        # Base path to music datasets
        base_data_path = os.getenv('CUSTOM_STORAGE_PATH', '/home/rebcecca/orcd/pool/music_datasets')
        
        # Path to token files
        self.tokens_dir = os.path.join(base_data_path, dataset_name, "tokens", tokenizer_type)
        self.token_file = os.path.join(self.tokens_dir, f"tokenized-events-{split}.txt")
        
        if not os.path.exists(self.token_file):
            raise FileNotFoundError(
                f"Token file not found: {self.token_file}\n"
                f"Expected structure:\n"
                f"  {os.path.join(base_data_path, dataset_name, 'tokens', tokenizer_type)}/\n"
                f"    ├── tokenized-events-train.txt\n"
                f"    ├── tokenized-events-validation.txt\n"
                f"    └── tokenized-events-test.txt"
            )
        
        # Load all sequences from the token file
        self.sequences = self._load_sequences()
        
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences found in {self.token_file}")
        
        print(f"✅ Loaded {len(self.sequences)} sequences from {dataset_name}/{tokenizer_type}/{split} split")
    
    def _load_sequences(self):
        """Load all token sequences from the file."""
        sequences = []
        with open(self.token_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        # Parse space-separated token IDs
                        tokens = list(map(int, line.split()))
                        sequences.append(torch.tensor(tokens, dtype=torch.long))
                    except ValueError as e:
                        print(f"⚠️  Warning: Could not parse line as token IDs: {line[:50]}...")
                        continue
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Return a sample with:
        - input_ids: sequence of token IDs, padded/truncated to context_length
        """
        sequence = self.sequences[idx]
        
        # Pad or truncate to context_length
        if len(sequence) < self.context_length:
            # Pad with zeros
            padded = torch.zeros(self.context_length, dtype=torch.long)
            padded[:len(sequence)] = sequence
            sequence = padded
        elif len(sequence) > self.context_length:
            # Truncate to context_length
            sequence = sequence[:self.context_length]
        
        return {"input_ids": sequence}
