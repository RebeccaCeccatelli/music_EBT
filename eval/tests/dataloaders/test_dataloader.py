import torch
from torch.utils.data import Dataset

class MusicSyntheticDataset(Dataset):
    """Synthetic music dataset that generates random MIDI token sequences for testing."""
    def __init__(self, hparams, size=10000):
        self.size = size
        self.context_length = hparams.context_length
        # REMI tokenizer has 284 tokens
        self.vocab_size = 284
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        random_sequence = torch.randint(1, self.vocab_size, (self.context_length,))
        return {"input_ids": random_sequence}
