"""
Baseline HF GPT2 Transformer for Music (Symbolic MIDI Tokens)

Uses the pre-trained GPT2 architecture from Hugging Face Transformers library.
This serves as a reference comparison baseline to validate the benefits of EBT.

Note: GPT2 has a fixed vocab of 50257, so we map REMI tokens to this space.
"""

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

import math
import os
from model.model_utils import *
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer


class Baseline_HF_GPT2_Transformer_MUS(L.LightningModule):
    """
    Baseline HF GPT2 Transformer for Music Generation (Symbolic)
    
    Uses the standard GPT2 architecture from transformers library as a reference.
    Automatically adapts the vocabulary size to match your REMI tokenizer.
    """
    
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        # ========== MUSIC-SPECIFIC: Vocabulary Setup ==========
        tokenizer_type = self.hparams.get('tokenizer_type', 'REMI')
        tokenizer_config_path = self.hparams.get('tokenizer_config_path', None)
        dataset_name = self.hparams.get('dataset_name', 'giga-midi')
        use_vanilla = self.hparams.get('use_vanilla', False)
        
        self.tokenizer, self.vocab_size, self.pad_token_id = load_tokenizer(
            tokenizer_type=tokenizer_type,
            tokenizer_config_path=tokenizer_config_path,
            dataset_name=dataset_name,
            use_vanilla=use_vanilla
        )
        
        print(f"Loaded {tokenizer_type} tokenizer with vocab_size={self.vocab_size}, pad_token_id={self.pad_token_id}")
        
        # ========== GPT2 Configuration ==========
        # Create custom GPT2 config matching your hyperparameters
        config = GPT2Config(
            n_embd=self.hparams.embedding_dim,
            n_layer=self.hparams.num_transformer_blocks,
            n_head=self.hparams.multiheaded_attention_heads,
            vocab_size=self.vocab_size,
            n_positions=self.hparams.context_length,
            use_cache=True,  # Enable KV caching for inference
        )
        
        self.model = GPT2LMHeadModel(config)
        
        # Resize token embeddings to match your vocabulary
        self.model.resize_token_embeddings(self.vocab_size)
        
        print(f"Initialized GPT2 with config:")
        print(f"  - Embedding dim: {config.n_embd}")
        print(f"  - Layers: {config.n_layer}")
        print(f"  - Heads: {config.n_head}")
        print(f"  - Vocab size: {self.vocab_size}")
        print(f"  - Context length: {config.n_positions}")
        
        self.finished_warming_up = False
    
    def forward(self, x, learning=True, return_raw_logits=False, attention_mask=None):
        """
        GPT2 forward pass.
        
        Args:
            x: Input MIDI token IDs, shape (batch, seq_len)
            learning: Whether model is in training mode (passed to model)
            return_raw_logits: Return logits (True) or loss (False during training)
            attention_mask: Optional attention mask for padding
        
        Returns:
            logits or loss depending on return_raw_logits and training mode
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (x != self.pad_token_id).long()
        
        # Forward pass through GPT2
        outputs = self.model(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=False,
        )
        
        logits = outputs.logits  # (B, S, V)
        
        if return_raw_logits:
            return logits
        else:
            return logits
    
    def forward_loss_wrapper(self, batch, phase="train"):
        """
        Compute next-token prediction loss.
        
        Args:
            batch: Dict with 'input_ids' (MIDI tokens)
            phase: "train", "valid", or "test"
        
        Returns:
            Dict with 'loss' and 'perplexity'
        """
        # Extract input and target tokens
        input_ids = batch['input_ids'].squeeze(dim=1)[:, :-1]  # All but last token
        next_token_indices = batch['input_ids'].squeeze(dim=1)[:, 1:]  # Ground truth
        
        # Forward pass
        logits = self(input_ids, learning=(phase == "train"), return_raw_logits=True)
        
        # Reshape for loss computation
        logits_flat = logits.reshape(-1, self.vocab_size)  # (B*S, V)
        next_token_indices = next_token_indices.reshape(-1)  # (B*S,)
        
        # Compute loss
        ce_loss = F.cross_entropy(
            logits_flat,
            next_token_indices,
            ignore_index=self.pad_token_id
        )
        
        # Compute perplexity
        ppl_loss = torch.exp(ce_loss).detach()
        
        # Return metrics
        log_dict = {
            'loss': ce_loss,
            'perplexity': ppl_loss
        }
        return log_dict
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.peak_learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        def lr_lambda(current_step):
            if current_step < self.hparams.warm_up_steps:
                # Warmup phase
                return float(current_step) / float(max(1, self.hparams.warm_up_steps))
            else:
                # Cosine annealing phase
                progress = float(current_step - self.hparams.warm_up_steps) / float(max(1, self.hparams.max_scheduling_steps - self.hparams.warm_up_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
