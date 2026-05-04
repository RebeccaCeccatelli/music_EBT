"""
Baseline Llama-Based Transformer for Music (Symbolic MIDI Tokens)

Standard autoregressive transformer for next-token prediction.
Uses custom Llama2-inspired architecture as a comparison baseline against the Energy-Based Transformer (EBT).

Simple architecture: Embed → Transformer → Output Projection → CE Loss
"""

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy

import math
import random
import os
from model.model_utils import *
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer


class Baseline_Llama_Transformer_MUS(L.LightningModule):
    """
    Baseline Llama-Based Transformer for Music Generation (Symbolic)
    
    Standard next-token prediction model without energy-based training.
    Uses custom Llama2-inspired architecture (RMSNorm, Rotary embeddings, etc).
    Use this to benchmark against EBT and validate the benefit of iterative refinement.
    """
    
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):  # passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        # ========== MUSIC-SPECIFIC: Vocabulary Setup ==========
        # Load tokenizer using shared utility
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
        
        # ========== Token Embeddings ==========
        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(
            self.embeddings,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain
        )
        
        # ========== LogSoftmax for probability conversion ==========
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # ========== Transformer Architecture ==========
        self.transformer = setup_transformer(self.hparams)
        
        # ========== Output Projection: Embeddings → Vocabulary ==========
        self.output = nn.Linear(self.hparams.embedding_dim, self.vocab_size, bias=False)
        init_whole_model_weights(
            self.output,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain
        )
        
        self.finished_warming_up = False
    
    def forward(self, x, learning=True, return_raw_logits=False):
        """
        Standard transformer forward pass.
        
        Args:
            x: Input MIDI token IDs, shape (batch, seq_len)
            learning: Whether to enable gradients (unused, kept for compatibility)
            return_raw_logits: Return logits or log probabilities
        
        Returns:
            predicted_logits or predicted_distribution
        """
        # Embed input tokens
        embeddings = self.embeddings(x)  # (B, S, D)
        
        # Pass through transformer
        predicted_embeddings = self.transformer(embeddings, start_pos=0, learning=learning)  # (B, S, D)
        
        # Project to vocabulary
        predicted_logits = self.output(predicted_embeddings)  # (B, S, V)
        
        if return_raw_logits:
            return predicted_logits
        else:
            # Convert to log probabilities for NLL loss
            predicted_distribution = self.log_softmax(predicted_logits).reshape(-1, self.vocab_size)  # (B*S, V)
            return predicted_distribution
    
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
        predicted_distribution = self(input_ids)
        
        # Reshape targets to 1D
        next_token_indices = next_token_indices.reshape(-1)  # (B*S,)
        
        # Compute loss
        ce_loss = F.nll_loss(
            predicted_distribution,
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
    