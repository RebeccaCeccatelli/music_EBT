"""
Energy-Based Transformer for Music (Symbolic MIDI Tokens)

Adapted from NLP EBT but rewritten cleanly for music generation.
Uses MCMC-style iterative refinement to generate high-quality MIDI sequences.
"""

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L

import math
import os
from model.model_utils import *
from model.replay_buffer import CausalReplayBuffer
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer


class EBT_MUS_SYMB(nn.Module):
    """
    Energy-Based Transformer for Music Generation (Symbolic)
    
    Works with tokenized MIDI (e.g., REMI tokens from miditok).
    Uses iterative refinement via MCMC to improve predictions step-by-step.
    """
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams if isinstance(hparams, dict) else vars(hparams)
        
        # ========== MUSIC-SPECIFIC: Vocabulary Setup ==========
        # Load tokenizer using shared utility
        tokenizer_type = self.hparams.get('tokenizer_type', 'REMI')
        tokenizer_config_path = self.hparams.get('tokenizer_config_path', None)
        dataset_name = self.hparams.get('dataset_name', 'giga-midi')
        use_vanilla = self.hparams.get('use_vanilla', False)  # New: vanilla vocabulary support
        
        self.tokenizer, self.vocab_size, self.pad_token_id = load_tokenizer(
            tokenizer_type=tokenizer_type,
            tokenizer_config_path=tokenizer_config_path,
            dataset_name=dataset_name,
            use_vanilla=use_vanilla  # New: pass vanilla flag to tokenizer
        )
        
        print(f"Loaded {tokenizer_type} tokenizer with vocab_size={self.vocab_size}, pad_token_id={self.pad_token_id}")
        
        # ========== Token Embeddings ==========
        # Map MIDI tokens to embedding space
        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(
            self.embeddings,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain
        )
        
        # ========== Vocab-to-Embedding Projection ==========
        # Convert probability distributions over tokens back to embeddings
        # This is key to EBT: we work with distributions, not direct embeddings
        self.vocab_to_embed = nn.Linear(
            self.vocab_size,
            self.hparams.embedding_dim,
            bias=False
        )
        init_whole_model_weights(
            self.vocab_to_embed,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain
        )
        
        # ========== Softmax/LogSoftmax for probability conversion ==========
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # ========== MCMC Parameters ==========
        # Alpha: MCMC step size (how far to move during each gradient step)
        self.alpha = nn.Parameter(
            torch.tensor(float(self.hparams.mcmc_step_size)),
            requires_grad=self.hparams.mcmc_step_size_learnable
        )
        
        # Langevin dynamics noise: adds exploration during MCMC
        self.langevin_dynamics_noise_std = nn.Parameter(
            torch.tensor(float(self.hparams.langevin_dynamics_noise)),
            requires_grad=False  # Can be enabled in warm_up_finished()
        )
        
        # ========== Transformer Architecture ==========
        self.transformer = setup_ebt(self.hparams)
        
        # ========== Optional: Replay Buffer for MCMC ==========
        # Reuse good predictions from previous batches as starting points
        self.mcmc_replay_buffer = (
            'mcmc_replay_buffer' in self.hparams and 
            self.hparams.mcmc_replay_buffer and 
            self.hparams.execution_mode != "inference"
        )
        if self.mcmc_replay_buffer:
            self.replay_buffer_samples = (
                self.hparams.batch_size_per_device * 
                self.hparams.mcmc_replay_buffer_sample_bs_percent
            )
            self.replay_buffer = CausalReplayBuffer(
                max_size=self.hparams.mcmc_replay_buffer_size,
                sample_size=self.replay_buffer_samples
            )
        
        # ========== Debugging ==========
        if self.hparams.debug_unused_parameters:
            self.used_parameters = set()
            self.parameters_not_to_check = set()
        
        self.finished_warming_up = False
    
    # =========================================================================
    # FORWARD PASS: MCMC Iterative Refinement
    # =========================================================================
    
    def forward(self, x, learning=True, return_raw_logits=False, 
                replay_buffer_logits=None, no_randomness=True):
        """
        MCMC-based forward pass for music generation.
        
        Args:
            x: Input MIDI token IDs, shape (batch, seq_len)
            learning: Whether to enable gradients for MCMC
            return_raw_logits: Return logits or log probabilities
            replay_buffer_logits: Reuse predictions from replay buffer
            no_randomness: Disable randomization (for validation/inference)
        
        Returns:
            predicted_distributions: List of predictions at each MCMC step
            predicted_energies: List of energy values at each MCMC step
        """
        
        predicted_distributions = []
        predicted_energies = []
        
        # ===== Step 1: Embed input tokens =====
        real_embeddings_input = self.embeddings(x)  # (B, S, D)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        # ===== Step 2: Initialize MCMC with noise =====
        # Start with random logits over vocabulary (or use replay buffer)
        predicted_tokens = self._corrupt_embeddings(real_embeddings_input)  # (B, S, V)
        
        if replay_buffer_logits is not None:
            # Use replayed predictions for some samples
            predicted_tokens[batch_size - replay_buffer_logits.shape[0]:] = replay_buffer_logits
        
        # ===== Step 3: Clamp alpha (avoid division by zero) =====
        alpha = torch.clamp(self.alpha, min=0.0001)
        
        # ===== Step 4: Randomize step size if enabled =====
        if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
            expanded_alpha = alpha.expand(batch_size, seq_length, 1)
            scale = self.hparams.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)
        
        # ===== Step 5: MCMC Loop - Iteratively refine predictions =====
        with torch.set_grad_enabled(True):
            for mcmc_step in range(self.hparams.mcmc_num_steps):
                
                # --- 5a: Make predictions differentiable ---
                if self.hparams.no_mcmc_detach:
                    predicted_tokens.requires_grad_()
                else:
                    predicted_tokens = predicted_tokens.detach().requires_grad_()
                    predicted_tokens = predicted_tokens.reshape(batch_size, seq_length, self.vocab_size)
                
                # --- 5b: Add Langevin dynamics noise for exploration ---
                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = (
                        torch.randn_like(predicted_tokens.detach()) * 
                        torch.clamp(self.langevin_dynamics_noise_std, min=1e-6)
                    )
                    predicted_tokens = predicted_tokens + ld_noise
                
                # --- 5c: Convert logits to embeddings ---
                # Option 1: Normalize to probability distribution first
                if self.hparams.normalize_initial_condition:
                    predicted_tokens = self.softmax(predicted_tokens)
                    # Probability distribution @ embedding matrix
                    predicted_embeddings = torch.matmul(
                        predicted_tokens,
                        self.embeddings.weight
                    )  # (B, S, D)
                # Option 2: Direct projection from logits
                else:
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens)  # (B, S, D)
                
                # --- 5d: Concatenate real and predicted embeddings ---
                # Transformer sees both input tokens AND current predictions
                all_embeddings = torch.cat(
                    (real_embeddings_input, predicted_embeddings),
                    dim=1
                )  # (B, 2*S, D)
                
                # --- 5e: Compute energy (how bad are the predictions?) ---
                # Energy = large when predictions are wrong, small when correct
                energy_preds = self.transformer(all_embeddings, mcmc_step=mcmc_step)  # (B, 2*S, 1)
                energy_preds = energy_preds.reshape(-1, 1)  # (B*S, 1)
                predicted_energies.append(energy_preds)
                
                # --- 5f: Compute gradient of energy w.r.t. predictions ---
                # Gradient points towards HIGHER energy (worse predictions)
                # We'll move OPPOSITE to this (towards lower energy)
                predicted_tokens_grad = torch.autograd.grad(
                    [energy_preds.sum()],
                    [predicted_tokens],
                    create_graph=learning
                )[0]  # (B, S, V)
                
                if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
                    raise ValueError("NaN/Inf gradients in MCMC step!")
                
                # --- 5g: Update predictions: gradient descent on energy ---
                # Move opposite to gradient: towards lower energy = better predictions
                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad
                
                # --- 5h: Prepare output for this step ---
                if return_raw_logits:
                    predicted_tokens_for_loss = predicted_tokens  # (B, S, V)
                else:
                    # Convert to log probabilities for loss computation
                    predicted_tokens_for_loss = self.log_softmax(predicted_tokens)
                    predicted_tokens_for_loss = predicted_tokens_for_loss.reshape(-1, self.vocab_size)
                
                predicted_distributions.append(predicted_tokens_for_loss)
        
        return predicted_distributions, predicted_energies
    
    # =========================================================================
    # LOSS COMPUTATION
    # =========================================================================
    
    def forward_loss_wrapper(self, batch, phase="train"):
        """
        Compute loss for MCMC music generation.
        
        Supervises the model at refinement steps, encouraging
        gradual improvement of predictions through MCMC iterations.
        
        Supports:
        - Label smoothing: soften supervision early MCMC steps
        - Truncated loss: only supervise final MCMC step
        
        Args:
            batch: Dict with 'input_ids' (MIDI tokens)
            phase: "train", "valid", or "test"
        
        Returns:
            Dict with 'loss' and various metrics
        """
        
        no_randomness = False if phase == "train" else True
        
        # ===== Get input and target tokens =====
        if not no_randomness and self.mcmc_replay_buffer:
            all_tokens = batch['input_ids'].squeeze(dim=1)
            input_ids, replay_buffer_logits, next_token_indices = (
                self.replay_buffer.get_batch(all_tokens)
            )
            predicted_distributions, predicted_energies = self(
                input_ids,
                return_raw_logits=True,
                replay_buffer_logits=replay_buffer_logits,
                no_randomness=no_randomness
            )
            self.replay_buffer.update(
                all_tokens.detach(),
                predicted_distributions[-1].detach()
            )
        else:
            input_ids = batch['input_ids'].squeeze(dim=1)[:, :-1]  # All but last token
            predicted_distributions, predicted_energies = self(
                input_ids,
                return_raw_logits=True,
                no_randomness=no_randomness
            )
            next_token_indices = batch['input_ids'].squeeze(dim=1)[:, 1:]  # Ground truth
        
        next_token_indices = next_token_indices.reshape(-1)  # (B*S,)
        
        # ===== Compute loss (with optional label smoothing) =====
        reconstruction_loss = 0
        total_mcmc_steps = len(predicted_energies)
        
        initial_loss = None
        final_loss = None
        initial_energy = None
        final_energy = None
        
        for mcmc_step, (predicted_dist, energy) in enumerate(
            zip(predicted_distributions, predicted_energies)):
            
            # Reshape logits for loss computation
            if isinstance(predicted_dist, torch.Tensor) and predicted_dist.dim() > 2:
                predicted_dist = predicted_dist.reshape(-1, self.vocab_size)
            
            # ===== Apply label smoothing if enabled =====
            # Progressively harden supervision from first to last MCMC step
            if hasattr(self.hparams, 'soften_target_prob_dist') and self.hparams.soften_target_prob_dist != 0.0:
                if total_mcmc_steps <= 1:
                    label_smoothing = 0.0
                else:
                    # Early steps: looser (higher label_smoothing)
                    # Later steps: tighter (lower label_smoothing)
                    label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * self.hparams.soften_target_prob_dist
                
                ce_loss = F.cross_entropy(
                    predicted_dist,
                    next_token_indices,
                    ignore_index=self.pad_token_id,
                    label_smoothing=label_smoothing
                )
            else:
                # Standard cross-entropy without label smoothing
                ce_loss = F.cross_entropy(
                    predicted_dist,
                    next_token_indices,
                    ignore_index=self.pad_token_id
                )
            
            # ===== Accumulate loss (with truncate_mcmc option) =====
            # If truncate_mcmc: only use final step loss
            # Otherwise: accumulate all steps and average
            if hasattr(self.hparams, 'truncate_mcmc') and self.hparams.truncate_mcmc:
                if mcmc_step == (total_mcmc_steps - 1):
                    reconstruction_loss = ce_loss
                    final_loss = ce_loss.detach()
                    final_energy = energy.squeeze().mean().detach()
            else:
                reconstruction_loss += ce_loss
                if mcmc_step == (total_mcmc_steps - 1):
                    final_loss = ce_loss.detach()
                    final_energy = energy.squeeze().mean().detach()
            
            # Track initial step
            if mcmc_step == 0:
                initial_loss = ce_loss.detach()
                initial_energy = energy.squeeze().mean().detach()
        
        # Average loss over MCMC steps (unless truncate_mcmc)
        if not (hasattr(self.hparams, 'truncate_mcmc') and self.hparams.truncate_mcmc):
            reconstruction_loss = reconstruction_loss / total_mcmc_steps
        
        # Compute energy improvement (lower is better)
        energy_improvement = initial_energy - final_energy if (
            initial_energy is not None and final_energy is not None
        ) else torch.tensor(0.0)
        
        # ===== Compute perplexity =====
        perplexity = torch.exp(final_loss) if final_loss is not None else torch.tensor(0.0)
        
        # ===== Return metrics =====
        log_dict = {
            'loss': reconstruction_loss,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'energy_improvement': energy_improvement,
            'perplexity': perplexity,
        }
        
        return log_dict
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _corrupt_embeddings(self, embeddings):
        """
        Initialize MCMC with corrupted/noisy embeddings.
        
        For music: start from random noise and refine towards real music.
        This gives the model freedom to explore the distribution.
        """
        batch_size = embeddings.shape[0]
        seq_length = embeddings.shape[1]
        
        if self.hparams.denoising_initial_condition == "random_noise":
            predicted_tokens = (
                torch.randn(
                    size=(batch_size, seq_length, self.vocab_size),
                    device=embeddings.device
                ) * self.hparams.gaussian_random_noise_scaling
            )
        elif self.hparams.denoising_initial_condition == "zeros":
            predicted_tokens = torch.zeros(
                size=(batch_size, seq_length, self.vocab_size),
                device=embeddings.device
            )
        else:
            raise NotImplementedError(
                f"denoising_initial_condition={self.hparams.denoising_initial_condition} "
                f"not supported for music yet"
            )
        
        return predicted_tokens
    
    def warm_up_finished(self):
        """Called after LR warmup finishes."""
        if self.hparams.langevin_dynamics_noise_learnable:
            self.langevin_dynamics_noise_std.requires_grad = True
        self.finished_warming_up = True