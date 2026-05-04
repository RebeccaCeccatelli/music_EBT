"""
Music generation for all model types (EBT, Baseline Llama, Baseline HF GPT2)

Unified generation interface supporting multiple tokenization schemes:
- REMI tokenization (flat autoregressive token stream)
- Anticipation tokenization (time-aware structured tokens with controls)

And multiple model architectures:
- Energy-Based Transformers (EBT) with MCMC refinement
- Baseline Llama-based transformers
- Baseline HF GPT2 transformers

Code adapted from Llama2 generation.py, HuggingFace transformers, and Anticipation library.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict
import math


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def call_model_forward_decode(hparams, model, input_tokens, start_pos, bsz):
    """
    Forward pass for music token generation.
    
    Handles both custom transformers (EBT, Llama) and HuggingFace models.
    
    Args:
        hparams: Hyperparameters containing model_name and inference settings
        model: The music generation model
        input_tokens: Input token sequences, shape (bsz, seq_len)
        start_pos: Starting position for KV caching (currently unused, set to 0)
        bsz: Batch size
    
    Returns:
        logits: Raw logits for next token prediction, shape (bsz, seq_len, vocab_size) or (bsz, vocab_size)
    """
    if hparams.model_name == "ebt":
        # Energy-Based Transformer with MCMC refinement
        if hparams.infer_ebt_advanced:
            ebt_outputs = model.ebt_advanced_inference(input_tokens, start_pos=0, learning=False)
            logits = ebt_outputs[0]  # Final predicted logits
        else:
            ebt_outputs = model.forward(input_tokens, start_pos=0, learning=False, return_raw_logits=True)
            logits = ebt_outputs[0][-1]  # Use final MCMC step logits
    elif hparams.model_name == "baseline_hf_gpt2_transformer":
        # HuggingFace GPT2 model
        attention_mask = (input_tokens != model.pad_token_id).long()
        outputs = model.model(
            input_ids=input_tokens,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits  # (B, S, V)
    else:
        # Baseline Llama transformer or other models
        logits = model.forward(input_tokens, start_pos=0, learning=False, return_raw_logits=True)
    
    return logits


def generate_music(model, batch, hparams):
    """
    Main dispatcher for music generation based on tokenizer type.
    
    Routes to appropriate generation function based on tokenization scheme:
    - REMI: Simple autoregressive generation
    - Anticipation: Time-aware structured generation with anticipatory constraints
    
    Args:
        model: The music generation model
        batch: Input batch with 'input_ids' (conditioning tokens)
        hparams: Hyperparameters including tokenizer_type
        
    Returns:
        dict: Generated sequences (format depends on tokenizer type)
    """
    if hparams.tokenizer_type == "REMI":
        return generate_remi(model, batch, hparams)
    elif hparams.tokenizer_type == "anticipation":
        return generate_anticipation(model, batch, hparams)
    else:
        raise ValueError(
            f"Unknown tokenizer type: {hparams.tokenizer_type}. "
            f"Supported types: 'REMI', 'anticipation'"
        )


def generate_remi(model, batch, hparams):
    """
    Generate music with REMI tokenization (flat autoregressive token stream).
    
    Simple autoregressive generation where each token is predicted independently.
    Works for any model architecture (EBT, Baseline Llama, Baseline HF GPT2).
    
    Features:
    - Temperature-controlled sampling
    - Top-p (nucleus) sampling
    - Log probability tracking
    - Supports unconditional generation (empty prompt)
    
    Args:
        model: The music generation model
        batch: Input batch containing 'input_ids' (conditioning tokens)
        hparams: Hyperparameters including:
            - infer_max_gen_len: Maximum tokens to generate
            - infer_temp: Temperature for sampling (0 = greedy)
            - infer_topp: Top-p threshold for nucleus sampling
            - infer_logprobs: Whether to track log probabilities
            - infer_echo: Whether to return prompt tokens in output
            - context_length: Maximum sequence length
    
    Returns:
        dict: Generated sequences with keys:
            - 'generation_tokens': List of generated token sequences (without prompt)
            - 'generation_logprobs': List of log probabilities (if tracking)
            - 'full_sequences': Full sequences including prompt (if echo=True)
    """
    # Extract configuration
    ids = batch['input_ids']
    max_gen_len = hparams.infer_max_gen_len
    temperature = hparams.infer_temp
    top_p = hparams.infer_topp
    logprobs = hparams.infer_logprobs
    echo = hparams.infer_echo
    
    # Get padding token ID from the model
    pad_token_id = model.pad_token_id if hasattr(model, 'pad_token_id') else 0
    
    # Extract prompt tokens (non-padded portion)
    prompt_tokens = []
    for row_ids in ids:
        row_ids = row_ids.squeeze() if row_ids.dim() > 1 else row_ids
        seq_len = len(row_ids)
        prompt_tokens.append(row_ids[:seq_len].tolist())
    
    # Get model parameters (for custom transformers)
    params = model.transformer.params if hasattr(model, 'transformer') and hasattr(model.transformer, 'params') else None
    bsz = len(prompt_tokens)
    
    if params is not None:
        assert bsz <= params.max_batch_size, f"Batch size {bsz} exceeds model max {params.max_batch_size}"
    
    # Determine sequence lengths
    min_prompt_len = min(len(t) for t in prompt_tokens) if prompt_tokens else 0
    max_prompt_len = max(len(t) for t in prompt_tokens) if prompt_tokens else 0
    
    # Ensure prompts fit in context
    assert max_prompt_len <= hparams.context_length, \
        f"Prompt length {max_prompt_len} exceeds context length {hparams.context_length}"
    
    total_len = min(hparams.context_length, max_gen_len + max_prompt_len)
    
    # Initialize token tensor
    tokens = torch.full(
        (bsz, total_len),
        pad_token_id,
        dtype=torch.long,
        device="cuda"
    )
    
    # Populate prompt tokens
    for k, t in enumerate(prompt_tokens):
        tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda").clone().detach()
    
    # Initialize log probability tracking if requested
    if logprobs:
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
    
    # Create input mask to track which tokens are part of prompt vs generated
    input_text_mask = tokens != pad_token_id
    
    # Autoregressive generation
    with torch.no_grad():
        # If min prompt length equals total length, compute logits once and we're done
        if min_prompt_len == total_len:
            logits = call_model_forward_decode(hparams, model, tokens, 0, bsz)
            if logprobs:
                # Handle both (B, S, V) and (B, V) shaped logits
                if logits.dim() == 3:
                    token_logprobs = -F.cross_entropy(
                        input=logits.transpose(1, 2),
                        target=tokens,
                        reduction="none",
                        ignore_index=pad_token_id,
                    )
        
        # Generate tokens one at a time
        for cur_pos in range(min_prompt_len, total_len):
            # Get model prediction
            input_tokens = tokens[:, :cur_pos]
            logits = call_model_forward_decode(hparams, model, input_tokens, 0, bsz)
            
            # Extract last token logits
            if logits.dim() == 3:
                last_logits = logits[:, -1, :]  # (B, V)
            else:
                last_logits = logits  # Already (B, V)
            
            # Sample next token
            if temperature > 0:
                # Temperature-scaled sampling
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                # Greedy decoding
                next_token = torch.argmax(last_logits, dim=-1)
            
            next_token = next_token.reshape(-1)
            
            # Only use generated token if we haven't reached the end of prompt
            next_token = torch.where(
                input_text_mask[:, cur_pos],
                tokens[:, cur_pos],
                next_token
            )
            
            tokens[:, cur_pos] = next_token
            
            # Track log probabilities
            if logprobs:
                token_logprobs[:, cur_pos] = -F.cross_entropy(
                    input=last_logits.unsqueeze(1).transpose(1, 2),
                    target=tokens[:, cur_pos:cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_token_id,
                ).squeeze()
    
    # Convert log probabilities to list format
    if logprobs:
        token_logprobs = token_logprobs.tolist()
    
    # Extract generation results
    out_tokens = []
    out_logprobs = []
    
    for i, toks in enumerate(tokens.tolist()):
        # Find where prompt ends (first pad token or use original prompt length)
        prompt_len = len(prompt_tokens[i])
        
        # Generated tokens (exclude prompt)
        generated = toks[prompt_len:]
        # Remove trailing padding
        generated = [t for t in generated if t != pad_token_id]
        
        out_tokens.append(generated)
        
        if logprobs:
            # Log probs for generated portion
            gen_logprobs = token_logprobs[i][prompt_len:prompt_len + len(generated)]
            out_logprobs.append(gen_logprobs)
    
    # Prepare output
    result = {
        'generation_tokens': out_tokens,
        'full_sequences': [toks for toks in tokens.tolist()] if echo else out_tokens,
    }
    
    if logprobs:
        result['generation_logprobs'] = out_logprobs
    
    return result


def generate_anticipation(model, batch, hparams) -> Dict:
    """
    Generate music with Anticipation tokenization (time-aware structured tokens).
    
    Tokens are structured as (time, duration, note) triplets with special handling for:
    - Time constraints (don't generate events in the past)
    - Anticipatory controls (melody constraints, instrument limits)
    - Anticipation infilling mode
    
    Features:
    - Time-aware token generation
    - Supports control inputs (e.g., melody constraints)
    - Top-p nucleus sampling with constraint-aware masking
    - Anticipatory and autoregressive modes
    
    Args:
        model: The music generation model
        batch: Input batch containing 'input_ids' and optional 'controls'
        hparams: Hyperparameters including:
            - infer_max_gen_len: Maximum seconds to generate
            - infer_temp: Temperature for sampling (0 = greedy)
            - infer_topp: Top-p threshold for nucleus sampling
            - context_length: Maximum context in seconds
            - anticipation_delta: Time delta for anticipation (seconds)
            - anticipation_lookback: Lookback window for history (tokens)
    
    Returns:
        dict: Generated sequences with keys:
            - 'generation_tokens': List of generated token sequences
            - 'generation_logprobs': List of log probabilities (if tracking)
    
    Note:
        This implementation adapts the Anticipation library's generate() function
        for use with arbitrary model architectures. Token structure:
        - Token triplets: [TIME, DURATION, NOTE, TIME, DURATION, NOTE, ...]
        - Special tokens used for mode selection and controls
    """
    try:
        from anticipation import ops
        from anticipation.config import (
            AUTOREGRESS, ANTICIPATE, CONTROL_OFFSET, SPECIAL_OFFSET,
            TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, ATIME_OFFSET, ADUR_OFFSET,
            ANOTE_OFFSET, MAX_DUR, MAX_TIME, MAX_INSTR, MAX_PITCH,
            TIME_RESOLUTION, DELTA
        )
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            "Anticipation tokenization requires the anticipation library. "
            "Make sure anticipation is installed as a submodule."
        )
    
    # Extract configuration
    ids = batch['input_ids']
    controls = batch.get('controls', None)
    max_gen_len = hparams.infer_max_gen_len
    temperature = hparams.infer_temp
    top_p = hparams.infer_topp
    logprobs = hparams.infer_logprobs
    
    # Convert seconds to time units
    anticipation_delta = getattr(hparams, 'anticipation_delta', DELTA * TIME_RESOLUTION)
    lookback_tokens = getattr(hparams, 'anticipation_lookback', 1020)
    
    # Extract prompt and controls
    prompt_tokens = []
    for row_ids in ids:
        row_ids = row_ids.squeeze() if row_ids.dim() > 1 else row_ids
        prompt_tokens.append(row_ids.tolist())
    
    if controls is not None:
        control_tokens = []
        for row_ids in controls:
            row_ids = row_ids.squeeze() if row_ids.dim() > 1 else row_ids
            control_tokens.append(row_ids.tolist())
    else:
        control_tokens = [[] for _ in prompt_tokens]
    
    bsz = len(prompt_tokens)
    generated_all = []
    generated_logprobs_all = []
    
    # Generate for each item in batch
    for batch_idx in range(bsz):
        prompt = prompt_tokens[batch_idx]
        controls = control_tokens[batch_idx] if control_tokens else []
        
        # Determine start and end times
        start_time = int(TIME_RESOLUTION * 0)  # Start from beginning
        end_time = int(TIME_RESOLUTION * max_gen_len)  # Generate max_gen_len seconds
        
        # Prepare prompt and future events
        prompt_padded = ops.pad(
            ops.clip(prompt, 0, start_time, clip_duration=False, seconds=False),
            start_time
        )
        
        future = ops.clip(
            prompt, start_time + 1,
            ops.max_time(prompt, seconds=False) if len(prompt) > 0 else 0,
            clip_duration=False, seconds=False
        )
        
        controls_clipped = ops.clip(
            controls, DELTA,
            ops.max_time(controls, seconds=False) if len(controls) > 0 else 0,
            clip_duration=False, seconds=False
        )
        
        # Determine mode: ANTICIPATE if controls/future, else AUTOREGRESS
        mode = [ANTICIPATE] if (len(controls_clipped) > 0 or len(future) > 0) else [AUTOREGRESS]
        
        # Interleave controls with future events
        tokens_list = prompt_padded.copy()
        control_sorted = ops.sort(controls_clipped + [CONTROL_OFFSET + t for t in future])
        
        if len(control_sorted) > 0:
            tokens_list_new, _ = ops.anticipate(tokens_list, control_sorted)
            tokens_list = tokens_list_new
        
        # Initialize tracking
        current_time = ops.max_time(prompt_padded, seconds=False) if len(prompt_padded) > 0 else 0
        generated = []
        gen_logprobs = []
        
        # Anticipation generation loop
        try:
            with torch.no_grad():
                for step in range(end_time - start_time):
                    # Truncate history to lookback window
                    lookback_start = max(len(tokens_list) - lookback_tokens, 0)
                    history = tokens_list[lookback_start:].copy()
                    time_offset = ops.min_time(history, seconds=False) if len(history) > 0 else 0
                    
                    # Relativize times
                    history[::3] = [t - time_offset for t in history[::3]]
                    
                    # Generate 3 tokens (time, duration, note)
                    new_token_triplet = []
                    for triplet_idx in range(3):
                        # Prepare input
                        input_seq = mode + history + new_token_triplet
                        input_tensor = torch.tensor(
                            input_seq,
                            dtype=torch.long,
                            device="cuda" if torch.cuda.is_available() else "cpu"
                        ).unsqueeze(0)
                        
                        # Get logits from model
                        logits = call_model_forward_decode(hparams, model, input_tensor, 0, 1)
                        
                        # Extract last position logits
                        if logits.dim() == 3:
                            last_logits = logits[0, -1, :]
                        else:
                            last_logits = logits[0]
                        
                        # Apply constraints based on token position in triplet
                        last_logits = mask_invalid_anticipation_tokens(
                            last_logits, triplet_idx, current_time - time_offset,
                            tokens_list
                        )
                        
                        # Sample
                        if temperature > 0:
                            probs = torch.softmax(last_logits / temperature, dim=-1)
                            next_token = sample_top_p(probs, top_p)
                        else:
                            next_token = torch.argmax(last_logits, dim=-1)
                        
                        next_token_val = int(next_token.item())
                        new_token_triplet.append(next_token_val)
                        
                        # Track logprobs
                        if logprobs:
                            logprob = F.log_softmax(last_logits, dim=-1)[next_token]
                            gen_logprobs.append(float(logprob.item()))
                    
                    # Adjust time back to global frame
                    new_token_triplet[0] += time_offset
                    
                    # Check if we've exceeded end time
                    new_time = new_token_triplet[0] - TIME_OFFSET
                    if new_time >= end_time:
                        break
                    
                    tokens_list.extend(new_token_triplet)
                    generated.extend(new_token_triplet)
                    current_time = new_time
        
        except Exception as e:
            print(f"Warning: Generation interrupted at step {step}: {e}")
        
        generated_all.append(generated)
        generated_logprobs_all.append(gen_logprobs)
    
    # Format output
    result = {
        'generation_tokens': generated_all,
    }
    
    if logprobs:
        result['generation_logprobs'] = generated_logprobs_all
    
    return result


def mask_invalid_anticipation_tokens(logits: torch.Tensor, triplet_idx: int,
                                      current_time: int, full_history: List[int]) -> torch.Tensor:
    """
    Mask invalid token logits based on anticipation constraints.
    
    In anticipation tokenization, tokens are structured as (time, duration, note) triplets.
    This function applies position-aware masking to prevent invalid tokens at each position.
    
    Args:
        logits: Logits tensor of shape (vocab_size,)
        triplet_idx: Index within triplet (0=time, 1=duration, 2=note)
        current_time: Current generation time
        full_history: Full token history for constraint checking
        
    Returns:
        logits: Masked logits tensor
    """
    try:
        from anticipation.config import (
            TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, CONTROL_OFFSET, SPECIAL_OFFSET,
            MAX_DUR, MAX_TIME, MAX_INSTR, MAX_PITCH
        )
        from anticipation import ops
    except ImportError:
        return logits
    
    # Don't generate control or special tokens in main generation
    logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf')
    logits[SPECIAL_OFFSET:] = -float('inf')
    
    # Position-specific masking
    if triplet_idx == 0:
        # TIME position: don't generate duration or note tokens
        logits[DUR_OFFSET:DUR_OFFSET + MAX_DUR] = -float('inf')
        logits[NOTE_OFFSET:CONTROL_OFFSET] = -float('inf')
        # Don't generate events in the past
        logits[TIME_OFFSET:TIME_OFFSET + current_time] = -float('inf')
    elif triplet_idx == 1:
        # DURATION position: don't generate time or note tokens
        logits[TIME_OFFSET:TIME_OFFSET + MAX_TIME] = -float('inf')
        logits[NOTE_OFFSET:CONTROL_OFFSET] = -float('inf')
    elif triplet_idx == 2:
        # NOTE position: don't generate time or duration tokens
        logits[TIME_OFFSET:TIME_OFFSET + MAX_TIME] = -float('inf')
        logits[DUR_OFFSET:DUR_OFFSET + MAX_DUR] = -float('inf')
        # Limit instruments to 15 max (16 - 1 for reserved track)
        instrs = ops.get_instruments(full_history)
        if len(instrs) >= 15:
            for instr in range(MAX_INSTR):
                if instr not in instrs:
                    note_start = NOTE_OFFSET + instr * MAX_PITCH
                    note_end = NOTE_OFFSET + (instr + 1) * MAX_PITCH
                    logits[note_start:note_end] = -float('inf')
    
    return logits