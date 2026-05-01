"""
Music generation for all model types (EBT, Baseline Llama, Baseline HF GPT2)

Unified generation interface supporting:
- Energy-Based Transformers (EBT) with MCMC refinement
- Baseline Llama-based transformers
- Baseline HF GPT2 transformers

Code adapted from Llama2 generation.py and HuggingFace transformers.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


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
    Generate symbolic music (MIDI tokens) using autoregressive decoding.
    
    Supports all model types:
    - Energy-Based Transformers (EBT)
    - Baseline Llama transformers
    - Baseline HF GPT2 transformers
    
    Features:
    - Temperature-controlled sampling
    - Top-p (nucleus) sampling
    - Log probability tracking
    
    Args:
        model: The music generation model (EBT, Baseline Llama, or Baseline HF GPT2)
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