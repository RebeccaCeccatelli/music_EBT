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
        # EBT attention splits the sequence into real/predicted halves using
        # original_seqlen = (full_seqlen+1)//2, then indexes rotary embeddings as
        # freqs_cis[2:original_seqlen+1]. This requires original_seqlen >= 2,
        # i.e. the input must have at least 2 tokens. Pad short inputs with the
        # pad token so the slice is never empty.
        MIN_EBT_INPUT = 2
        if input_tokens.shape[1] < MIN_EBT_INPUT:
            pad = torch.zeros(
                input_tokens.shape[0],
                MIN_EBT_INPUT - input_tokens.shape[1],
                dtype=input_tokens.dtype,
                device=input_tokens.device,
            )
            input_tokens = torch.cat([input_tokens, pad], dim=1)

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
    elif hparams.model_name == "baseline_llama_transformer":
        # Baseline Llama transformer - does NOT use start_pos parameter
        logits = model.forward(input_tokens, learning=False, return_raw_logits=True)
    else:
        # Default: assume standard transformer interface
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
    elif hparams.tokenizer_type.lower().startswith("anticipation"):
        return generate_anticipation(model, batch, hparams)
    else:
        raise ValueError(
            f"Unknown tokenizer type: {hparams.tokenizer_type}. "
            f"Supported types: 'REMI', 'anticipation*'"
        )


def generate_remi(model, batch, hparams):
    """
    Generate music with REMI tokenization using a sliding context window.

    Each generation step feeds the last context_length tokens to the model,
    so generation length is no longer bounded by context_length. A prompt
    longer than context_length is handled by seeding the window with its
    last context_length tokens (most recent musical context).

    Args:
        model: The music generation model
        batch: Input batch containing 'input_ids' (conditioning tokens)
        hparams: Hyperparameters including:
            - infer_max_gen_len: Exact number of new tokens to generate
            - infer_temp: Temperature for sampling (0 = greedy)
            - infer_topp: Top-p threshold for nucleus sampling
            - infer_logprobs: Whether to track log probabilities
            - infer_echo: Whether to include prompt in full_sequences output
            - context_length: Sliding window size

    Returns:
        dict with keys:
            - 'prompt_tokens': List of prompt token lists
            - 'generation_tokens': List of generated token lists (no prompt)
            - 'generation_logprobs': List of log-prob lists (if infer_logprobs)
            - 'full_sequences': prompt + generated (if echo), else generated only
    """
    ids = batch['input_ids']
    max_gen_len = hparams.infer_max_gen_len
    temperature = hparams.infer_temp
    top_p = hparams.infer_topp
    logprobs = hparams.infer_logprobs
    echo = hparams.infer_echo
    context_length = hparams.context_length

    pad_token_id = model.pad_token_id if hasattr(model, 'pad_token_id') else 0
    device = getattr(hparams, 'device', 'cuda')

    params = model.transformer.params if hasattr(model, 'transformer') and hasattr(model.transformer, 'params') else None

    # Extract prompt tokens, stripping padding
    prompt_tokens = []
    for row_ids in ids:
        row_ids = row_ids.squeeze() if row_ids.dim() > 1 else row_ids
        prompt = row_ids.tolist()
        while prompt and prompt[-1] == pad_token_id:
            prompt.pop()
        prompt_tokens.append(prompt)

    bsz = len(prompt_tokens)
    if params is not None:
        assert bsz <= params.max_batch_size, f"Batch size {bsz} exceeds model max {params.max_batch_size}"

    out_tokens = []
    out_logprobs = []

    for batch_idx in range(bsz):
        prompt = prompt_tokens[batch_idx]

        # Seed the sliding window with the last context_length tokens of the prompt.
        # If the prompt is shorter, use it as-is; if longer, this keeps the most
        # recent musical context at the start of generation.
        context = list(prompt[-context_length:]) if len(prompt) > context_length else list(prompt)

        generated = []
        gen_logprobs = []

        with torch.no_grad():
            for _ in range(max_gen_len):
                window = context[-context_length:]
                input_tensor = torch.tensor(
                    window, dtype=torch.long, device=device
                ).unsqueeze(0)

                logits = call_model_forward_decode(hparams, model, input_tensor, 0, 1)

                if logits.dim() == 3:
                    last_logits = logits[0, -1, :]
                else:
                    last_logits = logits[0]

                if temperature > 0:
                    probs = torch.softmax(last_logits / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(last_logits, dim=-1)

                next_token_val = int(next_token.reshape(-1)[0].item())

                if logprobs:
                    lp = float(F.log_softmax(last_logits, dim=-1)[next_token_val].item())
                    gen_logprobs.append(lp)

                context.append(next_token_val)
                generated.append(next_token_val)

        out_tokens.append(generated)
        out_logprobs.append(gen_logprobs)

    result = {
        'prompt_tokens': prompt_tokens,
        'generation_tokens': out_tokens,
        'full_sequences': [p + g for p, g in zip(prompt_tokens, out_tokens)] if echo else out_tokens,
    }

    if logprobs:
        result['generation_logprobs'] = out_logprobs

    return result


def generate_anticipation(model, batch, hparams) -> Dict:
    """
    Generate music with Anticipation tokenization (time-aware structured tokens).

    infer_max_gen_len is interpreted as a token count (same units as REMI).
    Dividing by 3 gives the number of event triplets to generate.

    Args:
        model: The music generation model
        batch: Input batch containing 'input_ids'
        hparams: Hyperparameters including:
            - infer_max_gen_len: Total tokens to generate (divided by 3 → events)
            - infer_temp: Temperature for sampling (0 = greedy)
            - infer_topp: Top-p threshold for nucleus sampling
            - anticipation_lookback: Lookback window size in tokens (default 1020)

    Returns:
        dict with 'prompt_tokens', 'generation_tokens', and optionally
        'generation_logprobs'.
    """
    try:
        from anticipation import ops
        from anticipation.config import TIME_RESOLUTION
        from anticipation.vocab_selector import (
            AUTOREGRESS, ANTICIPATE, CONTROL_OFFSET, SPECIAL_OFFSET,
            TIME_OFFSET,
        )
    except ImportError:
        raise ImportError(
            "Anticipation tokenization requires the anticipation library. "
            "Make sure anticipation is installed as a submodule in "
            "data/mus/symbolic/tokenization/anticipation/ and that "
            "load_tokenizer() was called before generate_anticipation()."
        )

    ids = batch['input_ids']
    max_gen_len = hparams.infer_max_gen_len   # total tokens; /3 → events
    temperature = hparams.infer_temp
    top_p = hparams.infer_topp
    logprobs = hparams.infer_logprobs
    lookback_tokens = getattr(hparams, 'anticipation_lookback', 1020)
    # EBT concatenates real + predicted embeddings before the transformer, doubling
    # the sequence length to 2*S. The rotary embeddings are precomputed up to
    # max_seq_len = context_length + 1. So we need 2*(1 + history + 2) ≤ max_seq_len,
    # i.e. history ≤ context_length // 2 - 2, then trim to a triplet boundary.
    if getattr(hparams, 'model_name', '') == 'ebt':
        ebt_max_seq = getattr(hparams, 'context_length', 1024) + 1
        ebt_max_history = (ebt_max_seq // 2) - 3   # 509 for context_length=1024
        ebt_max_history -= ebt_max_history % 3      # align to triplet boundary → 507
        lookback_tokens = min(lookback_tokens, ebt_max_history)
    device = getattr(hparams, 'device', 'cuda')

    pad_token_id = model.pad_token_id if hasattr(model, 'pad_token_id') else CONTROL_OFFSET - 1

    # Extract prompt tokens, stripping padding
    prompt_tokens = []
    for row_ids in ids:
        row_ids = row_ids.squeeze() if row_ids.dim() > 1 else row_ids
        prompt = row_ids.tolist()
        while prompt and prompt[-1] == pad_token_id:
            prompt.pop()
        prompt_tokens.append(prompt)

    bsz = len(prompt_tokens)
    generated_all = []
    full_event_sequences = []
    generated_logprobs_all = []

    for batch_idx in range(bsz):
        prompt = prompt_tokens[batch_idx]

        # Strip mode token (first token) if present
        if len(prompt) > 0 and prompt[0] in (AUTOREGRESS, ANTICIPATE):
            event_tokens = prompt[1:]
        else:
            event_tokens = list(prompt)

        # Trim to triplet boundary so stride-3 indexing is always aligned
        remainder = len(event_tokens) % 3
        if remainder != 0:
            event_tokens = event_tokens[:-remainder]

        # For ANTICIPATE-mode prompts, controls are interleaved with events.
        # Strip them so tokens_list contains only event triplets and current_time
        # reflects the actual last *event* time, not the max control lookahead time.
        # Always generate in AUTOREGRESS mode — the model's training objective is
        # left-to-right event prediction, regardless of the prompt's mode token.
        events_only, _ = ops.split(event_tokens) if len(event_tokens) > 0 else ([], [])
        mode = [AUTOREGRESS]

        # tokens_list holds absolute-time event vocab IDs (no mode token, no controls)
        tokens_list = list(events_only)

        # current_time: absolute ticks of the last event in the prompt
        current_time = ops.max_time(tokens_list, seconds=False) if len(tokens_list) > 0 else 0

        # Number of event triplets to generate
        max_events = max(1, max_gen_len // 3)

        generated = []
        gen_logprobs = []

        with torch.no_grad():
            for _step in range(max_events):
                # Align lookback window to a triplet boundary
                raw_start = max(len(tokens_list) - lookback_tokens, 0)
                lookback_start = raw_start - (raw_start % 3)
                history = tokens_list[lookback_start:].copy()

                time_offset = ops.min_time(history, seconds=False) if len(history) > 0 else 0

                # Relativize all TIME positions (0, 3, 6, ...) in the history
                history[::3] = [t - time_offset for t in history[::3]]

                new_token_triplet = []
                for triplet_idx in range(3):
                    input_seq = mode + history + new_token_triplet
                    input_tensor = torch.tensor(
                        input_seq, dtype=torch.long, device=device
                    ).unsqueeze(0)

                    logits = call_model_forward_decode(hparams, model, input_tensor, 0, 1)

                    if logits.dim() == 3:
                        last_logits = logits[0, -1, :]
                    else:
                        last_logits = logits[0]

                    # Mask invalid tokens; relative current time = current_time - time_offset
                    rel_current = max(0, current_time - time_offset)
                    last_logits = mask_invalid_anticipation_tokens(
                        last_logits.clone(), triplet_idx, rel_current, tokens_list
                    )

                    if temperature > 0:
                        probs = torch.softmax(last_logits / temperature, dim=-1)
                        next_token = sample_top_p(probs, top_p)
                    else:
                        next_token = torch.argmax(last_logits, dim=-1)

                    next_token_val = int(next_token.item())
                    new_token_triplet.append(next_token_val)

                    if logprobs:
                        lp = float(F.log_softmax(last_logits, dim=-1)[next_token_val].item())
                        gen_logprobs.append(lp)

                # Restore absolute time for the TIME token
                new_token_triplet[0] += time_offset

                current_time = new_token_triplet[0] - TIME_OFFSET
                tokens_list.extend(new_token_triplet)
                generated.extend(new_token_triplet)

        generated_all.append(generated)
        full_event_sequences.append(list(tokens_list))
        generated_logprobs_all.append(gen_logprobs)

    result = {
        'prompt_tokens': prompt_tokens,
        'generation_tokens': generated_all,
        # events_only extracted from prompt + newly generated events, all triplet-aligned.
        # Use this instead of prompt + generated when decoding Anticipation sequences to
        # avoid the triplet-boundary mismatch that occurs when the raw prompt length is
        # not a multiple of 3 (after stripping the mode token).
        'full_event_sequences': full_event_sequences,
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
            MAX_DUR, MAX_TIME, MAX_INSTR, MAX_PITCH
        )
        from anticipation.vocab_selector import (
            TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, CONTROL_OFFSET, SPECIAL_OFFSET
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
