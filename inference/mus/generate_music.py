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


def _remi_step_is_relevant(last_token: int, attribute: str) -> bool:
    """
    Whether the upcoming token-generation step can actually move `attribute`,
    given the last committed token — used to gate R³ attribute guidance so it
    only pushes on steps where it has a real lever, instead of firing
    uniformly on every step regardless of token type (which was nudging e.g.
    Pitch-value selection toward a density target that density doesn't even
    measure — a likely source of the coherence damage seen at higher lambda).

    Based on an empirical pass over 40 real validation songs (681k tokens):
    Pitch/PitchDrum -> Velocity and Velocity -> Duration are both 100%
    deterministic transitions in this REMI vocab, so those are exact,
    reliable gates. Duration/Position -> next is NOT deterministic (dominant
    next-category covers only ~27-33% of occurrences) — that's precisely the
    "does another note start here" branch point density guidance should act
    on, and the closest this vocab gets to a density-relevant decision.
    See attribute_control/attributes.py for the REMI id ranges used here.
    """
    from attribute_control.note_density import (
        REMI_PITCH_MIN, REMI_PITCH_MAX, REMI_PITCHDRUM_MIN, REMI_PITCHDRUM_MAX,
    )
    from attribute_control.attributes import (
        REMI_VELOCITY_MIN_ID, REMI_VELOCITY_MAX_ID,
        REMI_DURATION_MIN_ID, REMI_DURATION_MAX_ID,
        REMI_POSITION_MIN_ID, REMI_POSITION_MAX_ID,
        REMI_PROGRAM_MIN_ID, REMI_PROGRAM_MAX_ID,
    )
    is_pitch = (REMI_PITCH_MIN <= last_token <= REMI_PITCH_MAX
                or REMI_PITCHDRUM_MIN <= last_token <= REMI_PITCHDRUM_MAX)
    is_velocity = REMI_VELOCITY_MIN_ID <= last_token <= REMI_VELOCITY_MAX_ID
    is_duration = REMI_DURATION_MIN_ID <= last_token <= REMI_DURATION_MAX_ID
    is_position = REMI_POSITION_MIN_ID <= last_token <= REMI_POSITION_MAX_ID
    is_program = REMI_PROGRAM_MIN_ID <= last_token <= REMI_PROGRAM_MAX_ID
    if attribute == 'velocity':
        return is_pitch
    if attribute == 'duration':
        return is_velocity
    if attribute == 'density':
        return is_duration or is_position
    if attribute == 'pitch_register':
        return is_program
    if attribute == 'polyphony':
        return is_duration or is_position
    if attribute == 'rhythm':
        return is_duration or is_position
    if attribute == 'drum_density':
        return is_program
    if attribute == 'melodic_interval':
        return is_program
    if attribute == 'syncopation':
        return is_duration or is_position
    return True  # unknown attribute: fail open, don't gate


def _anticipation_step_is_relevant(triplet_idx: int, attribute: str) -> bool:
    """Anticipation's equivalent of _remi_step_is_relevant — but simpler to
    get right, since Anticipation's grammar is already a strict, fixed
    (time, duration, note) triplet cycle (see mask_invalid_anticipation_tokens's
    own triplet_idx convention: 0=time, 1=duration, 2=note, note=instr*128+pitch
    — see attribute_control/attributes.py's compute_pitch_register). No
    last-committed-token vocab-range decoding needed: which triplet slot is
    about to be generated is already known exactly from the loop position."""
    if attribute == 'duration':
        return triplet_idx == 1
    if attribute == 'pitch_register':
        return triplet_idx == 2
    return True  # unknown attribute: fail open, don't gate


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


def call_model_forward_decode(hparams, model, input_tokens, start_pos, bsz, attr_energy_fn=None,
                               energy_sink=None):
    """
    Forward pass for music token generation.

    Handles both custom transformers (EBT, Llama) and HuggingFace models.

    Args:
        hparams: Hyperparameters containing model_name and inference settings
        model: The music generation model
        input_tokens: Input token sequences, shape (bsz, seq_len)
        start_pos: Starting position for KV caching (currently unused, set to 0)
        bsz: Batch size
        energy_sink: Optional dict — if given, EBT's own per-MCMC-step energy
            (otherwise discarded here, kept only as ebt_outputs[1]) is written
            to energy_sink['energies'] so a caller can build an energy-vs-step
            trajectory for the demo's generation-diagnostics plot.

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
            ebt_outputs = model.ebt_advanced_inference(input_tokens, start_pos=0, learning=False,
                                                        attr_energy_fn=attr_energy_fn)
            logits = ebt_outputs[0]  # Final predicted logits
        else:
            ebt_outputs = model.forward(input_tokens, start_pos=0, learning=False, return_raw_logits=True,
                                        attr_energy_fn=attr_energy_fn)
            logits = ebt_outputs[0][-1]  # Use final MCMC step logits
            if energy_sink is not None:
                energy_sink['energies'] = ebt_outputs[1]  # predicted_energies, one per MCMC step
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
        if attr_energy_fn is not None:
            # PPLM-style single-step guidance. EBT already differentiates its
            # own energy w.r.t. predicted_tokens as part of its native MCMC
            # refinement loop, so adding the attribute term there is nearly
            # free; Llama has no such loop to piggyback on, so this takes one
            # extra gradient step directly on the raw next-token logits —
            # the closest analogue to the position EBT's own gradient acts
            # on (predicted_tokens[:, -1:, :]) — without inventing any
            # EBT-specific machinery. attr_energy_fn already handles a raw
            # (B, S, V) logits tensor identically to EBT's predicted_tokens
            # (softmaxes internally when values fall outside [0, 1]), so no
            # separate closure is needed here.
            #
            # Cost: this requires a full backward pass through the whole
            # transformer every generation step (no cheaper hook exists,
            # unlike EBT which is already backpropagating for its own
            # refinement) — meaningfully more expensive per step than plain
            # sampling, but tractable at listen-sweep scale on one GPU.
            #
            # There's no native model gradient to normalize against here
            # (unlike ebt_symbolic.py's ebt_norm/attr_norm convention), so
            # lambda_scale is applied directly as a perturbation magnitude in
            # raw logit units — NOT on the same scale as EBT's lambda, and
            # needs its own empirical calibration.
            # A plain learning=False call earlier in this same run (baseline/
            # unguided steps use it) permanently marks the transformer's
            # shared freqs_cis buffer as an inference-mode tensor (that
            # branch wraps it in torch.inference_mode()), which can then
            # never participate in autograd again on this model instance —
            # not even from inside torch.enable_grad(). Re-clone it once it's
            # poisoned so this guided forward pass can actually backprop.
            rotary = getattr(model.transformer, 'freqs_cis', None)
            if rotary is not None and rotary.is_inference():
                model.transformer.freqs_cis = rotary.clone()
            with torch.enable_grad():
                logits = model.forward(input_tokens, learning=True, return_raw_logits=True)
                attr_energy = attr_energy_fn(logits)
                attr_grad = torch.autograd.grad([attr_energy], [logits])[0]
            lam = getattr(attr_energy_fn, 'lambda_scale', 0.0)
            attr_norm = attr_grad.norm().clamp(min=1e-8)
            logits = logits.detach() - lam * attr_grad / attr_norm
        else:
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

    # ── Attribute control (R³) setup ──────────────────────────────────────────
    # Paper: Du et al. 2023, Eq. 6 — classifier guidance on the EBT MCMC state.
    # ∇_x log p(x|y) = ∇_x log p_EBT(x) + λ ∇_x log p(y|x)
    # p(y|x) is a learned attribute regressor (density/velocity/duration/...,
    # see attribute_control/attributes.py) applied to a LOCAL window: the last
    # `attr_hard_window` committed tokens plus the single soft (MCMC)
    # distribution for the next token about to be sampled. Only that one
    # position actually affects the output at each decode step, so the energy
    # — and the resulting gradient — is restricted to it; pooling over the
    # model's whole per-position prediction tensor (as an earlier version of
    # this code did) diluted the guidance across mostly-discarded positions.
    #
    # Multi-attribute composition (R³'s "Reduce" step, Du et al. 2023): pass
    # hparams.attribute_regressor_ckpts (list) + attribute_targets (list),
    # optionally attribute_weights (list, default 1.0 each — a RELATIVE mix
    # between attributes' contributions, applied before the existing
    # gradient-normalize-then-scale step in ebt_symbolic.py). lambda_attribute
    # stays a single overall scalar controlling the combined term's strength
    # vs. EBT's own gradient, same interpretable convention as before. Each
    # attribute keeps its OWN gate (_remi_step_is_relevant) and its own
    # hard_window from its regressor's metadata — at a given step only the
    # attributes whose gate is currently active contribute to the sum, so two
    # attributes that never share a gate (e.g. velocity/duration) never
    # actually need summing; two that do (e.g. density/polyphony, both gated
    # on Duration/Position) genuinely combine at that step.
    #
    # Falls back to the legacy singular hparams (attribute_target/
    # lambda_attribute/attribute_regressor_ckpt, or the density-specific
    # density_target/lambda_density/density_regressor_ckpt still set by
    # demo/app.py) when the plural ones aren't set — single-attribute
    # behavior is unchanged, just implemented as a one-item list internally.
    attr_lambda = getattr(hparams, 'lambda_attribute', 0.0)
    attr_ckpts = getattr(hparams, 'attribute_regressor_ckpts', None)
    attr_targets = getattr(hparams, 'attribute_targets', None)
    attr_weights = getattr(hparams, 'attribute_weights', None)
    if not attr_ckpts:
        single_target = getattr(hparams, 'attribute_target', None)
        single_ckpt = getattr(hparams, 'attribute_regressor_ckpt', None)
        if single_ckpt is None:
            single_target = getattr(hparams, 'density_target', None)
            attr_lambda = getattr(hparams, 'lambda_density', 0.0)
            single_ckpt = getattr(hparams, 'density_regressor_ckpt', None)
        attr_ckpts = [single_ckpt] if single_ckpt is not None else []
        attr_targets = [single_target] if single_target is not None else []
    if not attr_weights:
        attr_weights = [1.0] * len(attr_ckpts)
    attr_steering = (attr_lambda > 0 and len(attr_ckpts) > 0
                      and len(attr_ckpts) == len(attr_targets)
                      and all(c is not None and t is not None for c, t in zip(attr_ckpts, attr_targets)))

    # Optional λ taper: hold at full strength for the first
    # `attr_lambda_taper_hold_frac` of generation, then linearly decay to
    # `attr_lambda_taper_floor` (a fraction of attr_lambda, not absolute) by
    # the end. Motivated by the compounding-drift failure mode — guidance
    # establishes the target trajectory early, then continuing to push at
    # full strength on a context that's increasingly itself guided (rather
    # than real data) seems to be what drives later-generation cacophony,
    # not to prevent it.
    attr_lambda_taper = getattr(hparams, 'attribute_lambda_taper', False)
    attr_lambda_taper_hold_frac = getattr(hparams, 'attribute_lambda_taper_hold_frac', 0.4)
    attr_lambda_taper_floor = getattr(hparams, 'attribute_lambda_taper_floor', 0.2)

    # Each entry: {name, regressor, hard_window, target, weight}
    attr_specs = []
    if attr_steering:
        from attribute_control.note_density import NoteDensityRegressor
        reg_device = next(model.parameters()).device
        emb_weight = model.embeddings.weight.detach().to(reg_device)
        for ckpt_path, tgt, w in zip(attr_ckpts, attr_targets, attr_weights):
            reg_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            emb_dim = reg_ckpt['emb_dim']
            hidden = reg_ckpt.get('hidden_dim', 256)
            regressor = NoteDensityRegressor(emb_dim=emb_dim, hidden_dim=hidden)
            regressor.load_state_dict(reg_ckpt['model_state'])
            regressor.eval().to(reg_device)
            attr_specs.append({
                'name': reg_ckpt.get('attribute', 'density'),
                'regressor': regressor,
                'hard_window': reg_ckpt.get('density_hard_window', 48),
                'target': tgt,
                'weight': w,
            })
        taper_str = (f"  taper=hold{attr_lambda_taper_hold_frac:.0%}->floor{attr_lambda_taper_floor:.0%}"
                     if attr_lambda_taper else "")
        specs_str = "  ".join(f"{s['name']}(target={s['target']:.3f} w={s['weight']} "
                               f"hard_window={s['hard_window']})" for s in attr_specs)
        print(f"  Attribute control (R³{'+compose' if len(attr_specs) > 1 else ''}): "
              f"{specs_str}  λ={attr_lambda}{taper_str}")
    # attr_name/attr_target are only meaningful for a single-attribute run (the
    # compose/multi-attribute case has no one name or target to report) — used
    # below both for the diagnostics achieved-value trace and the wandb logging
    # block, which previously referenced these same names without ever
    # defining them (a silent NameError, swallowed by that block's bare
    # `except Exception: pass`, so it had never actually logged anything).
    attr_name = attr_specs[0]['name'] if len(attr_specs) == 1 else None
    attr_target = attr_specs[0]['target'] if len(attr_specs) == 1 else None
    compute_attr_fn = None
    if attr_name is not None:
        from attribute_control.attributes import ATTRIBUTES
        compute_attr_fn = ATTRIBUTES.get(attr_name)
    # ─────────────────────────────────────────────────────────────────────────

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

    # Confirmed by direct listening (not just metrics): attribute-guided
    # generation from a drum-free prompt routinely drifts into drums partway
    # through, even though unguided baselines from the same prompts don't.
    # density/polyphony/rhythm/syncopation's gate (is_duration or is_position)
    # doesn't distinguish a melodic note from a drum hit, so pushing for
    # "more/faster onsets" can be satisfied cheaply by switching to a drum
    # program, which is unconstrained by melody/harmony. Since Program_-1
    # (REMI_PROGRAM_MAX_ID, the drum-kit marker) is a single fixed vocab id,
    # a hard logit mask reliably prevents it rather than hoping a soft energy
    # term outweighs whatever's driving the drift.
    suppress_drum = getattr(hparams, 'attr_suppress_drum_if_prompt_drum_free', False)
    if suppress_drum:
        from attribute_control.note_density import REMI_PITCHDRUM_MIN, REMI_PITCHDRUM_MAX
        from attribute_control.attributes import REMI_PROGRAM_MAX_ID as _DRUM_PROGRAM_ID

    diagnostics_per_batch = []

    for batch_idx in range(bsz):
        prompt = prompt_tokens[batch_idx]
        model_energy_trace = []       # [(step_idx, energy)] — EBT's own MCMC energy, every step
        attr_energy_trace = []        # [(step_idx, energy)] — only steps where guidance was active
        achieved_trace = []           # [(step_idx, value)] — sampled periodically, guided runs only

        prompt_has_drums = suppress_drum and any(
            REMI_PITCHDRUM_MIN <= t <= REMI_PITCHDRUM_MAX for t in prompt
        )
        mask_drum_program = suppress_drum and not prompt_has_drums

        # Seed the sliding window with the last context_length tokens of the prompt.
        # If the prompt is shorter, use it as-is; if longer, this keeps the most
        # recent musical context at the start of generation.
        context = list(prompt[-context_length:]) if len(prompt) > context_length else list(prompt)

        generated = []
        gen_logprobs = []

        for step_idx in range(max_gen_len):
            window = context[-context_length:]
            input_tensor = torch.tensor(
                window, dtype=torch.long, device=device
            ).unsqueeze(0)

            # ── R³ attribute energy closure (Du et al. 2023, Eq. 6, generalized to a sum
            # over multiple active attributes — the "Reduce" composition step) ─────────
            # x = predicted_tokens (B, S, V): the full MCMC state. Only the LAST
            # position (predicted_tokens[:, -1:, :]) is ever sampled into the
            # output — the rest are refined for training purposes but discarded
            # every decode step — so each attribute's energy (and its gradient) is
            # computed from a bounded local window: the last `hard_window` committed
            # tokens (that attribute's OWN hard_window) plus that one soft
            # next-token distribution.
            # Each spec is gated independently (REMI only — see
            # _remi_step_is_relevant); only specs whose gate is active this step
            # contribute to the sum, so e.g. velocity/duration (different gates)
            # never actually combine at one step, while density/polyphony (same
            # gate) genuinely do. hparams.attr_gate_by_token_type=False reproduces
            # the old (ungated, fires-every-step) behavior for A/B comparison.
            gate_disabled = (not getattr(hparams, 'attr_gate_by_token_type', True)
                              or hparams.tokenizer_type != "REMI")
            active_specs = [
                s for s in attr_specs
                if gate_disabled or _remi_step_is_relevant(context[-1], s['name'])
            ] if attr_steering else []

            if active_specs:
                with torch.no_grad():
                    active_prepped = []
                    for s in active_specs:
                        local_hard = context[-s['hard_window']:]
                        hard_tokens = torch.tensor(local_hard, dtype=torch.long,
                                                   device=emb_weight.device)
                        active_prepped.append({
                            **s,
                            'hard_emb_sum': emb_weight[hard_tokens].sum(0),  # (D,)
                            'n_hard': len(local_hard),
                        })

                _dbg_called = [False]

                def _attribute_energy_fn(
                    predicted_tokens,
                    _specs=active_prepped,
                    _W=emb_weight,
                    _dbg=_dbg_called,
                ):
                    # Only the last position affects the sampled output; slicing
                    # here means autograd naturally zeros the gradient at every
                    # other position, so gradient-norm-matching in ebt_symbolic.py
                    # concentrates its whole budget on the one position that matters.
                    last = predicted_tokens[:, -1:, :]              # (B, 1, V)
                    # forward() with normalize_initial_condition=True passes softmax
                    # probabilities; ebt_advanced_inference passes raw logits.
                    if last.min() < 0 or last.max() > 1:
                        probs = torch.softmax(last.float(), dim=-1)
                    else:
                        probs = last.float()  # already probabilities
                    soft_emb_sum = (probs @ _W).sum(dim=1)          # (B, D)
                    n_soft = 1
                    dbg_parts = []
                    energy = 0.0
                    for s in _specs:
                        total = s['n_hard'] + n_soft
                        mean_emb = (s['hard_emb_sum'].unsqueeze(0) + soft_emb_sum) / total
                        pred_attr = s['regressor'](mean_emb)        # (B,)
                        # Unscaled per-attribute energy; the OUTER lambda_attribute is
                        # applied in ebt_symbolic.py after gradient normalisation so
                        # that λ=1 means "same magnitude as EBT gradient" regardless
                        # of regressor weight scale — the per-attribute `weight` here
                        # only controls the RELATIVE mix when composing >1 attribute.
                        term = s['weight'] * (total * (pred_attr - s['target']) ** 2).sum()
                        energy = energy + term
                        dbg_parts.append(f"{s['name']}: pred={pred_attr.item():.4f} "
                                          f"target={s['target']:.4f} term={term.item():.4f}")
                    if not _dbg[0]:
                        _dbg[0] = True
                        print(f"  [attr_fn] " + "  ".join(dbg_parts) +
                              f"  combined_energy={energy.item():.4f}")
                    # Stashed on the function object (same convention as
                    # lambda_scale below) so the per-step loop can read this
                    # step's attribute energy from the outside — it's computed
                    # here, inside the model's own forward pass, with no other
                    # way back out to the caller.
                    _attribute_energy_fn.last_energy = float(energy.item())
                    return energy

                # λ is read by ebt_symbolic.py via getattr(attr_energy_fn, 'lambda_scale')
                if attr_lambda_taper:
                    progress = step_idx / max(1, max_gen_len - 1)  # 0.0 at start, 1.0 at end
                    if progress <= attr_lambda_taper_hold_frac:
                        current_lambda = attr_lambda
                    else:
                        decay_progress = ((progress - attr_lambda_taper_hold_frac)
                                           / (1.0 - attr_lambda_taper_hold_frac))
                        current_lambda = attr_lambda * (
                            1.0 - decay_progress * (1.0 - attr_lambda_taper_floor)
                        )
                else:
                    current_lambda = attr_lambda
                _attribute_energy_fn.lambda_scale = current_lambda
                attr_energy_fn = _attribute_energy_fn
            else:
                attr_energy_fn = None
            # ──────────────────────────────────────────────────────────────────────

            with torch.no_grad():
                energy_sink = {}
                logits = call_model_forward_decode(hparams, model, input_tensor, 0, 1,
                                                   attr_energy_fn=attr_energy_fn, energy_sink=energy_sink)

                if logits.dim() == 3:
                    last_logits = logits[0, -1, :]
                else:
                    last_logits = logits[0]

                if mask_drum_program:
                    last_logits = last_logits.clone()
                    last_logits[_DRUM_PROGRAM_ID] = -float('inf')

            # ── Generation-diagnostics collection (demo trajectory plot) ────────
            # Model energy is recorded every step regardless of guidance — it's
            # what EBT is minimizing even in plain, unguided generation. Attribute
            # energy and the achieved-value trace only exist when guidance is
            # active this step (attr_energy_fn is None otherwise).
            energies = energy_sink.get('energies')
            if energies:
                model_energy_trace.append((step_idx, float(energies[-1].mean().item())))
            if attr_energy_fn is not None and hasattr(attr_energy_fn, 'last_energy'):
                attr_energy_trace.append((step_idx, attr_energy_fn.last_energy))
            # Achieved value is a token-counting/statistics function, not a model
            # call — cheap, but still only sampled every 8 steps (not every
            # step) since it needs no finer resolution to show a convergence
            # trend, and skipped on very short prefixes where these functions'
            # own statistics aren't meaningful yet.
            if compute_attr_fn is not None and len(generated) >= 8 and step_idx % 8 == 0:
                try:
                    achieved_trace.append((step_idx, compute_attr_fn(generated, hparams.tokenizer_type)))
                except Exception:
                    pass
            # ──────────────────────────────────────────────────────────────────────

            if temperature > 0:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(last_logits, dim=-1)

            next_token_val = int(next_token.reshape(-1)[0].item())

            if logprobs:
                with torch.no_grad():
                    lp = float(F.log_softmax(last_logits, dim=-1)[next_token_val].item())
                gen_logprobs.append(lp)

            context.append(next_token_val)
            generated.append(next_token_val)

        out_tokens.append(generated)
        out_logprobs.append(gen_logprobs)

        # Final sample so the achieved-value trace always ends on the actual
        # finished generation's value, not whatever the last step%8==0 landed on.
        if compute_attr_fn is not None and generated and (not achieved_trace or achieved_trace[-1][0] != max_gen_len - 1):
            try:
                achieved_trace.append((max_gen_len - 1, compute_attr_fn(generated, hparams.tokenizer_type)))
            except Exception:
                pass

        diagnostics_per_batch.append({
            'model_energy': model_energy_trace,
            'attribute_energy': attr_energy_trace if attr_steering else None,
            'achieved_trace': ({'name': attr_name, 'target': attr_target, 'points': achieved_trace}
                                if attr_name is not None else None),
        })

    result = {
        'prompt_tokens': prompt_tokens,
        'generation_tokens': out_tokens,
        'full_sequences': [p + g for p, g in zip(prompt_tokens, out_tokens)] if echo else out_tokens,
        # bsz is always 1 for every current caller (demo, sweeps) — indexing
        # [0] here rather than keeping a per-batch-item list keeps callers
        # simple; documented rather than silently assuming it.
        'diagnostics': diagnostics_per_batch[0] if len(diagnostics_per_batch) == 1 else diagnostics_per_batch,
    }

    if logprobs:
        result['generation_logprobs'] = out_logprobs

    # ── Attribute guidance WandB logging ──────────────────────────────────────
    if attr_steering and out_tokens:
        try:
            import wandb
            from attribute_control.attributes import ATTRIBUTES
            compute_fn = ATTRIBUTES.get(attr_name)
            achieved = compute_fn(
                [int(t) for t in out_tokens[0]],
                hparams.tokenizer_type,
            )
            log_data = {
                f'{attr_name}/target':   attr_target,
                f'{attr_name}/achieved': achieved,
                f'{attr_name}/error':    achieved - attr_target,
                f'{attr_name}/lambda':   attr_lambda,
                f'{attr_name}/gen_len':  len(out_tokens[0]),
            }
            if wandb.run is not None:
                wandb.log(log_data)
            else:
                proj = getattr(hparams, 'wandb_project', 'mus_symb_attr_control')
                wandb.init(
                    project=proj,
                    job_type='inference',
                    reinit='allow',
                    name=f"infer-{attr_name}-tgt{attr_target:.2f}-λ{attr_lambda:.1f}",
                )
                wandb.log(log_data)
                wandb.finish()
        except Exception:
            pass
    # ─────────────────────────────────────────────────────────────────────────

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

    # ── Attribute control (R³) setup — mirrors generate_remi's (see its own
    # comment block for the full derivation of the mechanism itself). This
    # was previously MISSING entirely from generate_anticipation: hparams.
    # attribute_target/lambda_attribute/attribute_regressor_ckpt were being
    # set by callers (listen_density_sweep.py, the demo) but never read here,
    # so every "guided" Anticipation generation was silently identical to an
    # unguided one. Only duration and pitch_register are meaningful for
    # Anticipation — velocity isn't encoded in this vocabulary at all (see
    # compute_velocity's docstring).
    attr_lambda = getattr(hparams, 'lambda_attribute', 0.0)
    attr_ckpts = getattr(hparams, 'attribute_regressor_ckpts', None)
    attr_targets = getattr(hparams, 'attribute_targets', None)
    attr_weights = getattr(hparams, 'attribute_weights', None)
    if not attr_ckpts:
        single_target = getattr(hparams, 'attribute_target', None)
        single_ckpt = getattr(hparams, 'attribute_regressor_ckpt', None)
        attr_ckpts = [single_ckpt] if single_ckpt is not None else []
        attr_targets = [single_target] if single_target is not None else []
    if not attr_weights:
        attr_weights = [1.0] * len(attr_ckpts)
    attr_steering = (attr_lambda > 0 and len(attr_ckpts) > 0
                      and len(attr_ckpts) == len(attr_targets)
                      and all(c is not None and t is not None for c, t in zip(attr_ckpts, attr_targets)))
    gate_disabled = not getattr(hparams, 'attr_gate_by_token_type', True)

    attr_specs = []
    if attr_steering:
        from attribute_control.note_density import NoteDensityRegressor
        reg_device = next(model.parameters()).device
        emb_weight = model.embeddings.weight.detach().to(reg_device)
        for ckpt_path, tgt, w in zip(attr_ckpts, attr_targets, attr_weights):
            reg_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            emb_dim = reg_ckpt['emb_dim']
            hidden = reg_ckpt.get('hidden_dim', 256)
            regressor = NoteDensityRegressor(emb_dim=emb_dim, hidden_dim=hidden)
            regressor.load_state_dict(reg_ckpt['model_state'])
            regressor.eval().to(reg_device)
            attr_specs.append({
                'name': reg_ckpt.get('attribute', 'density'),
                'regressor': regressor,
                'hard_window': reg_ckpt.get('density_hard_window', 48),
                'note_only_window': reg_ckpt.get('note_only_window', False),
                'target': tgt,
                'weight': w,
            })
        specs_str = "  ".join(f"{s['name']}(target={s['target']:.3f} w={s['weight']} "
                               f"hard_window={s['hard_window']}"
                               f"{' note_only' if s['note_only_window'] else ''})" for s in attr_specs)
        print(f"  Attribute control (R³{'+compose' if len(attr_specs) > 1 else ''}, Anticipation): "
              f"{specs_str}  λ={attr_lambda}")

    attr_name = attr_specs[0]['name'] if len(attr_specs) == 1 else None
    attr_target = attr_specs[0]['target'] if len(attr_specs) == 1 else None
    compute_attr_fn = None
    if attr_name is not None:
        from attribute_control.attributes import ATTRIBUTES
        compute_attr_fn = ATTRIBUTES.get(attr_name)

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

    diagnostics_per_batch = []

    for batch_idx in range(bsz):
        prompt = prompt_tokens[batch_idx]
        model_energy_trace = []       # [(step_idx, energy)] — EBT's own MCMC energy, every step
        attr_energy_trace = []        # [(step_idx, energy)] — only steps where guidance was active
        achieved_trace = []           # [(step_idx, value)] — sampled periodically, guided runs only

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

                    # ── R³ attribute energy closure — see generate_remi's own
                    # copy of this for the full derivation; identical mechanism,
                    # gated by _anticipation_step_is_relevant (triplet position)
                    # instead of REMI's last-token vocab-range decoding. ───────
                    active_specs = [
                        s for s in attr_specs
                        if gate_disabled or _anticipation_step_is_relevant(triplet_idx, s['name'])
                    ] if attr_steering else []

                    if active_specs:
                        committed = history + new_token_triplet
                        with torch.no_grad():
                            active_prepped = []
                            for s in active_specs:
                                # Must mirror train_density_regressor.py's
                                # DensityDataset exactly — a train/inference
                                # windowing mismatch silently produces a
                                # regressor that scores fine on its own
                                # training distribution but gives near-useless
                                # gradients here (confirmed the hard way on an
                                # earlier pitch_register regressor).
                                if s.get('note_only_window'):
                                    local_hard = committed[2::3][-s['hard_window']:]
                                else:
                                    local_hard = committed[-s['hard_window']:]
                                hard_tokens = torch.tensor(local_hard, dtype=torch.long,
                                                           device=emb_weight.device)
                                active_prepped.append({
                                    **s,
                                    'hard_emb_sum': emb_weight[hard_tokens].sum(0),
                                    'n_hard': len(local_hard),
                                })

                        def _attribute_energy_fn(predicted_tokens, _specs=active_prepped, _W=emb_weight):
                            last = predicted_tokens[:, -1:, :]
                            if last.min() < 0 or last.max() > 1:
                                probs = torch.softmax(last.float(), dim=-1)
                            else:
                                probs = last.float()
                            soft_emb_sum = (probs @ _W).sum(dim=1)
                            n_soft = 1
                            energy = 0.0
                            for s in _specs:
                                total = s['n_hard'] + n_soft
                                mean_emb = (s['hard_emb_sum'].unsqueeze(0) + soft_emb_sum) / total
                                pred_attr = s['regressor'](mean_emb)
                                term = s['weight'] * (total * (pred_attr - s['target']) ** 2).sum()
                                energy = energy + term
                            _attribute_energy_fn.last_energy = float(energy.item())
                            return energy

                        _attribute_energy_fn.lambda_scale = attr_lambda
                        attr_energy_fn = _attribute_energy_fn
                    else:
                        attr_energy_fn = None
                    # ──────────────────────────────────────────────────────────

                    energy_sink = {}
                    logits = call_model_forward_decode(hparams, model, input_tensor, 0, 1,
                                                        attr_energy_fn=attr_energy_fn, energy_sink=energy_sink)

                    if logits.dim() == 3:
                        last_logits = logits[0, -1, :]
                    else:
                        last_logits = logits[0]

                    # Mask invalid tokens; relative current time = current_time - time_offset
                    rel_current = max(0, current_time - time_offset)
                    last_logits = mask_invalid_anticipation_tokens(
                        last_logits.clone(), triplet_idx, rel_current, tokens_list
                    )

                    # ── Generation-diagnostics collection (demo trajectory plot) ──
                    flat_step_idx = _step * 3 + triplet_idx
                    energies = energy_sink.get('energies')
                    if energies:
                        model_energy_trace.append((flat_step_idx, float(energies[-1].mean().item())))
                    if attr_energy_fn is not None and hasattr(attr_energy_fn, 'last_energy'):
                        attr_energy_trace.append((flat_step_idx, attr_energy_fn.last_energy))
                    if (compute_attr_fn is not None and len(generated) >= 24
                            and flat_step_idx % 24 == 0):
                        try:
                            achieved_trace.append((flat_step_idx, compute_attr_fn(generated, hparams.tokenizer_type)))
                        except Exception:
                            pass
                    # ────────────────────────────────────────────────────────────

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

        if compute_attr_fn is not None and generated and (not achieved_trace or achieved_trace[-1][0] != len(generated) - 1):
            try:
                achieved_trace.append((len(generated) - 1, compute_attr_fn(generated, hparams.tokenizer_type)))
            except Exception:
                pass

        diagnostics_per_batch.append({
            'model_energy': model_energy_trace,
            'attribute_energy': attr_energy_trace if attr_steering else None,
            'achieved_trace': ({'name': attr_name, 'target': attr_target, 'points': achieved_trace}
                                if attr_name is not None else None),
        })

    result = {
        'prompt_tokens': prompt_tokens,
        'generation_tokens': generated_all,
        'diagnostics': diagnostics_per_batch[0] if len(diagnostics_per_batch) == 1 else diagnostics_per_batch,
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
