"""
Plug-and-play attribute guidance for the autoregressive baselines (GPT-2,
Llama) — the comparison point for EBT's R³ guidance. Every method here leaves
the AR model frozen and untouched, same as EBT's: the question it answers is
"given a frozen model and no retraining, how well can each architecture be
steered", not "how well can a model be trained to be conditional" (that's
training-time conditioning — attribute tokens / cross-attention / CFG — a
different family, see docs/diary/2026-09-23.md's literature notes).

Methods (see attribute_control/ar_guidance_sweep.py for how each is run):

  - ExpectationTilt (logit processor, REMI only): at the steps where the next
    token directly carries the attribute (same gates as EBT's guidance,
    _remi_step_is_relevant), exponentially tilts the model's distribution over
    the attribute's own value tokens (Velocity_*, Duration_*, Pitch_*) so its
    expected value moves a fraction `strength` of the way from the model's
    own expectation toward the target. This is the minimum-KL change to the
    model's distribution that satisfies that moment constraint (an
    I-projection), i.e. the gentlest possible per-step nudge that gets there.
    It needs NO learned regressor, only a hand-written token->value map —
    which is only possible because these three REMI attributes are each read
    straight off a single token type. That makes it a deliberately strong,
    attribute-specific baseline; EBT's regressor-based guidance is generic
    (works for attributes like density that no single token carries).

  - Best-of-N (reranking, any tokenizer): sample N unguided continuations and
    keep the one whose rule-based attribute value is closest to the target.
    Uses the exact scoring function the evaluation itself uses, so it's an
    oracle-reranker — the standard sanity-check baseline.

  - PPLM-style gradient (Llama only): one normalized gradient step on the
    next-token logits through the SAME regressor energy EBT uses, trained on
    Llama's own embedding space. Already implemented in
    inference/mus/generate_music.py (call_model_forward_decode's Llama
    branch); the sweep script just drives it.
"""

from typing import Optional

import torch

from attribute_control.attributes import (
    REMI_VELOCITY_MIN_ID, REMI_VELOCITY_MAX_ID, REMI_VELOCITY_MIN_VAL, REMI_VELOCITY_STEP,
    REMI_DURATION_MIN_ID, REMI_DURATION_MAX_ID,
    REMI_PITCH_REGISTER_MIN_ID, REMI_PITCH_REGISTER_MAX_ID, REMI_PITCH_MIDI_OFFSET,
)


def remi_attribute_token_values(attribute: str):
    """(token_ids, values) for the tokens that directly carry `attribute`,
    each value on exactly the scale attributes.py's compute_* averages — so a
    per-step expected value equal to the target gives a sequence-level value
    near the target."""
    if attribute == 'velocity':
        ids = list(range(REMI_VELOCITY_MIN_ID, REMI_VELOCITY_MAX_ID + 1))
        vals = [(REMI_VELOCITY_MIN_VAL + REMI_VELOCITY_STEP * (t - REMI_VELOCITY_MIN_ID)) / 127.0
                for t in ids]
    elif attribute == 'duration':
        span = REMI_DURATION_MAX_ID - REMI_DURATION_MIN_ID
        ids = list(range(REMI_DURATION_MIN_ID, REMI_DURATION_MAX_ID + 1))
        vals = [(t - REMI_DURATION_MIN_ID) / span for t in ids]
    elif attribute == 'pitch_register':
        ids = list(range(REMI_PITCH_REGISTER_MIN_ID, REMI_PITCH_REGISTER_MAX_ID + 1))
        vals = [(t + REMI_PITCH_MIDI_OFFSET) / 127.0 for t in ids]
    else:
        raise NotImplementedError(
            f"ExpectationTilt has no token->value map for '{attribute}': only attributes "
            "read straight off one REMI token type (velocity, duration, pitch_register) "
            "can be steered this way."
        )
    return ids, vals


def _tilted_mean(log_q: torch.Tensor, v: torch.Tensor, theta: float) -> float:
    w = torch.softmax(log_q + theta * v, dim=-1)
    return float((w * v).sum().item())


class ExpectationTilt:
    """
    Logit processor for generate_remi (hparams.logit_processor). Called once
    per decode step as processor(context, last_logits) -> last_logits, on raw
    (pre-temperature) logits; temperature is accounted for internally so the
    tilt is solved on the distribution that's actually sampled from.

    strength in [0, 1]: fraction of the gap between the model's own per-step
    expected value and the target to close. 1.0 = per-step expectation equals
    the target exactly; 0 = no change.
    """

    # Below this much probability mass on the attribute's tokens, the gate
    # misfired (e.g. a drum Program followed by PitchDrum, not Pitch) — leave
    # the step alone rather than force an out-of-grammar token.
    MIN_ATTR_MASS = 0.05
    THETA_MAX = 500.0
    BISECT_ITERS = 40

    def __init__(self, attribute: str, target: float, strength: float, temperature: float,
                 gate_by_token_type: bool = True):
        from inference.mus.generate_music import _remi_step_is_relevant
        self._is_relevant = _remi_step_is_relevant
        self.attribute = attribute
        self.target = float(target)
        self.strength = float(strength)
        self.temperature = float(temperature) if temperature > 0 else 1.0
        self.gate_by_token_type = gate_by_token_type
        ids, vals = remi_attribute_token_values(attribute)
        self._ids_list = ids
        self._ids = None   # moved to the logits' device lazily
        self._v_cpu = torch.tensor(vals, dtype=torch.float32)
        self.n_tilted_steps = 0

    def __call__(self, context, last_logits: torch.Tensor) -> torch.Tensor:
        if self.strength <= 0:
            return last_logits
        if self.gate_by_token_type and not self._is_relevant(context[-1], self.attribute):
            return last_logits
        if self._ids is None or self._ids.device != last_logits.device:
            self._ids = torch.tensor(self._ids_list, dtype=torch.long, device=last_logits.device)

        scaled = last_logits.float() / self.temperature
        log_p = torch.log_softmax(scaled, dim=-1)
        log_p_attr = log_p[self._ids]
        if float(log_p_attr.exp().sum().item()) < self.MIN_ATTR_MASS:
            return last_logits
        # Only 32-89 values: solve on CPU, avoiding ~40 GPU syncs per step.
        log_q = torch.log_softmax(log_p_attr, dim=-1).cpu()  # distribution within the attribute's tokens
        v = self._v_cpu

        model_mean = float((log_q.exp() * v).sum().item())
        desired = model_mean + self.strength * (self.target - model_mean)
        # The tilted mean is monotonic in theta and bounded by the extreme
        # values in the support, so the target is clipped into what's reachable.
        lo_v, hi_v = float(v.min().item()), float(v.max().item())
        desired = min(max(desired, lo_v + 1e-4), hi_v - 1e-4)

        lo, hi = -self.THETA_MAX, self.THETA_MAX
        for _ in range(self.BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            if _tilted_mean(log_q, v, mid) < desired:
                lo = mid
            else:
                hi = mid
        theta = 0.5 * (lo + hi)

        # Reweight within the attribute's tokens while keeping their TOTAL
        # mass unchanged, so the tilt never shifts probability onto or off of
        # other token types — it only changes WHICH value gets picked.
        log_norm = torch.logsumexp(log_q + theta * v, dim=-1)
        delta = theta * v - log_norm                          # in tempered-logit units
        out = last_logits.clone()
        out[self._ids] = out[self._ids] + (delta * self.temperature).to(out.device, out.dtype)
        self.n_tilted_steps += 1
        return out


def make_logit_processor(method: str, attribute: str, target: Optional[float], strength: float,
                         temperature: float, gate_by_token_type: bool = True):
    if method == 'tilt':
        return ExpectationTilt(attribute, target, strength, temperature, gate_by_token_type)
    return None
