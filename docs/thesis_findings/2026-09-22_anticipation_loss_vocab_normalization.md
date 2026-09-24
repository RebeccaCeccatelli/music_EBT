# Anticipation's pretraining loss gap is mostly a vocabulary-size artifact

**Date:** 2026-09-22
**Reproduce with:** `attribute_control/compute_unigram_entropy.py`

## The question

REMI's EBT checkpoint sits at valid_loss ≈ 0.76; Anticipation-Arrival-Time's
sits at valid_loss ≈ 1.09. Read at face value, that looks like Anticipation
pretraining is going considerably worse — and that reading motivated a
separate investigation into whether specific training hyperparameters
(effective batch size, MCMC step-size multiplier) explain the gap (see
"Open threads" below).

## Why the raw comparison is misleading

REMI's vocabulary is 427 tokens; Anticipation-Arrival-Time's is 55,028 — about
**129x larger**. Cross-entropy loss is bounded below by how much uncertainty
exists to eliminate in the first place, and a much larger vocabulary starts
with much more of it. Comparing the two raw loss values is closer to comparing
raw scores from a 4-question exam and a 500-question exam than a fair
apples-to-apples read of "how good is this model."

## Method

For each tokenizer, measure two baselines a model's loss can be compared
against:

1. **Uniform baseline** — `ln(vocab_size)`, the entropy if every token were
   equally likely. Quick sanity check, but unrealistic — real token
   frequencies are highly skewed.
2. **Oracle-unigram baseline** — the actual Shannon entropy of each
   tokenizer's real, empirical token-frequency distribution over validation
   data. This is what a "trivial but distribution-aware" model would already
   achieve with no context at all, and is the fairer bar for a trained model
   to clear.

Sampled by **token budget, not song count** (30M tokens each) — an earlier
pass sampled 2000 "songs" from each dataset, but REMI's dataset unit is a
full variable-length song (~15.8k tokens average) while Anticipation's is a
fixed 1024-token window, so equal song counts gave REMI ~15x more raw tokens
and undersampled Anticipation's long, sparse tail (only 17,851 of 55,028
vocab ids observed), biasing its entropy estimate down. Re-run with a matched
30M-token budget for both (Anticipation needed 29,297 sampled sequences to
reach it, seeing 31,575 of 55,028 ids — good coverage).

## Result

| | valid_loss | vocab size | uniform H (nats) | oracle-unigram H (nats) | reduction vs. uniform | reduction vs. oracle-unigram |
|---|---|---|---|---|---|---|
| REMI | 0.7551 | 427 | 6.057 | 4.568 | 87.53% | **83.47%** |
| Anticipation-Arrival-Time | 1.0864 | 55,028 | 10.916 | 7.431 | 90.05% | **85.38%** |

(REMI checkpoint: job22827004, step 18088. Anticipation checkpoint: the
"stabilized" lineage, job22885702, step 27400 — both current as of this
writing; these numbers will drift as those jobs keep training.)

> **Correction (2026-09-23):** an earlier version of this investigation
> treated REMI's valid_loss=0.6038 (step 10584, the checkpoint the REMI
> regressors were trained against) as a genuine baseline and later flagged
> "REMI'''s loss keeps climbing" as a new concern when spot-checks showed
> 0.7551 and 0.7647. Pulling the full wandb history resolved this: step
> 10584 sits right next to a single, isolated anomalous dip to ~0.665 at
> step 10016 that snaps back to ~0.796 at the very next logged point — the
> same spurious one-off post-resume-validation artifact already documented
> elsewhere in this project (previously seen as anomalously *good* readings
> right after a restart; this is the same mechanism). The REAL trend,
> excluding that one point, is a smooth, healthy, monotonic decrease from
> ~0.79 (step 11000) to ~0.752 (step 18008) — normal training, not
> degradation. The 0.7551 figure used in the Result table above was NOT
> affected by this artifact and remains valid; only the earlier
> "REMI is climbing" claim (made in conversation, not in this file) was
> wrong, and is retracted here for the record.

Against the fairer, distribution-aware baseline, **Anticipation's model
eliminates a slightly larger fraction of its own available uncertainty than
REMI's** — the opposite of what the raw loss numbers suggest.

## Interpretation

The "Anticipation pretrains worse" framing, based on raw loss, appears to be
substantially — maybe entirely — a units artifact of comparing cross-entropy
across two very differently sized vocabularies. Once normalized, there's no
clear evidence the pretraining itself is underperforming relative to REMI's.

## Caveats

- The oracle-unigram baseline is itself an *oracle* — it assumes perfect
  knowledge of the true marginal distribution, estimated here from a large
  but finite validation sample. Its accuracy depends on that sample being
  representative, which 30M tokens per tokenizer should comfortably provide,
  but it's still an estimate, not a closed form.
- This is a **snapshot** comparison of two checkpoints at different absolute
  step counts (18088 vs 27400) from different training lineages — it says
  "how good is each model at reducing its own task's uncertainty right now,"
  not "which one converges faster" (that question is what the batch-size /
  step-size probes below are for).
- This finding is about base language-modeling quality. It does **not**
  explain the separate, still-open finding that Anticipation's
  attribute-guided generation is much less accurate than REMI's (duration
  ~48% vs ~97% directional accuracy in the listening sweeps) — that looks
  more like a downstream steering-mechanism gap than a pretraining quality
  gap. See "Open threads."

## Open threads

- **Batch-size / step-size probes** (isolate-one-variable resume experiments
  from a shared Anticipation-stabilized checkpoint, step 27400,
  valid_loss=1.0864): raising the MCMC step-size LR multiplier to 20
  (matching REMI's) showed no improvement over 100 steps in two independent
  runs (1.0978→1.0985, then 1.0864→1.0877 — both flat to very slightly
  worse). Raising the effective batch size to 256 (matching REMI's) showed a
  real drop — **1.0864→1.0383 in 150 steps**, ~150x faster than the
  production run's own natural pace over a comparable window (1400 steps,
  1.0895→1.0864, only ~0.003) — consistent with the batch-size hypothesis
  (a larger batch smooths the noisier gradients a 129x-larger vocabulary
  produces; gradient norms were separately measured at 2-2.65x higher for
  Anticipation than REMI). One reading, not yet a confirmed trend — worth a
  longer confirmatory run before treating this as settled, but it is the
  first real positive signal in this investigation, and step-size showed
  nothing comparable across two tries.
- **Attribute-control accuracy gap**: resolved, see
  `2026-09-22_anticipation_attribute_guidance_was_missing.md` — the
  guidance wasn't "ungated," it was never implemented for Anticipation at
  all. Fixed and verified; independent of the pretraining-quality question
  this entry addresses.
