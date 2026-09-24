# Anticipation's attribute-guided generation was never actually implemented

**Date:** 2026-09-22
**Status:** Root-caused and fixed in `inference/mus/generate_music.py`, verified working.

## The (wrong) prior conclusion

Every listening sweep run against Anticipation-Arrival-Time this project
(duration, pitch_register) found guidance to be weak — directional accuracy
around 48-53%, barely above chance, with tight-looking dose-response curves
that never actually got close to their targets. The working conclusion was
that Anticipation's attribute control is *real but weak*, plausibly because
its guidance fires ungated on every generation step (unlike REMI's, which is
gated to only the steps where it has an actual lever — see
`_remi_step_is_relevant`).

**That conclusion was wrong, and understated the problem by a lot.**

## What was actually happening

`generate_music.py` has two tokenizer-specific generation functions,
`generate_remi()` and `generate_anticipation()`. All of the R³
attribute-guidance machinery — building the regressor-based energy closure,
threading it into the model's forward pass as `attr_energy_fn` — lived
*only* inside `generate_remi()`. `generate_anticipation()`'s only model call
was:

```python
logits = call_model_forward_decode(hparams, model, input_tensor, 0, 1)
```

No `attr_energy_fn` argument, ever. `listen_density_sweep.py` (and the demo)
set `hparams.attribute_target` / `lambda_attribute` / `attribute_regressor_ckpt`
and call `generate_music()`, which routes Anticipation prompts to
`generate_anticipation()` — where those hparams are simply never read.

**Every "guided" Anticipation sample this project has ever generated was
functionally identical to an unguided baseline.** The regressor checkpoint
loaded without error (so nothing looked broken), it just never influenced
generation. The ~48-53% "barely better than chance" accuracy numbers are
exactly what you'd expect from measuring *unguided* samples against
essentially arbitrary target directions — because that's what they were.

## The fix

Ported the same R³ energy-closure mechanism into `generate_anticipation()`,
with one adaptation: Anticipation doesn't need REMI's last-token vocab-range
gate (`_remi_step_is_relevant`) at all. Its grammar is already a strict,
fixed `(time, duration, note)` triplet cycle, so which triplet slot is about
to be generated is known exactly from the loop position — no decoding
needed. New gate: `_anticipation_step_is_relevant(triplet_idx, attribute)`:

- `duration` → fires at `triplet_idx == 1` (about to generate the duration token)
- `pitch_register` → fires at `triplet_idx == 2` (about to generate the note token, which encodes pitch)

Also added the same per-step diagnostics collection (model energy,
attribute energy, achieved-value trace) that REMI's path already had, for
parity with the demo's generation-trajectory plot.

## Verification

CPU sanity check, Anticipation EBT checkpoint (job21959744, step 55500),
pitch_register regressor, 8 generated events (24 tokens), same prompt for
both runs:

| | pitch_register achieved | target |
|---|---|---|
| Unguided | 0.3425 | — |
| Guided (λ=0.08) | **0.5846** | 0.95 |

A real, large shift toward the target from a short, cheap generation — not
the near-chance behavior seen in every prior sweep. The gate fired exactly
where expected (`attribute_energy` populated only at triplet_idx==2: steps
2, 5, 8, 11, ...), confirming both the guidance and the gating work as
designed.

## Implication

The prior "Anticipation attribute control is weak" finding (and the
REMI-vs-Anticipation accuracy comparison built on it) is invalid — it was
comparing REMI's real guided generations against Anticipation's *unguided*
ones. This needs to be corrected before it goes anywhere near the thesis.

## Open threads

- **Full comprehensive sweeps need to be re-run for Anticipation** (duration,
  pitch_register) now that guidance actually works — the existing Anticipation
  sweep data in wandb is stale/invalid and shouldn't be cited.
- The REMI comprehensive sweeps (`thesis-{velocity,duration,pitch_register}-remi-comprehensive`)
  are unaffected — REMI's guidance was always correctly implemented.
- Worth deciding real λ ranges for Anticipation guidance from scratch (the
  demo's `_LAMBDA_RANGES` currently has no Anticipation-specific tuning at
  all) before running a large sweep, the same way REMI's ranges were tuned
  from real listening-sweep data.
- The `attr_gate_by_token_type=False` (ungated) A/B comparison that's
  supported for REMI is now available for Anticipation too, if useful for
  ablating how much the gate itself (vs. guidance existing at all)
  contributes.
