# Literature check: what is new in the guidance results?

_2026-10-09. Web searches plus the local `papers/` collection. Absence of
evidence is not proof of novelty: phrase claims as "to our knowledge"._

## Not new (cite as related work / baseline)
- **PPLM** (Dathathri et al., ICLR 2020, arXiv:1912.02164,
  https://arxiv.org/abs/1912.02164). Our "PPLM" baseline is a simplified
  *PPLM-style logit-gradient* variant: one normalized gradient step on the
  output logits through the attribute regressor. Original PPLM perturbs past
  key/values over several iterations and adds a KL term and geometric-mean
  fusion. No application of PPLM itself to symbolic music was found.
- **Inference-time steering of frozen music models** is an active area:
  - SMITIN, classifier-probe attention-head interventions (Koo et al., IEEE
    OJSP 2025, arXiv:2404.02252). Audio (MusicGen-style); monitors
    intervention strength because too much makes music incoherent.
  - MusicRFM, Recursive-Feature-Machine concept directions (Zhao &
    Beaglehole, ICLR 2026, arXiv:2510.19127). Audio.
  - Activation steering of the Multitrack Music Transformer on pitch and
    duration (arXiv:2605.31295, 2026). **Symbolic, closest prior work**,
    same kind of attributes.
  - PID feedback-control activation steering for symbolic music (in `papers/`).
  - Amadeus (arXiv:2508.20665): training-free attribute control by
    specifying attribute values during decoding.
- **Classifier/gradient guidance for symbolic music** exists for
  *diffusion* models (e.g. note-density classifier guidance in discrete
  diffusion, SCHmUBERT, IJCAI 2023; loss-gradient guidance cited in
  "Efficient Fine-Grained Guidance…"), not for autoregressive or energy-based
  token models.
- **"Strong guidance hurts quality / goes off-manifold"** is well known
  for diffusion (classifier and classifier-free guidance; e.g.
  arXiv:2505.20934, 2412.10193). In discrete diffusion, strong guidance is
  reported to *concentrate* probability.

## Plausibly new (to our knowledge)
1. **Energy-Based Transformers for symbolic music**, with attribute
   guidance folded into EBT's own MCMC refinement (R³). The EBT paper
   (Gladstone et al., arXiv:2507.02092) tests text and images. No music,
   MIDI or audio application was found.
2. **A controlled head-to-head** of EBT energy guidance vs plug-and-play AR
   methods (PPLM-style, tilt, best-of-N) on symbolic music under one
   protocol, with compute, prompt-bootstrap CIs and MIDI-level musical
   cost.
3. **The failure mechanism of energy guidance on discrete tokens.** Strong
   guidance *flattens* the attribute-token distribution (value entropy
   roughly doubles), which (a) drives the attribute toward its
   "random-token value" and so predicts the asymmetric steering in 5/5
   cases, and (b) produces random-note dissonance. This is the opposite of
   the probability *concentration* reported for guided discrete diffusion,
   and more specific than the generic "off-manifold" account.
4. **Evaluation:** sign-based directional accuracy can badly overstate
   control (best-of-N ~1.0 accuracy with ~10% progress toward target);
   report progress toward target by direction.

Not in this check: training-time conditioning (FIGARO, MuseMorphose,
Composer's Assistant 2, MIDI-RWKV attribute tokens, etc.). That is a
different family, already noted in the diary (2026-09-23).
