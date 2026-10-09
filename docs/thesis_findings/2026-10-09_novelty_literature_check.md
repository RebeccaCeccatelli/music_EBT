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
- **Post-hoc steering of conventional (AR) symbolic-music models predates
  2026**, so the AR side of our work is baselines, not a contribution:
  - Ferreira, Lelis & Whitehead 2022, MCTS decoding steered by an emotion
    classifier on a frozen symbolic LM (arXiv:2208.05162). The closest
    analogue of our best-of-N / reranking baseline.
  - Ferreira & Whitehead, ISMIR 2019: sentiment neurons in an mLSTM
    adjusted by a genetic algorithm (arXiv:2103.06125). Sometimes classed
    as training-time.
  - PID feedback-control activation steering for symbolic music
    (arXiv:2606.18790, June 2026), plus Prokopiou et al. 2026 (below).
  - Our PPLM-style logit-gradient and expectation-tilt variants were not
    found published for symbolic music, but they are direct transfers of
    standard NLP techniques: present them as well-chosen baselines.
  - Contrast: Kaliakatsos-Papakostas et al. ("Interactive Control of
    Explicit Musical Features in LSTM-based systems") is training-time
    conditioning (features as inputs).
- **Classifier/gradient guidance for symbolic music** exists for
  *diffusion* models (e.g. note-density classifier guidance in discrete
  diffusion, SCHmUBERT, IJCAI 2023; loss-gradient guidance cited in
  "Efficient Fine-Grained Guidance…"), not for autoregressive or energy-based
  token models.
- **"Strong guidance hurts quality / goes off-manifold"** is well known
  for diffusion (classifier and classifier-free guidance; e.g.
  arXiv:2505.20934, 2412.10193). In discrete diffusion, strong guidance is
  reported to *concentrate* probability.

## Closest prior work in detail
Prokopiou et al., "Latent Space Disentanglement via Activation Steering for
Interpretable Attribute Control in Symbolic Music Generation"
(arXiv:2605.31295, May 2026; `papers/Explicit control over generation/`):
- Frozen Multitrack Music Transformer (SOD); difference-in-means activation
  steering for pitch and duration; Gram–Schmidt for dual steering.
- Metrics: "Steering Success" (sign-based) and quality degradation from
  pitch-class entropy, scale and groove consistency (δ≈10 ≈ random notes).
- They report **asymmetric steering**: pitch +15.5 vs −29 semitones at
  α=±2 around a 65.7 baseline; duration −59% (floor) vs +407%. They
  attribute duration's asymmetry to a physical lower bound and leave
  pitch unexplained. Our random-token-value account predicts pitch's
  easier direction too (uniform MIDI pitch ≈ 63.5 < 65.7 → down easier),
  as a consistency check only: different steering mechanism, and their
  pitch-token range was not verified.

## Plausibly new (to our knowledge)
1. **Energy-Based Transformers for symbolic music**, with attribute
   guidance folded into EBT's own MCMC refinement (R³). The EBT paper
   (Gladstone et al., arXiv:2507.02092; ICLR 2026) tests text and images.
   **Citation check (2026-10-09):**
   - Google Scholar lists 22 citations of the ICLR version (that list was
     blocked by a bot check) and 5 of the arXiv version.
   - Semantic Scholar lists **37 citing papers; none applies EBTs to music,
     MIDI or audio generation**.
   - The only music-related one is "Text Dictates, Music Decorates:
     Energy-based Attention for Editable Dance Motion Generation" (Yoo et
     al., 2026, ECCV/arXiv). It generates *dance motion* conditioned on
     music, a different task.
   - The rest are pretraining, reasoning/IR and recursive models (e.g.
     "Energy-guided Recursive Model", "Explorative Modeling").
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
