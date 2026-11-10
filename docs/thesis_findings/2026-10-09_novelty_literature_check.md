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

## Addendum 2026-11-09: GigaMIDI + Anticipation, EBT for music, AR steering techniques

Web searches only (arXiv, Hugging Face, GitHub via search engine); no
exhaustive GitHub fork scan.

**GigaMIDI + Anticipation: not first, weaken the claim.**
- **MIDI-LLM** (Wu et al., arXiv:2511.03942, Nov 2025;
  https://arxiv.org/abs/2511.03942) adopts the AMT MIDI-like tokenization,
  continues pretraining Llama-3.2-1B on GigaMIDI (~1.4B tokens) and applies
  anticipation (infilling) augmentation "leveraging the AMT framework".
  So AMT tokenization has been applied to GigaMIDI before.
- They release only the model checkpoint and inference code
  (github.com/slSeanWU/MIDI-LLM). They don't describe GigaMIDI preprocessing
  or release a modified `anticipation` library, and no public fork of
  `jthickstun/anticipation` with GigaMIDI support was found.
- Original AMT (Thickstun et al., arXiv:2306.08620) trains on Lakh MIDI only.
  GigaMIDI's own tooling is MidiTok/Symusic-based (e.g. MIDI-GPT).
- **Thesis phrasing:** don't say "first to apply Anticipation to GigaMIDI".
  Instead: "we release an open-source adaptation of the anticipation
  library to GigaMIDI (fork), and train arrival-time anticipation models
  on it from scratch; MIDI-LLM [cite] also uses AMT tokens on GigaMIDI
  for LLM continued pretraining but does not release the data pipeline."
  The contribution is the reusable pipeline, the tokenizer comparison
  (REMI vs Ant-AT) and the corpus analysis.

**EBT for symbolic music: still plausibly new.**
- No hit applying Energy-Based Transformers (arXiv:2507.02092) to music,
  MIDI or audio, consistent with the 2026-10-09 citation check.
- Prior *energy-based* symbolic music exists and must be cited:
  MusicAOG (arXiv:2401.02678, EBM over hierarchical And-Or graphs,
  Metropolis-Hastings sampling, per-attribute temperature control) and
  CGRNN (energy-based recurrent model, 2019). Neither is a Transformer
  over token sequences with gradient-based refinement. Phrase as "to our
  knowledge, the first application of Energy-Based Transformers to
  symbolic music", not "first energy-based music model".

**NLP steering techniques on symbolic music: unchanged conclusion.**
- Still no published PPLM, expectation/exponential tilt or plain
  best-of-N reranking with an attribute regressor on symbolic music.
- New related work to cite (inference-time control of symbolic AR models):
  - Dual-level beam search for tonal tension (arXiv:2511.19342): token-
    and bar-level reranking, the closest relative of our best-of-N.
  - "Inference-time control of tonal tension in symbolic music
    generation" (OpenReview A5OnFNe23x; content not retrieved, bot wall:
    check before citing).
  - Activation steering on Aria for harmony (arXiv:2610.15957, "Harmonic
    Representations in Symbolic Music Transformers are Localized and
    Controllable"; title seen in results, page not retrieved).
  - Genre control via activation steering (arXiv:2506.10225, audio).
  - Already listed: PID steering (2606.18790), Prokopiou et al.
    (2605.31295), SMITIN, Amadeus, Ferreira et al. MCTS.
- Framing stays: the AR methods are transfers of NLP techniques and serve
  as strong baselines, not contributions.

## Correction 2026-11-10: PPLM has been applied to symbolic music

Earlier sections say no application of PPLM to symbolic music was found. That
is wrong. **Imasato, Miyazawa, Duncan & Nagai, "Using a Language Model to
Generate Music in Its Symbolic Domain While Controlling Its Perceived
Emotion", IEEE Access 11:52412–52428, 2023 (doi:10.1109/ACCESS.2023.3280603;
PDF in `papers/Explicit control over generation/`)**:
- GPT-2 trained from scratch on piano MIDI (MAESTRO, VGMIDI, EMOPIA),
  pitch-and-duration pair tokens (~20.8k vocab), context 1024.
- Full original PPLM: perturbs past key/values over m iterations with
  gradients from a one-layer discriminator on hidden states, KL term,
  geometric-mean fusion (γ_gm). Discriminator trained on EMOPIA's four
  Russell quadrants (MSE on class probabilities).
- Generation in 64-token chunks with a grid of PPLM hyperparameters; one
  candidate per chunk kept by an mgeval-based score plus discriminator
  probability, or by a human (so PPLM + best-of-N-style reranking).
- Evaluation: VGMIDI valence/arousal classifiers, mgeval metrics, 939-person
  listening study.

**Revised claim (used in the thesis, Background):** PPLM has been applied to
symbolic music for a categorical emotion attribute. To our knowledge it has
not been used to steer low-level musical attributes towards continuous
targets; expectation tilting has not been evaluated on symbolic music; and
these plug-and-play methods have not been compared under one protocol. Our
PPLM-style baseline (one logit-space step) is simpler than their full PPLM;
the Method says so.

Also: arXiv:2505.20934, cited above as a "strong guidance goes off-manifold"
source, is NatADiff (adversarial image diffusion) and is not relevant. The
thesis cites Rojas et al. 2025 (arXiv:2507.08965) for concentration under
strong guidance in masked discrete diffusion instead.

## Check 2026-11-10: the other steering methods

Web searches only (several queries per method, extended mode); not exhaustive.
- **Expectation tilting:** no application to symbolic music found (only
  constrained decoding framed as KL projection, for text). Claim kept,
  "to our knowledge".
- **Best-of-N / reranking:** no plain best-of-N with an attribute regressor
  found; search-based relatives exist: Ferreira et al. 2022 (MCTS, emotion),
  dual-level beam search (tonal tension, arXiv:2511.19342),
  **Text2midi-InferAlign** (Roy, Puri & Herremans, arXiv:2505.12669;
  reward-guided tree search, CLAP + key-consistency rewards, no retraining).
  Never claimed as new; InferAlign now cited.
- **FUDGE / GeDi on music:** nothing found. Not claimed.
- **EBT energy guidance:** nothing found; closest are classifier guidance in
  symbolic diffusion: SCHmUBERT (note density), **AGE-DM** (Zhang, Ferris,
  Yue & Xu, MMAsia 2024, doi:10.1145/3696409.3700289; emotion, guidance
  scale vs quality).
- **New closest prior work for low-level continuous inference-time control:**
  **Stochastic Control Guidance** (Huang et al., ICML 2024 oral,
  arXiv:2402.14285): training-free, plug-and-play guidance of a pretrained
  diffusion model towards pitch histogram and note density (L2, numeric
  targets from real pieces) and chord progression (0-1), forward evaluations
  of the rule only. Diffusion, not EBT/AR, and no comparison with AR
  plug-and-play methods, so our claims stand; now cited in the Background.
