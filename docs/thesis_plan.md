# TONIC Thesis Implementation Plan

_Nov 9, 2026 · Rebecca. Snapshot of the living doc at
https://claude.ai/artifact/YbbrY4TEMAM7EheQZBDVQD (edit there, then re-export here)._

## Overview

The thesis is due on 2026-11-13. The experiments are finished, but most of the text is still template placeholders. The thesis source is `eth-mit-master-thesis/main.tex` and currently builds to 33 pages. This plan fixes what the thesis claims, how each claim is phrased given prior work, what goes in each chapter, and what must happen before writing.

- **System name:** TONIC (decided 2026-10-09). Code and paths keep their old names.
- **Sources for the text:** `docs/thesis_findings/` (citable results), `docs/diary/` (history), `docs/STATUS.md` (checkpoints and open to-dos).
- **Scope:** no new experiments. The only analysis still missing is the combined-control write-up.

## Contributions

The Introduction lists six contributions. The second and third are the core of the thesis: EBTs for music, and steering through the MCMC refinement loop.

| # | Contribution | How to phrase it | Evidence |
| --- | --- | --- | --- |
| 1 | Anticipation on GigaMIDI: forked `anticipation` library with GigaMIDI support, Ant-AT models trained from scratch, corpus attribute-distribution analysis | "We release an open-source adaptation…", citing MIDI-LLM as prior AMT-on-GigaMIDI use. Never "first". Corpus analysis folded in here, not a headline item. | Fork repo; `2026-09-24_*corpus_attribute_distributions.md`; `2026-09-22_anticipation_loss_vocab_normalization.md`; `2026-10-09_anticipation_control_free_rerun.md` |
| 2 | TONIC: Energy-Based Transformers for symbolic music (REMI and Ant-AT), compared fairly with GPT-2 and Llama trained on the same recipe | "To our knowledge, the first application of EBTs to symbolic music". State openly that on context-matched loss TONIC ties on REMI and trails on Anticipation (1.102 vs Llama 1.056). | `2026-09-25_validation_loss_curves.md`; `2026-10-05_tokens_matched_validation_loss.md`; `2026-10-06_unguided_music_quality.md` |
| 3 | Plug-and-play attribute steering: regressors trained on frozen EBT embeddings steer the MCMC refinement loop at inference time, with no retraining. Compared against AR steering baselines (PPLM-style, expectation tilt, best-of-N). | The EBT is never retrained with guidance. The AR methods are transfers of NLP techniques, so they are baselines, not contributions. | `2026-10-09_guidance_ebt_vs_ar.md`; `2026-10-15_paired_significance.md`; `2026-11-05_heldout_guidance.md` |
| 4 | Combined-attribute control (REMI loudness + note length) | Needs a findings write-up with numbers and CIs first (see Gaps). | Compose sweeps 25416469–73; showcase Combined Control |
| 5 | Benchmarking and evaluation framework: TONIC vs GPT-2/Llama on (a) likelihood (context- and token-matched loss), (b) MIDI-level music quality (KDE overlap vs real music), (c) steering strength vs quality on held-out prompts with CIs and paired significance, (d) compute cost (forward-pass equivalents per token: EBT about 6x AR; compute-matched best-of-4). Progress toward target replaces sign accuracy. | The sign-accuracy critique and the flattening failure mechanism appear new (to our knowledge). | `eval/music_quality.py`; `2026-10-09_novelty_literature_check.md` |
| 6 | Interactive demo, static showcase site, blind listening survey (English and Italian) | Supporting artefacts; the survey results go in Results. | `demo/`; https://rebeccaceccatelli.github.io/music_EBT/ ; survey responses sheet |

Corpus-analysis sentence for the Introduction: "We adapt the Anticipation tokenization to GigaMIDI and characterize the corpus distributions of the attributes we control. These distributions define the scale of our guidance targets and explain why steering is asymmetric."

## Novelty and related work

Two claims changed after the 2026-11-09 search: "first Anticipation on GigaMIDI" is out, and earlier energy-based music models must be cited. Full notes are in the addendum to `docs/thesis_findings/2026-10-09_novelty_literature_check.md`.

| Claim | Verdict | Cite |
| --- | --- | --- |
| Anticipation tokenization on GigaMIDI | Not first. MIDI-LLM already uses AMT tokens on GigaMIDI, but releases no data pipeline or modified library. | [MIDI-LLM](https://arxiv.org/abs/2511.03942), [AMT](https://arxiv.org/abs/2306.08620), [GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI) |
| EBTs for symbolic music | Still appears new. No EBT application to music, MIDI or audio found. | [EBT](https://arxiv.org/abs/2507.02092); earlier EBMs: [MusicAOG](https://arxiv.org/abs/2401.02678), [CGRNN](https://researchconnect.stonybrook.edu/en/publications/energy-based-recurrent-model-for-stochastic-modeling-of-music/) |
| PPLM, tilt, best-of-N on symbolic music | Not found published, but direct NLP transfers: present as baselines. | [PPLM](https://arxiv.org/abs/1912.02164), [dual-level beam search](https://arxiv.org/abs/2511.19342), [Ferreira et al. MCTS](https://arxiv.org/abs/2208.05162) |
| Inference-time steering of music models | Active area, prior art. | [Prokopiou et al.](https://arxiv.org/abs/2605.31295) (closest), [PID steering](https://arxiv.org/abs/2606.18790), [SMITIN](https://arxiv.org/abs/2404.02252), [Amadeus](https://arxiv.org/abs/2508.20665), [Aria harmony steering](https://arxiv.org/abs/2610.15957), [genre steering](https://arxiv.org/abs/2506.10225) |
| Gradient guidance for symbolic music | Exists for diffusion models only. | SCHmUBERT and related (see findings file) |

Verify before citing: the Aria harmony-steering paper (title only seen) and "Inference-time control of tonal tension" (OpenReview A5OnFNe23x, page did not load).

## Chapter plan

The chapter order in `main.tex` stays as it is. Setup details go only in Experiments, findings only in Results, and interpretation only in Discussion.

| Chapter (file) | What goes in | Sources |
| --- | --- | --- |
| Abstract (`main.tex`) | 250–350 words: problem, TONIC, steering via MCMC, key numbers, takeaway. Written last. | All |
| Introduction (`introduction.tex`) | Motivation; gap (controllable symbolic generation without retraining); research questions; the six contributions; chapter roadmap. | Contributions section above |
| Background (`background.tex`) | Audio vs symbolic music; symbolic tokenization; state of the art with AR models; EBMs and EBTs; attribute control, training-time vs inference-time. Outline below. | Novelty section above; `papers/` |
| Method (`method.tex`) | TONIC architecture and objective; MCMC refinement; attribute regressors on frozen embeddings with soft-embedding simulation; guidance energy and λ; combined control; AR steering baselines as implemented. | `attribute_control/`; diary |
| Experiments (`experiments.tex`) | GigaMIDI and the Anticipation fork; splits; attribute definitions and corpus distributions (target scale in SD); training setup and EBT-recipe baselines; metrics; held-out tuning protocol; survey design; compute. | Corpus distribution findings; `STATUS.md` checkpoints |
| Results (`results.tex`) | Unguided quality and loss (context- and token-matched); single-attribute steering EBT vs AR with CIs; held-out guidance; combined control; compute cost and compute-matched comparisons; listening survey. Organised by research question. | `thesis_findings/` 2026-10-05 to 2026-11-05; survey sheet |
| Discussion (`discussion.tex`) | Why EBT steering works or fails; quality vs strength trade-off and operating points; asymmetric steering explained by corpus skew; flattening failure mechanism; limitations. | Novelty findings; corpus analysis |
| Conclusion (`conclusion.tex`) | Contributions as answered research questions; Future Work. No new results. | — |
| Appendix (`appendix_tokenization.tex` + new) | Tokenizer details; Anticipation inference fixes; hyperparameter tables; survey instructions and screenshots; per-condition breakdowns. | Diary; job scripts |

### Background outline

The chapter moves from the general to the specific and ends on the gap TONIC fills: control at inference time, with no retraining.

1. **Audio vs symbolic music** (short). Audio models generate waveforms or codec tokens (e.g. MusicGen, Jukebox): rich timbre, but costly and hard to edit. Symbolic music (MIDI, scores) encodes notes, timing and velocity explicitly: compact, editable, and its attributes can be measured exactly. This thesis works in the symbolic domain.
2. **Representing symbolic music as tokens.** Event-based MIDI-like encodings; REMI (bar and position tokens, beat-based); AMT arrival-time triplets and anticipation (interleaved controls for infilling). MidiTok and the `anticipation` library as tooling. Datasets: Lakh MIDI, GigaMIDI.
3. **Symbolic generation with "traditional" models.** Autoregressive Transformers (Music Transformer, Pop Music Transformer, MMT, AMT, Aria, MIDI-LLM) as the current forefront; briefly diffusion for symbolic music. Brief note on how these are evaluated.
4. **Energy-based models and EBTs.** EBMs and MCMC sampling; earlier music EBMs (MusicAOG, CGRNN); Energy-Based Transformers: prediction as energy minimisation by iterative refinement.
5. **Attribute control.**
   1. *Training-time* (most existing methods): conditioning tokens or attributes as inputs (FIGARO, MuseMorphose, MIDI-RWKV, SymPAC, XMusic, Composer's Assistant). Each new attribute needs new labels and retraining.
   2. *Inference-time / plug-and-play*: steering a frozen model. NLP origin (PPLM, reranking/best-of-N); music: MCTS with a classifier, activation steering (Prokopiou et al., PID steering, SMITIN), beam search for tension, Amadeus; classifier guidance in symbolic diffusion.
   3. *Where TONIC sits:* inference-time and plug-and-play. A small regressor per attribute is trained on frozen embeddings; the base model is never retrained, and attributes can be combined. The EBT's refinement loop is a natural place for the gradient, which AR models lack.

## Gaps and to-dos

The combined-control write-up is the only gap that blocks a Results section. The rest are checks and writing.

- [ ] Write a `thesis_findings/` entry for combined control from compose sweeps 25416469–73: per-method accuracy and progress toward target, with CIs.
- [ ] Analyse the listening-survey responses, with and without the flagged respondent 686e30f1 (diary 2026-11-09).
- [ ] Run a quick final check that the forked `anticipation` repo is public and citable (URL, licence, README).
- [ ] Verify the two unchecked citations: Aria harmony steering (arXiv:2610.15957) and the OpenReview tonal-tension paper.
- [ ] Add the new references to `references.bib`: MIDI-LLM, MusicAOG, CGRNN, dual-level beam search, and the steering papers above.
- [ ] Fill Future Work in `conclusion.tex`, with each item tied to a limitation in `discussion.tex`.
- [ ] Use the name TONIC consistently, and never cite post-resume validation readings.

## Writing order

Write in this order: Method, Experiments, Results, Discussion, Introduction, Conclusion, then Abstract. Each chapter then builds on settled text.
