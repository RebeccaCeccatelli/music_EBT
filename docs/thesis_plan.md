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
| PPLM, tilt, best-of-N on symbolic music | **Corrected 2026-11-10:** PPLM *has* been applied to symbolic music, for a categorical emotion attribute (Imasato et al., IEEE Access 2023: full PPLM on GPT-2 piano music, EMOPIA 4-quadrant discriminator, plus per-chunk hyperparameter search and reranking). Not found: PPLM for low-level attributes with continuous targets, tilt on symbolic music, or a comparison of these methods under one protocol. Still baselines. | [Imasato et al.](https://doi.org/10.1109/ACCESS.2023.3280603), [PPLM](https://arxiv.org/abs/1912.02164), [dual-level beam search](https://arxiv.org/abs/2511.19342), [Ferreira et al. MCTS](https://arxiv.org/abs/2208.05162) |
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

The combined-control write-up is the only gap that blocks a Results section (results in since 2026-11-09). The rest are checks and writing.

- [ ] Write a `thesis_findings/` entry for combined control: EBT 25416469 (REMI shared λ), 28462287/88 (Ant shared λ), 28472156/60 (per-attribute held-out operating points), by-ear 28472158/62; EBT-recipe baselines 28462282–91 + Ant PPLM 16:32 28472164. Per-method accuracy and progress toward target, with CIs. (25416470–74 used the 100k baselines: not comparable.)
- [x] (written 2026-11-10; Future Work item still to add) **Limitations (`discussion.tex`): combined control does not test energy composition.** Guidance is gated by token type, and every attribute pair we combined acts on different steps (REMI: velocity after Pitch, duration after Velocity, pitch_register after Program, density after Duration/Position; Anticipation: duration at triplet slot 1, pitch_register at slot 2; `inference/mus/generate_music.py` `_remi_step_is_relevant` / `_anticipation_step_is_relevant`). So at each step only one regressor energy is active and the summed-energy (R³ "Reduce") case never occurs: the results show two single-attribute controls running side by side (joint accuracy ≈ product of the per-attribute accuracies: REMI 0.94 × 0.81 ≈ 0.75, Ant 0.90 × 0.92 ≈ 0.84), not that EBT energies compose. AR baselines act on the same gated steps, so the comparison is fair but shows no EBT-specific composition advantage. A real test needs attributes sharing a gate (e.g. density + polyphony) or ungated steering with re-tuned λ; not run (deadline). Nuance: attributes still interact through the generated context (Ant duration accuracy 0.90 → 0.80 at the same duration λ when pitch λ drops from 0.012 to 0.005/0.006, diary 2026-11-09). Pair it with a Future Work item in `conclusion.tex`.
- [x] Corpus attribute distributions written up in Experiments (`sec:exp-data-attributes`, table + figure, 2026-11-10). Interpretation (skew/floor → up/down asymmetry; σ_a = 2.6× Ant duration's own SD) still to write in `discussion.tex` (TODO comment there).
- [ ] Regressor held-out accuracy table (Pearson r, MAE per regressor/checkpoint) for Experiments; needs a short eval job per regressor (ask before submitting). TODO comment in `experiments.tex` after "Checkpoint selection".
- [x] Analyse the listening-survey responses, with and without the flagged respondent 686e30f1 (diary 2026-11-09). First full analysis 2026-11-11 (54 participants); results in diary 2026-11-11 and `music_EBT_logs/survey/analysis/`. Remaining: figures, thesis text.
- [ ] Run a quick final check that the forked `anticipation` repo is public and citable (URL, licence, README).
- [ ] Verify the two unchecked citations: Aria harmony steering (arXiv:2610.15957) and the OpenReview tonal-tension paper.
- [ ] Add the new references to `references.bib`: MIDI-LLM, MusicAOG, CGRNN, dual-level beam search, and the steering papers above.
- [x] (drafted 2026-11-11) Write a short "Training stability" paragraph in Experiments (`sec:exp-models`, TODO comment before "Checkpoint selection"): what went wrong when training the EBTs (early divergence, Anticipation gradient norms) and what fixed it. Contribution 2 in the Introduction promises it.
- [ ] **Training recipes: where each part goes** (agreed 2026-11-10, do after Method).
  - *Experiments `sec:exp-models`:* **done 2026-11-10** (EBT-recipe main table, three recipes, EBT variants, basis for comparison, why small). Still: Training stability paragraph; VERIFY on "small" reason and AdamW.
  - *(original note)* it currently presents the 100k AMT-style baselines (ctx 1024, ~100k steps) as the main ones, which contradicts the Introduction and all final numbers. Make the EBT-recipe baselines (`ebtrecipe`: ctx 512, EBT batch/steps/LR-schedule position) the main table; describe the 100k and token-matched Llama runs as reference points. Add a short "Training recipes" paragraph: EBT variants tried (REMI s1, Ant original s1, Ant "stabilized" with Langevin noise + random step size + replay buffer → dropped, worse on shared windows; step-size LR multiplier probe 2 vs 20 → no gain) and why s1 was kept. The "Training stability" paragraph (TODO already in experiments.tex) goes right after.
  - *Results `sec:res-lm`:* EBT-recipe comparison as headline (ctx-matched 0.683/0.700 vs 0.696; Ant 1.056/1.070 vs 1.102); reference points in one table/sentence: token-matched Llama, 100k baselines, stabilized EBT lineage.
  - *Discussion:* keep the existing Limitations "Scaling" paragraph (100k + token-matched); add why EBT-recipe is the fair comparison and what the 100k runs show; EBTs trained without the regularisers, untested whether they widen the narrow λ window.
  - *Appendix:* full hyperparameter tables for every run incl. unused lineages (from `docs/training_runs.md`); short note on the 2026-09-22 vocabulary-size explanation of Anticipation's higher loss, pointed to from Results.
- [x] (written 2026-11-11, 8 paragraphs in conclusion.tex sec:future) **Future Work: recipes not tried** (`conclusion.tex`). Candidates, each tied to a limitation:
  - the three EBT regularisers (Langevin noise, randomised step size, replay buffer) on REMI, and on Anticipation with a full-length run (the stabilized lineage stopped at ~44.7k steps); test whether they widen the usable λ window;
  - more refinement steps K at training and/or inference (only K = 2 used), trading compute for quality and controllability;
  - larger models and several sizes, to test the EBT scaling claim (single "small" size only);
  - longer EBT context (512 vs the AMT-style 1024), if memory allows (e.g. gradient checkpointing through the MCMC loop);
  - seeds: a single training seed per model.
  - **per-voice / per-instrument attributes:** every attribute is currently scored as an average over all notes in the window (mean velocity, mean duration, mean pitch), so a request like "louder" is met on average across instruments. Future work: regressors and targets per voice or per instrument (e.g. louder strings, longer bass notes), which is closer to how musicians think and gives finer-grained control. Motivated by the survey: a couple of participants asked whether they should judge the continuation per instrument or overall (user, 2026-11-10). Also mention this ambiguity in Discussion/Limitations next to the survey (instructions did not say whether to judge per instrument or overall).
- [ ] **Limitation (Discussion): compute environment.** Single L40S per run on a shared Slurm cluster with per-user GPU caps, job time limits and a preemptable partition (no warning on preemption) → every run spans many jobs; we built checkpoint/auto-resume logic (resume from highest compatible step, self-resubmit, requeue). Consequences worth stating: long wall-clock times, a corrupted checkpoint (stabilized Ant, 44,800), stitched W&B segments, and the limits it put on model size / context (EBT 512) and on the number of runs and seeds. Mechanism itself is described in Experiments `sec:exp-implementation`.
- [x] **Experiments `sec:exp-survey`:** written 2026-11-11 (only the recruitment/consent sentence left as a TODO).
- [x] **Experiments `sec:exp-implementation`:** W&B link (public 2026-11-11), fork URL, time limits, GPU hours (per run + project total) all in.
- [x] Fill Future Work in `conclusion.tex` (done 2026-11-11).
- [ ] Use the name TONIC consistently, and never cite post-resume validation readings.

## Completeness plan (agreed 2026-11-11, deadline 2026-11-13)

No new experiments: every claim has data. What is missing is analysis and plots from finished runs, and above all
the Results chapter (`results.tex`): only the listening-study section is written (by the other session, which owns
the survey). Agree which session writes which Results sections before editing `results.tex`.

**Must do (existing data, no new runs), in this order:**
1. [x] **Combined-control aggregation** — done 2026-11-11: `thesis_findings/2026-11-11_combined_control.md`,
   script `attribute_control/aggregate_combined_control.py`. (Was: only real analysis gap; also the `thesis_findings/` entry in Gaps above).
   EBT joint accuracy known (REMI 0.75 at 0.05:0.055, job 28472156; Ant 0.84 at 0.005:0.012, 28472160; Ant Llama PPLM
   16:32 0.92, 28472164). Missing: REMI baselines' joint accuracy (PPLM 32, tilt 1.0, best-of-N) + CIs, per-attribute
   accuracy, progress, quality. Outputs in `music_EBT_logs/attr_control/compose_guidance/`. CPU scoring.
2. [x] **Flattening mechanism (contribution 6):** done 2026-11-11, conclusion unchanged; final figure `thesis_findings/figures/guidance_final_ebtrecipe/` (update section in 2026-10-09_guidance_ebt_vs_ar.md). Was: check `thesis_findings/figures/guidance_final/*/guidance_mechanism.json`
   uses the EBT-recipe baselines for the AR side. The 2026-10-09 analysis compared against the old 100k baselines;
   if not redone, re-analyse saved outputs (no new runs).
3. [x] **Results sections** — drafted 2026-11-11 (LM, quality, control REMI/Ant, combined, trade-off + flattening); recheck Intro/Discussion numbers against them. Was: thesis figures as PDF (current ones are PNG working figures) + text:
   - `sec:res-lm`: loss curves + same-context (512) loss table, EBT vs EBT-recipe baselines, references
     (findings 2026-09-25, 2026-10-05, diary 2026-10-17).
   - `sec:res-quality`: unguided OA table with CIs (`figures/unguided_quality_ci/*_ebtrecipe.json`).
   - `sec:res-control-remi` / `-ant`: held-out table, paired Holm tests, compute-matched best-of-4, progress by
     direction (`figures/guidance_heldout/`, `figures/guidance_final/`). Caveat Ant duration progress/MAE
     (regressor ⅓-scale bug, `tab:regressor-accuracy`).
   - `sec:res-tradeoff`: strength vs musicality + flattening (`guidance_tradeoff.png`, `guidance_mechanism.png`).
   - Combined control (from item 1).

**Small checks (minutes each):**
- [ ] `anticipation-musicEBT` fork public, licence, README (also in Gaps above).
- [ ] Verify the two unchecked citations; add missing references (also in Gaps above).
- [ ] Future Work in `conclusion.tex`; compute-environment limitation in Discussion (writing only).

**Not before the deadline (new runs; already stated as limitations / future work):** fix + retrain the Ant duration
regressors and rerun those sweeps (few GPU-days); same-gate composition on the final checkpoints (preliminary runs in
`sec:exp-tried`); more seeds or larger models; activation-steering baseline.

## Writing order

Write in this order: Method, Experiments, Results, Discussion, Introduction, Conclusion, then Abstract. Each chapter then builds on settled text.
