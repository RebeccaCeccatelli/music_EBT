# Unguided music quality: EBT vs GPT-2 vs Llama (MIDI-level metrics)

_2026-10-06. Tool: `eval/music_quality.py`. Generation:
`job_scripts/mus/eval/music_quality_generate.sh` (jobs 24948186–91)._

## Setup
- 100 validation songs (seed 0, the same songs for every model). Prompt =
  the first 64 tokens of the song; each model generates 256 tokens with
  T=0.7, top-p 0.9, no guidance.
- Final checkpoints: EBT REMI 33,732, EBT Ant s1 88,800, baselines at their
  final step.
- Only the continuation is scored: the prompt's notes are removed. Pieces
  with < 8 pitched notes are skipped, leaving 85 (REMI) and 89–94 (Ant).
- Reference = the real continuation of the same songs (prompt + 256 real
  tokens), scored the same way.
- Score = KDE overlapping area (OA) of each metric's distribution with the
  reference (Yang & Lerch 2020). **Ceiling:** two random 85-piece samples
  of real music overlap at OA 0.89 ± 0.01 (REMI) / 0.88 ± 0.02 (Ant)
  (bootstrap over the 500-window reference sets).

## Result: mean OA over the 8 quality metrics (ceiling ≈ 0.88–0.89)

| | EBT | GPT-2 | Llama |
|---|---|---|---|
| REMI | 0.83 | 0.86 | 0.86 |
| Anticipation | 0.76 | 0.78 | 0.79 |

Per-metric means (real → EBT / GPT-2 / Llama):

| metric | REMI | Anticipation |
|---|---|---|
| sharp dissonance | 0.051 → 0.053 / 0.046 / 0.030 | 0.072 → 0.045 / 0.039 / 0.054 |
| scale consistency | 0.973 → 0.984 / 0.979 / 0.984 | 0.980 → 0.992 / 0.993 / 0.994 |
| pitch-class entropy | 2.12 → 1.79 / 1.87 / 1.90 | 2.22 → 1.66 / 1.85 / 1.75 |
| pitch range (semitones) | 31.8 → 26.8 / 27.9 / 27.8 | 35.9 → 26.9 / 31.2 / 28.2 |
| bar self-similarity | 0.22 → 0.34 / 0.31 / 0.28 | 0.08 → 0.28 / 0.27 / 0.24 |
| groove consistency | 0.61 → 0.70 / 0.67 / 0.66 | 0.50 → 0.55 / 0.57 / 0.53 |
| key continuity (vs prompt) | 0.74 → 0.78 / 0.75 / 0.77 | 0.62 → 0.69 / 0.71 / 0.71 |

Across-set diversity (pitch / rhythm JSD): every model is as diverse as
or more diverse than the real continuations (REMI real 0.74/0.51, models
0.77–0.80/0.51–0.53; Ant real 0.73/0.58, models 0.78–0.82/0.54). No model
collapses to similar output across prompts.

## Reading
- **REMI: all three models are close to the real-data ceiling**
  (0.83–0.86 vs 0.89). Ant is further away (0.76–0.79) for all models.
- **EBT is slightly the lowest in both** (by ~0.03). That gap is about
  2× the ceiling's bootstrap spread, so call it small. It is not clearly
  significant.
- **The failure mode is "safe and repetitive", not cacophonic.** Unguided,
  no model is more dissonant than real music, and all stay in key at least
  as well as real continuations. They deviate in the same direction:
  fewer distinct pitch classes, narrower range, and much more bar-to-bar
  repetition (Ant: 0.24–0.28 vs real 0.08). EBT is the most extreme on
  entropy/range/repetition.
- So the cacophony heard in some outputs is not typical of unguided
  sampling. Candidates: guidance at high λ (the λ sweeps showed musicality
  costs), individual outliers (dissonance std is 0.10–0.14 vs real
  0.08–0.09, i.e. a heavier tail), or the old prompt/tokenizer
  inconsistencies. Next: score guided samples at several λ with the same
  pipeline.

## Caveats
- One sample per prompt at T=0.7; same-prompt diversity not yet measured.
- Metrics validated on synthetic sanity checks only. Calibrating them
  against listening ratings is still to do.
