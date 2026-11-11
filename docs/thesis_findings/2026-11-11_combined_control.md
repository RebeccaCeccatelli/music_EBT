# 2026-11-11 — Combined control (two attributes at once), EBT vs EBT-recipe baselines

**Reproduce with:** `python attribute_control/aggregate_combined_control.py OUT.json`
(output: `music_EBT_logs/attr_control/compose_guidance/combined_control_summary.json`).

## Question
How well does each method steer two attributes in the same generation, at the operating points tuned for
each attribute on its own, and at what musical cost?

## Setup
- Pairs: REMI velocity + duration; Anticipation duration + pitch register. 16 test prompts, targets ±1 and ±2 SD
  for both attributes, all four direction combinations → 8 conditions × 16 prompts = 128 samples per system.
- Runs (2026-11-09, `compose_guidance_sweep.py`): EBT REMI 33,732 / Ant 88,800; Llama-ebtrecipe REMI 33,196 /
  Ant 88,500; GPT-2-ebtrecipe REMI 33,732 / Ant 87,900. Jobs: EBT held-out 28472156 (REMI), 28472160 (Ant); EBT by
  ear 28472158, 28472162; Llama PPLM 28462282 (REMI), 28472164 (Ant 16:32); tilt 28462283/84; best-of-N
  28462285/86 (REMI, complete dirs `..._102537_...`, `..._103823_...`), 28462290/91 (Ant).
- Operating points: single-attribute held-out points (`figures/guidance_heldout/tune_*.json`). EBT per-attribute λ
  (REMI 0.05:0.055, Ant 0.005:0.012); PPLM REMI 32, Ant 16:32; tilt 1.0; best-of-N one N for both attributes
  (REMI 16 = larger of the two single-attribute N, Ant 8); plus compute-matched best-of-4 and the EBT by-ear points.
- Joint accuracy: both attributes moved from the system's own unguided baseline in the requested direction
  (strict). CI: bootstrap over prompts (10,000). Tests: exact sign-flip vs EBT held-out, paired by prompt (per-prompt
  joint accuracy over the 8 conditions), Holm over the 10 comparisons. Harsh = mean sharp dissonance of the guided
  samples vs the same run's unguided draws (cached `midi/quality_scores.csv`, ~85% of samples scored).

## Results (test prompts)

| tok | system | joint [95% CI] | mixed-direction joint | attr 1 / attr 2 acc | Δ vs EBT (Holm p) | harsh unguided → guided |
|---|---|---|---|---|---|---|
| REMI | **EBT** (0.05:0.055) | **0.75** [0.65, 0.86] | 0.78 | vel 0.94 / dur 0.81 | — | 0.051 → 0.060 |
| REMI | EBT by ear (0.03:0.02) | 0.77 [0.67, 0.88] | 0.77 | 0.95 / 0.82 | — | 0.058 → 0.056 |
| REMI | Llama PPLM 32 | 0.86 [0.75, 0.95] | 0.86 | 1.00 / 0.86 | +0.11 (0.31) | 0.056 → 0.071 |
| REMI | Llama tilt 1.0 | 0.77 [0.62, 0.91] | 0.75 | 1.00 / 0.77 | +0.02 (1.00) | 0.056 → 0.070 |
| REMI | GPT-2 tilt 1.0 | 0.76 [0.60, 0.90] | 0.73 | 1.00 / 0.76 | +0.01 (1.00) | 0.060 → 0.100 |
| REMI | Llama best-of-16 | 0.44 [0.23, 0.64] | 0.34 | 0.72 / 0.62 | −0.31 (0.02) | 0.052 → 0.079 |
| REMI | GPT-2 best-of-16 | 0.34 [0.19, 0.50] | 0.30 | 0.64 / 0.58 | −0.41 (0.001) | 0.064 → 0.070 |
| REMI | Llama best-of-4 | 0.31 [0.16, 0.47] | 0.34 | 0.67 / 0.48 | −0.44 (0.001) | 0.052 → 0.059 |
| REMI | GPT-2 best-of-4 | 0.25 [0.14, 0.36] | 0.19 | 0.59 / 0.50 | −0.50 (<0.001) | 0.064 → 0.060 |
| Ant | **EBT** (0.005:0.012) | **0.84** [0.76, 0.92] | 0.81 | dur 0.90 / pitch 0.92 | — | 0.052 → 0.093 |
| Ant | EBT by ear (0.005:0.005) | 0.58 [0.48, 0.67] | 0.59 | 0.80 / 0.72 | — | 0.041 → 0.049 |
| Ant | Llama PPLM 16:32 | 0.92 [0.88, 0.97] | 0.97 | 0.98 / 0.94 | +0.08 (0.61) | 0.046 → 0.055 |
| Ant | Llama best-of-8 | 0.56 [0.46, 0.66] | 0.67 | 0.60 / 0.94 | −0.29 (0.03) | 0.055 → 0.049 |
| Ant | GPT-2 best-of-8 | 0.51 [0.44, 0.58] | 0.48 | 0.57 / 0.91 | −0.34 (0.003) | 0.054 → 0.048 |
| Ant | Llama best-of-4 | 0.50 [0.44, 0.56] | 0.63 | 0.56 / 0.92 | −0.34 (0.004) | 0.055 → 0.048 |
| Ant | GPT-2 best-of-4 | 0.39 [0.33, 0.45] | 0.38 | 0.55 / 0.77 | −0.45 (<0.001) | 0.054 → 0.041 |

## Reading
- **Gradient/tilt methods ≈ EBT; best-of-N far behind.** EBT is statistically tied with PPLM and tilt on REMI and
  with PPLM on Anticipation (PPLM numerically ahead, +0.11 / +0.08, not significant after Holm). Every best-of-N
  setting is significantly worse (−0.29 to −0.50): reranking cannot find samples that hit two targets at once, and
  this holds at the compute-matched N = 4 and at N = 8–16 (2–3× the EBT's compute).
- **Joint ≈ product of single accuracies** for EBT (REMI 0.94 × 0.81 = 0.76 vs 0.75; Ant 0.90 × 0.92 = 0.83 vs
  0.84): the two attributes are guided on different tokens (no shared gate), so this is two controls side by side,
  not energy composition (see `sec:disc-limitations`, `sec:exp-tried`).
- **Mixed directions are not harder** for the guided methods (EBT 0.78 / 0.81, PPLM 0.86 / 0.97), but are for
  best-of-N on REMI (0.19–0.34): unguided samples rarely move two correlated attributes in opposite directions.
- **By-ear points trade accuracy for musicality** on Anticipation (0.58 joint at lower λ).
- **Musical cost:** on REMI EBT's harsh dissonance rises least among the guided methods (+0.009; PPLM +0.015,
  tilt +0.014 / +0.040). On Anticipation EBT's rises most (0.052 → 0.093; PPLM +0.009), driven by the pitch-register
  λ 0.012 (by-ear 0.005 keeps it at 0.049). Best-of-N does not raise it (it only reranks unguided samples).
- **Caveat:** the Anticipation duration regressors are scaled to ~⅓ of the true value (window-alignment bug,
  `tab:regressor-accuracy`), for EBT and PPLM alike. Combined control only uses ±1 and ±2 SD targets; a −1 SD request
  would be pushed the wrong way only for an unguided duration above ~0.077 (where baseline/3 < baseline − σ), and the
  largest Anticipation baseline in the single-attribute test runs is 0.071, so no wrong-sign requests are expected
  here. The missing stopping point (guidance keeps pushing past an upward target) still applies.
