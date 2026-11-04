# Paired significance tests: guidance (EBT vs AR methods) and unguided quality

_2026-10-15. Tools: `eval/paired_guidance_stats.py` (guidance),
`eval/unguided_quality_ci.py` + `job_scripts/mus/eval/unguided_quality_ci.sh`
(unguided quality). Raw results:
`figures/guidance_final/{remi,ant}/paired_vs_ebt.json`,
`figures/unguided_quality_ci/{remi,ant-at-full-ar}.json`._

## Why
`2026-10-09_guidance_ebt_vs_ar.md` gives each method its own 95%
prompt-bootstrap CI. Every method ran on the **same 16 prompts**, so
overlapping per-method CIs are not a test. The right comparison is paired by
prompt.

## Guidance: method − EBT at the operating points (same tables as 2026-10-09)
Per prompt: strict accuracy over its 6 targets (±0.5/1/2 SD). Reported: mean
paired difference (other − EBT), 95% CI from resampling prompts (10k), exact
two-sided sign-flip permutation p over prompts (2^16 flips), and Holm-adjusted
p over all 29 comparisons (REMI + Anticipation, operating points + best-of-4).

**REMI**

| attribute | vs | diff | 95% CI | p | Holm |
|---|---|---|---|---|---|
| duration | Llama PPLM | −0.010 | [−0.063, +0.042] | 1.00 | 1.00 |
| duration | tilt (GPT-2 / Llama) | −0.188 / −0.167 | [−0.26, −0.12] / [−0.24, −0.08] | 0.001 / 0.004 | **0.026** / 0.09 |
| duration | best-of-16 (GPT-2 / Llama) | −0.177 / −0.198 | | 0.11 / 0.03 | n.s. |
| pitch_register | Llama PPLM | +0.062 | [−0.031, +0.156] | 0.33 | 1.00 |
| pitch_register | tilt (GPT-2 / Llama) | +0.094 / +0.052 | | 0.08 / 0.51 | n.s. |
| pitch_register | best-of-16 | −0.05 / −0.07 | | 0.71 / 0.53 | n.s. |
| velocity | Llama PPLM | +0.031 | [0.000, +0.094] | 1.00 | 1.00 |
| velocity | tilt | 0.000 / +0.021 | | 1.00 | n.s. |
| velocity | best-of-16 | −0.28 / −0.31 | | 0.06 | n.s. |
| all three | best-of-4 (≈ compute-matched) | −0.23 to −0.44 | | 0.002–0.05 | n.s. after Holm |

**Anticipation** (only best-of-N applies on the AR side)

| attribute | vs | diff | 95% CI | p | Holm |
|---|---|---|---|---|---|
| duration | best-of-16 / best-of-8 (GPT-2 / Llama) | +0.219 / +0.219 | [+0.135, +0.302] | 0.001 | **0.026** |
| duration | best-of-4 | +0.125 / +0.094 | | 0.06 / 0.24 | n.s. |
| pitch_register | best-of-16 (GPT-2 / Llama) | +0.208 / +0.240 | [+0.13, +0.29] / [+0.17, +0.30] | 0.001 / <0.001 | **0.024 / 0.004** |
| pitch_register | best-of-4 | +0.083 / +0.146 | | 0.17 / 0.04 | n.s. |

## Reading
- **EBT vs PPLM on REMI: no detectable difference on any attribute**
  (differences −0.01 to +0.06, all p ≥ 0.33; 12–15 of 16 prompts tied). The
  2026-10-09 wording "EBT ≈ PPLM within CIs" is confirmed by the paired test;
  it is a tie, not an EBT advantage.
- **EBT > tilt on REMI duration** is the one REMI difference that survives
  the multiple-comparison correction (GPT-2 tilt; Llama tilt p = 0.004
  uncorrected). EBT beats best-of-N on REMI in direction but, after Holm, not
  significantly at these 16 prompts.
- **Best-of-16 > EBT on Anticipation** survives correction on both
  attributes. At ≈ equal compute (best-of-4) the advantage is in the same
  direction but not significant after correction, so "best-of-N beats EBT
  even compute-matched" (2026-10-09) should be softened to "is at least as
  good".
- Recall from 2026-10-09: sign accuracy flatters best-of-N (its progress
  toward the target is only ~0.1–0.3).
- **Granularity:** with 16 prompts and many ties, the bootstrap CI and the
  permutation p can disagree (e.g. REMI pitch_register GPT-2 tilt: CI excludes
  0, p = 0.08). Prefer the permutation p.

## Caveats (unchanged by this analysis)
- **Operating points were chosen on these same 16 prompts**, which is
  optimistic for every method with a strength knob.
- **One training seed per model.** These tests cover prompt sampling only.

## Unguided quality: paired CIs and permutation tests
Job 25912904 (500 song-bootstrap draws + 500 paired permutations per comparison).
Setup: songs scored by every model and by the reference (REMI 83, Anticipation
control-free 96); the reference is the baselines' real continuation (prompt + 256
tokens; GPT-2 and Llama files are identical, checked). EBT's folder stores a
differently cut ground truth (228 vs 320 tokens on REMI), so it is not used.
Song-bootstrap draws bias KDE overlap downward (duplicated pieces), so the
permutation p is the main test; Holm over the 4 comparisons.

| | mean OA EBT / GPT-2 / Llama | GPT-2 − EBT | p (Holm) | Llama − EBT | p (Holm) |
|---|---|---|---|---|---|
| REMI | 0.830 / 0.869 / 0.851 | +0.040 [+0.006, +0.072] | 0.010 (**0.040**) | +0.022 [−0.023, +0.058] | 0.19 (0.38) |
| Anticipation | 0.786 / 0.823 / 0.788 | +0.037 [+0.002, +0.067] | 0.040 (0.12) | +0.002 [−0.049, +0.039] | 0.95 (0.95) |

- **GPT-2 is closer to real music than EBT on REMI** (survives correction), and
  by the same margin on Anticipation (p = 0.04, not after correction).
- **Llama and EBT are indistinguishable** on both tokenizers.
- So the 2026-10-06 reading "EBT slightly lowest, not clearly significant"
  becomes: EBT ≈ Llama; GPT-2 slightly ahead of both (≈ 0.04 OA, about half
  the gap between the models and the real-data ceiling). Same caveats: one
  sample per prompt, one training seed, metrics not yet validated by listeners.

## Likelihood at matched recipe: EBT vs Llama-ebtrecipe, paired over windows
Job 25908528 (`eval/context_matched_loss.py --per_window`, `SET=ebtrecipe WHICH=llama`). Llama trained
with EBT's recipe (ctx 512, same steps); best-validation checkpoint at or before EBT's step. Same 1000
seeded validation windows at ctx 512; EBT = last MCMC step; 10k paired bootstrap over windows.

| | EBT loss / top-1 | Llama loss / top-1 | Llama − EBT loss [95% CI] | top-1 diff [95% CI] |
|---|---|---|---|---|
| REMI | 0.6964 / 0.784 | 0.7004 / 0.784 | +0.0040 [+0.0024, +0.0057] | −0.0005 [−0.0013, +0.0003] |
| Anticipation | 1.1024 / 0.701 | 1.0563 / 0.711 | −0.0461 [−0.0496, −0.0429] | +0.0093 [+0.0083, +0.0103] |

- **REMI:** EBT's loss is lower by 0.6% (significant over windows), with identical top-1: a practical tie.
- **Anticipation:** Llama is clearly better on these windows (4% lower loss, +0.9 pt top-1), although the
  logged validation losses were almost equal (0.9903 vs 0.9878). Reason (confirmed by job 25938052,
  which reproduces both logged values: Llama 0.9879 on the first 8,576, EBT ≈ 0.990 MCMC-avg on the first
  4,288; on those same 8,576 windows Llama − EBT = −0.044 [−0.045, −0.043]): Anticipation training validation is unshuffled with
  `limit_val_batches=1072`, so EBT (batch 4) logs on the first 4,288 sequences and the baselines
  (batch 8) on the first 8,576, of 1.86 M. The logged Ant values are not comparable across models;
  these random, shared windows are. (An earlier note here blamed a ≥ 1024-token window filter; wrong:
  every stored Ant sequence is 1024 tokens.)
- Window CIs only; one seed per model. GPT-2-ebtrecipe comparison follows when those runs finish.
