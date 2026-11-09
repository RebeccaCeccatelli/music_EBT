# 2026-11-05 — Guidance with held-out operating points, EBT vs EBT-recipe baselines

**Supersedes the operating points of 2026-10-09 / the paired tests of 2026-10-15** for the main comparison:
operating points are now chosen on separate tuning prompts, and the baselines are the EBT-recipe ones
(same context 512, batch, steps and LR-schedule position as EBT; diary 2026-10-19).

## Setup
- Models: EBT REMI 33,732 / Ant s1 88,800. Llama-ebtrecipe REMI 33,196 / Ant 88,500; GPT-2-ebtrecipe
  REMI 33,732 / Ant 87,900. PPLM regressors trained on the Llama-ebtrecipe features (REMI 26348315–17,
  Ant 27933069 / 27917682). EBT keeps its 2026-10-05 regressors.
- Prompts: 16 **tune** prompts per tokenizer (`attr_control/prompt_splits.json`, disjoint from test) and the
  original 16 **test** prompts. Same targets (±0.5/1/2 SD), 3 unguided repeats, T 0.7, top-p 0.9, 256 tokens,
  same strength grids for tune and test.
- Operating point per (system, attribute), chosen on **tune**: highest strict accuracy with harsh dissonance
  ≤ real mean + 1 SD and OA vs own unguided ≥ 0.75 (same rule as before). Then evaluated on **test**.
- Tests on test prompts: paired by prompt, exact sign-flip p (2^16), Holm over all 31 comparisons
  (REMI + Ant, operating points + compute-matched best-of-4).
- Scripts: `job_scripts/mus/eval/score_guidance_heldout.sh` (job 27972185). Data:
  `figures/guidance_heldout/` (`tune_*.json` = tuning-set scores + operating points; `final_*_ebtrecipe.json`
  = test-set scores; `heldout_paired_holm.json` = all tests).

## Results (test prompts, at operating points tuned on held-out prompts)

| tok | attribute | EBT (λ) | Llama PPLM | Llama tilt | GPT-2 tilt | Llama best-of-N | GPT-2 best-of-N |
|---|---|---|---|---|---|---|---|
| REMI | duration | 0.79 (0.055) | 0.88 | 0.68 | 0.65 | 0.69 (N=8) | 0.69 (16) |
| REMI | pitch register | **0.51 (0.01)** ⚠ | 0.79 | 0.85 | 0.85 | 0.78 (16) | 0.75 (16) |
| REMI | velocity | 0.95 (0.05) | 0.99 | 0.97 | 0.98 | 0.75 (16) | 0.75 (8) |
| Ant | duration | 0.78 (0.005) | 0.95 | – | – | 0.97 (8) | 0.94 (8) |
| Ant | pitch register | 0.78 (0.012) | 0.93 | – | – | 0.97 (8) | 0.94 (8) |

Paired differences (other − EBT), uncorrected p → Holm:
- **REMI duration / velocity:** no significant difference after Holm. EBT ≈ PPLM (duration +0.08, p 0.13;
  velocity +0.04, p 0.5) ≈ tilt on velocity; EBT ahead of tilt on duration (−0.12 to −0.15, p 0.04–0.05, n.s.
  after Holm) and of compute-matched best-of-4 (−0.24 to −0.26, p 0.01, Holm 0.28–0.31).
- **REMI pitch register:** EBT behind every baseline; significant after Holm vs both tilts (+0.34, Holm 0.004)
  and Llama best-of-16 (+0.27, Holm 0.014). **This is driven by the operating-point choice** (below).
- **Anticipation:** PPLM and best-of-8 ahead of EBT by +0.15 to +0.19 on both attributes (p 0.004–0.03),
  not significant after Holm (0.11–0.60). Compute-matched best-of-4 is not clearly better (GPT-2 +0.09 / −0.01).

## ⚠ The REMI pitch-register operating point is a threshold artefact
On the tune prompts EBT pitch register reaches 0.85 at λ 0.02 with OA_unguided **0.749** — 0.001 under the
0.75 budget — so the rule falls back to λ 0.01 (tune 0.49, test 0.51). At λ ≥ 0.04 the music is clearly
cacophonic (OA_unguided 0.33–0.44, dissonance 0.18–0.25, both sets). With λ 0.02, EBT's test accuracy is
0.79 (OA_unguided 0.79): equal to PPLM (0.79) and best-of-16 (0.75–0.78), below tilt (0.85), as in the
in-sample results. Report the pre-registered rule's result, and this sensitivity alongside it.

The substantive point is the same in both readings: **EBT's usable pitch-register window is one grid step
wide** (λ 0.01 too weak, 0.02 borderline, 0.04 cacophonic), so a held-out procedure can't land in it
reliably. The baselines' operating points transfer from tune to test without this problem.

## Conclusions for the thesis
1. With fair baselines and held-out tuning, **REMI EBT guidance ties the like-for-like method (PPLM) on
   duration and velocity** and beats tilt / compute-matched best-of-N on duration.
2. **EBT guidance is fragile**: on pitch register its strength window is so narrow that held-out tuning
   misses it; the AR methods are robust to the same procedure.
3. **On Anticipation, PPLM and best-of-8 beat EBT by ~0.15–0.19** (consistent direction on both attributes,
   uncorrected p < 0.05, not significant after Holm over 31 tests with 16 prompts).
4. EBT costs ~6 forward-equivalents per token vs 1 for PPLM/tilt (best-of-N: N).

## Caveats
- 16 test prompts per tokenizer; Holm over 31 tests is conservative.
- Single training seed per model.
- EBTs trained without Langevin noise / step-size randomization / replay buffer (see training_runs.md);
  whether these widen the usable λ window is untested (decided 2026-11-05, deadline).
- The 100k baselines were not re-tuned (no tune sweeps); they remain a reference in `final_*_ebtrecipe.json`.
