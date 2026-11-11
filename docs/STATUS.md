# Project status

_Last updated: 2026-11-11._ Living snapshot of where music-EBT stands; the
day-by-day history is in [diary/](diary/README.md), citable results in
[thesis_findings/](thesis_findings/README.md).

**Phase: wrap-up.** Pretraining is stopped; remaining work is regressors →
final experiments → demo freeze → writing. Thesis: Intro–Experiments drafted (Experiments: corpus attribute section added 2026-11-10); note σ_a for targets is REMI's, used for both tokenizations. 2026-11-11: Method TODO/VERIFY resolved, PPLM-on-Anticipation + cross-refs fixed; Training-stability paragraph, composition caveat, EBT-vs-diffusion positioning drafted (diary).

**Next (2026-11-12): see "Completeness plan" in [thesis_plan.md](thesis_plan.md).** No new experiments; the
Results chapter is mostly empty (only the listening study is written). Order: combined-control aggregation →
check the flattening analysis uses the EBT-recipe baselines → Results figures/tables/text → small checks.

## Final checkpoints
Under `~/orcd/scratch/rebcecca/music_EBT_logs/checkpoints/`. "Real best"
excludes spurious post-resume validation dips — **never pick a checkpoint
by the lowest `valid_loss` in its filename** (e.g. 0.1895, 0.1941, 0.3335,
0.5869, 0.6038 are all artifacts).

| Model | Use this (real best) | Also on disk |
|---|---|---|
| EBT small-REMI s1 | step 33,732 · val 0.7150 (job23868523) | 34,268 latest; 19,496 + 10,584 (old regressor pairings) |
| EBT small-Ant-AT s1 | step 88,800 · val 0.9903 (job22827009) | 99,900 final; 55,500 (old regressor pairing) |
| EBT small-Ant-AT stab+ | step 42,900 · val 1.0209 | 44,700; 44,800 is corrupted (do not load) |
| GPT-2 small-REMI (step-matched, ebtrecipe) | step 33,732 · val 0.6972 (job25810910) | fully trained: 99,660 · val 0.4756 |
| Llama small-REMI (step-matched, ebtrecipe) | step 33,196 · val 0.7203 (job25810908) | fully trained: 99,660 · val 0.5239 (job21725352) |
| GPT-2 small-Ant-AT (step-matched, ebtrecipe) | step 87,900 · val 0.9971 (job25810911) | fully trained: 98,900 · val 0.7557 |
| Llama small-Ant-AT (step-matched, ebtrecipe) | step 88,500 · val 0.9878 (job25810909) | fully trained: 100,000 · val 0.7735 |

The step-matched baselines are the ones used in the final comparisons. The demo
offers each final plus one extra (baselines: fully trained; EBT: last save).

Scratch: ~39 GB used by checkpoints after the 2026-10-05 cleanup.

## Running jobs
| Job | What | Paired EBT ckpt |
|---|---|---|
| 28472156–64 | Combined control at per-attribute operating points (new `--strengths a:b`): EBT REMI 0.05:0.055 + Ant 0.005:0.012 (held-out), EBT by-ear pairs, Ant Llama PPLM 16:32 | REMI 33,732 · Ant 88,800 |
| 28462282–91 | Combined-control rerun on EBT-recipe baselines: REMI vel+dur (5 baseline jobs; EBT 25416469 reused), Ant dur+pitch (EBT ×2 λ halves, Llama PPLM/BoN, GPT-2 BoN) | REMI 33,732 · Ant 88,800 |
| ✅ 25810908 | Llama REMI EBT-recipe: done 2026-10-14 (33,732; best val 0.7203 @ 33,196) | REMI 33,732 |
| ✅ 25810909 | Llama Ant EBT-recipe: done 2026-10-15 (88,800; best val 0.9878 @ 88,500) | Ant s1 88,800 |
| ✅ 25810910 | GPT-2 REMI EBT-recipe: done 2026-10-15 (33,732; best val 0.6972) | REMI 33,732 |
| ✅ 25810911 | GPT-2 Ant EBT-recipe: done 2026-10-15 (88,800; best val 0.9971 @ 87,900, first-8,576 slice) | Ant s1 88,800 |
| ✅ 25908528 | Context-matched per-window eval, EBT vs Llama-ebtrecipe: REMI tie (+0.004 loss), Ant Llama better (−0.046) | both |
| ✅ 25938052 | Confirmed: logged Ant val uses different slices (EBT first 4,288, baselines first 8,576); on the same slice Llama −0.044. Was: Ant EBT vs Llama-ebtrecipe on the training-validation slice (first 8,576 seqs), to explain the logged-vs-eval gap | Ant s1 88,800 |
| ✅ 26151843 | Ctx-matched: REMI GPT-2 0.683 < EBT 0.696 ≈ Llama 0.700; Ant Llama 1.056 < GPT-2 1.070 < EBT 1.102 | both |
| ✅ 26152811 | `SET=antcurve` done: s1 88,800 best EBT (1.102); same-recipe Llama 1.056 / GPT-2 1.070; tokmatched Llama 1.136 | Ant |
| 27917681–82 | Ant Llama-ebtrecipe regressors (duration, pitch_register), resubmitted on preemptable, ~12 h (26348318/19 timed out) | Ant 88,500 (Llama) |
| 27917679–80 | REMI unguided gen Llama/GPT-2 ebtrecipe (26348856/57 crashed on json bug, fixed) | — |
| ✅ 27972185 | Held-out guidance analysis done → thesis_findings/2026-11-05_heldout_guidance.md | all |
| 27917761–62 | REMI Llama-ebtrecipe PPLM: tune + test prompts (~30 min each) | REMI 33,196 (Llama) |
| 27917683–85 | EBT tune sweeps REMI duration, Ant duration fine, Ant pitch_register fine (quota failures) | REMI 33,732 · Ant 88,800 |
| 26348315–19 | Regressors for Llama-ebtrecipe: REMI velocity/duration/pitch_register, Ant duration/pitch_register → then PPLM | REMI 33,196 · Ant 88,500 (Llama) |
| 26348856–70 | EBT-recipe baselines: unguided gen (4), REMI tilt + best-of-N (4), Ant best-of-N (2) for the fair survey | — |
| 26348915–21 | EBT tuning sweeps on held-out prompts (REMI 3, Ant 4) | REMI 33,732 · Ant 88,800 |
| 26348925–30 | EBT-recipe baseline tuning sweeps (tilt, best-of-N) on held-out prompts | — |

Resubmitted 2026-10-14 after node3006 (bad GPU) killed the previous chain; scripts now exclude it.

Outputs: `~/orcd/scratch/rebcecca/music_EBT_logs/attr_control/<attr>_regressor_<tok>_<timestamp>/best.pt`;
music-quality refs in `.../music_EBT_logs/music_quality/reference_<tok>_256tok/`.

**All 6 regressors retrained 2026-10-05, no NaN:** REMI density/velocity/duration/pitch_register
(24934689–92, vs REMI 33,732) and Ant duration/pitch_register (24934693/4, vs Ant s1 88,800).
Ant pitch_register verified: MAE 0.009, r 0.970 on real windows (old one: r 0.095).

## Open TODOs (in order)
1. Check the regressor jobs: no NaN (the new guard would now fail fast),
   sensible val loss; for Ant pitch_register re-run
   `attribute_control/eval_pitch_register_regressor.py` to confirm the
   predicted range is no longer collapsed.
2. Final experiments: **done 2026-10-09** →
   `thesis_findings/2026-10-09_guidance_ebt_vs_ar.md` (REMI + Anticipation, all methods,
   CIs, compute, operating points). Remaining optional: blind listening calibration.
   Original plan: EBT vs plug-and-play GPT-2/Llama guidance (REMI),
   paired checkpoints only. Code: `attribute_control/ar_guidance_sweep.py`,
   `job_scripts/mus/attr_control/ar_guidance_sweep.sh`,
   `attribute_control/aggregate_guidance_sweeps.py` (diary 2026-10-05).
   - [x] AR sweeps 24940374–80 done 2026-10-05 (tables in `attr_control/ar_guidance/`).
   - [x] Llama-space regressors done; PPLM 25261493 done 2026-10-08.
   - [x] best_of_n on Ant 25261494/5 done 2026-10-08.
   - [x] EBT REMI reruns submitted 25261488–90; Ant EBT 25261491/2; Ant best_of_n 25261494/5.
   - (old notes:) Tilt does about 2k generations per model and
         hasn't been timed on a GPU; if it hits the 6h limit, resubmit per
         attribute (`ATTRIBUTES=velocity` etc.).
   - [ ] Check Llama-space regressors 24940382/84/86 (no NaN, sensible val loss).
   - [ ] Submit the Llama PPLM sweep with them: `METHOD=pplm MODEL=llama
         ATTRIBUTES=velocity,duration,pitch_register REGRESSOR_CKPTS=<3 best.pt, same order>`.
   - [ ] Re-run the EBT REMI sweeps on step 33,732 with the new regressors
         (24934690-92), using the same 16 prompt ids and ±0.5/1/2 sd targets
         as `sweep_tables/`.
   - [ ] Anticipation: EBT sweeps for duration/pitch_register (only
         best_of_n applies on the AR side; tilt is REMI-only).
   - [ ] Score every sweep's saved MIDI (`<out_dir>/midi/`, EBT:
         `attr_control/listen_midi/<jobid>/`) with `eval/music_quality.py
         score` + `compare` against the real reference sets; join on
         `sample_id` to get musicality vs. strength per method.
   - [ ] Score everything with `aggregate_guidance_sweeps.py` (strict
         accuracy) and make an accuracy/MAE vs. bigram_ll comparison figure
         per attribute → new `thesis_findings/` entry.
   - [ ] Regenerate `thesis_findings/2026-09-24_remi_guidance_strength_sweeps.md`
         and its figures with strict accuracy. The old `aggregated.json`
         counted achieved==baseline ties as "down" hits, which inflates
         low-λ accuracy and probably part of the up/down crossover.
3. Demo freeze: **done 2026-10-08.** Defaults to the final checkpoints
   ("★ final"), and regressors are paired with the selected EBT checkpoint
   (warns otherwise). Remaining: a final click-through of the running demo.
   2026-11-11: EBT λ defaults = by-ear operating points (thesis_findings/2026-11-11_ear_operating_points.md; pitch register λ follows target direction).
   Static listening page: `demo/showcase/` (GitHub Pages via
   `.github/workflows/showcase-pages.yml`). Remaining: listen through the
   clips, swap any in `selection.json`. Live: https://rebeccaceccatelli.github.io/music_EBT/
   Blind listening survey: `demo/survey/` (page at /music_EBT/survey/, test
   mode until the endpoint was connected on 2026-10-09). Responses go to a Google Sheet
   and are emailed to the owner. Italian version at /music_EBT/survey/it/ (same sheet,
   `lang: "it"`). Collected v5 to 2026-11-11 (54 participants after dropping 1 empty); analysed with `demo/survey/analyze_survey.py` (v5 key), outputs in `music_EBT_logs/survey/analysis/`, summary in diary 2026-11-11. Figures (eval/plot_survey.py) and thesis text (sec:exp-survey, sec:res-survey, app:survey, limitation) done 2026-11-11; recruitment sentence filled (people the user knows + group chats).
4. Writing: thesis sections from `thesis_findings/` + diary, then paper.
   Don't cite post-resume val readings. Thesis source: `eth-mit-master-thesis/` (own git repo,
   ignored here; Overleaf export, ch. 2-4 drafted 2026-10-08; build with
   `eth-mit-master-thesis/build.sh`). Reference PDFs: `papers/` (not tracked).
   2026-11-09: thesis plan in `docs/thesis_plan.md`; novelty addendum in
   `thesis_findings/2026-10-09_novelty_literature_check.md`. To do before
   writing results: add the combined-control limitation (no shared gate → no energy summing; see thesis_plan.md) to discussion.tex; combined-control findings write-up (EBT 25416469 + EBT-recipe reruns 28462282–91; the old 25416470–74 used 100k baselines).
5. Music-quality evaluation (branch `music-quality-eval`, worktree
   `.claude/worktrees/music-quality`): `eval/music_quality.py` scores decoded
   MIDI (scale consistency, sharp dissonance, groove, bar self-similarity, …)
   against real windows by KDE overlap. Next: generate unguided + guided MIDI
   from each final model and score; calibrate against blind listening ratings;
   small A/B listening test. Treat `ebt_energy`/`repetition_ratio` as diagnostics only.
   Also has prompt coherence + diversity. **Done 2026-10-06:** unguided
   quality across models (`thesis_findings/2026-10-06_unguided_music_quality.md`)
   and context-matched perplexity. Next: guided samples at several λ.
   Smoke tests 24938101/2 pending
   (CPU quota); then 6 × 100-sample runs. Tokens-matched loss done:
   `thesis_findings/2026-10-05_tokens_matched_validation_loss.md` (branch).
6. ~~Anticipation prompts/references contain anticipated controls~~ **Done
   2026-10-09:** re-ran Ant guidance + unguided quality on control-free
   prompts/reference (`thesis_findings/2026-10-09_anticipation_control_free_rerun.md`).
   Ant numbers in 2026-10-06 are superseded. Thesis §4.3 updated.
7. Fairness follow-ups (diary 2026-10-09): Llama PPLM on Anticipation
   (regressors 25385280/81 → sweep), and token-matched Llama baselines
   (25404902 REMI → step 16,866; 25404903 Ant → step 22,200; then regressors,
   PPLM, unguided quality).
8. Repo tidy: project README (currently the upstream EBT one), final tag.

## Decisions
- 2026-10-09: the EBT music generation system is named **TONIC** (the tonic
  as the "home" where tension resolves ↔ EBT settling into an energy
  minimum). Use it in the thesis and demo; code/paths keep their names.
- 2026-10-05: no further pretraining. Final EBT ckpts = REMI 33,732 and
  Ant s1 88,800 (better val than stab+). Medium-Ant-AT dropped.
