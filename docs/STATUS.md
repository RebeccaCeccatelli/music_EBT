# Project status

_Last updated: 2026-10-09._ Living snapshot of where music-EBT stands; the
day-by-day history is in [diary/](diary/README.md), citable results in
[thesis_findings/](thesis_findings/README.md).

**Phase: wrap-up.** Pretraining is stopped; remaining work is regressors →
final experiments → demo freeze → writing.

## Final checkpoints
Under `~/orcd/scratch/rebcecca/music_EBT_logs/checkpoints/`. "Real best"
excludes spurious post-resume validation dips — **never pick a checkpoint
by the lowest `valid_loss` in its filename** (e.g. 0.1895, 0.1941, 0.3335,
0.5869, 0.6038 are all artifacts).

| Model | Use this (real best) | Also on disk |
|---|---|---|
| EBT small-REMI s1 | step 33,732 · val 0.7150 (job23868523) | 34,268 latest; 19,496 + 10,584 (old regressor pairings) |
| EBT small-Ant-AT s1 | step 88,800 · val 0.9903 (job22827009) | 99,900 final; 55,500 (old regressor pairing) |
| EBT small-Ant-AT stab+ | step 42,900 · val 1.0209 | 44,800 latest |
| GPT-2 small-REMI | step 99,660 · val 0.4756 | latest |
| Llama small-REMI | step 99,660 · val 0.5239 | latest |
| GPT-2 small-Ant-AT | step 98,900 · val 0.7557 | latest |
| Llama small-Ant-AT | step 100,000 · val 0.7735 | — |

Scratch: ~39 GB used by checkpoints after the 2026-10-05 cleanup.

## Running jobs
| Job | What | Paired EBT ckpt |
|---|---|---|


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
4. Writing: thesis sections from `thesis_findings/` + diary, then paper.
   Don't cite post-resume val readings. Thesis source: `eth-mit-master-thesis/` (own git repo,
   ignored here; Overleaf export, ch. 2-4 drafted 2026-10-08; build with
   `eth-mit-master-thesis/build.sh`). Reference PDFs: `papers/` (not tracked).
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
7. Repo tidy: project README (currently the upstream EBT one), final tag.

## Decisions
- 2026-10-05: no further pretraining. Final EBT ckpts = REMI 33,732 and
  Ant s1 88,800 (better val than stab+). Medium-Ant-AT dropped.
