# Project status

_Last updated: 2026-10-05._ Living snapshot of where music-EBT stands; the
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
| 24934689 | REMI density regressor | REMI 33,732 |
| 24934690 | REMI velocity regressor | REMI 33,732 |
| 24934691 | REMI duration regressor | REMI 33,732 |
| 24934692 | REMI pitch_register regressor | REMI 33,732 |
| 24934693 | Ant duration regressor (5 ep, 300k) | Ant s1 88,800 |
| 24934694 | Ant pitch_register regressor (note-only, 5 ep, 300k) | Ant s1 88,800 |
| 24940374 / 24940376 | AR sweep Llama: tilt / best_of_n | Llama REMI 99,660 |
| 24940378 / 24940380 | AR sweep GPT-2: tilt / best_of_n | GPT-2 REMI 99,660 |
| 24940382 / 84 / 86 | Llama-space regressors velocity / duration / pitch_register (for PPLM), moved to `mit_preemptable` | Llama REMI 99,660 |

| 24944752 | Context-matched loss: EBT vs baselines on identical windows at 512 (+1024) | final ckpts |
| 24938101/2 | Music-quality generation smoke tests (3 samples) | REMI 33,732 / Llama Ant |
| 24936920 | Music-quality reference sets (500 real REMI + Ant windows → MIDI → scores.csv) | — |

Outputs: `~/orcd/scratch/rebcecca/music_EBT_logs/attr_control/<attr>_regressor_<tok>_<timestamp>/best.pt`;
music-quality refs in `.../music_EBT_logs/music_quality/reference_<tok>_256tok/`.

REMI density + velocity regressors (24934689/90) COMPLETED cleanly (no NaN).

## Open TODOs (in order)
1. Check the regressor jobs: no NaN (the new guard would now fail fast),
   sensible val loss; for Ant pitch_register re-run
   `attribute_control/eval_pitch_register_regressor.py` to confirm the
   predicted range is no longer collapsed.
2. Final experiments: EBT vs plug-and-play GPT-2/Llama guidance (REMI),
   paired checkpoints only. Code: `attribute_control/ar_guidance_sweep.py`,
   `job_scripts/mus/attr_control/ar_guidance_sweep.sh`,
   `attribute_control/aggregate_guidance_sweeps.py` (diary 2026-10-05).
   - [ ] Check AR sweeps 24940374/76 (Llama tilt/best_of_n), 24940378/80
         (GPT-2 tilt/best_of_n). Tilt does about 2k generations per model and
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
3. Demo freeze: point it at REMI 33,732 / Ant s1 88,800; consider making
   `_find_attribute_regressor()` check the regressor's `ebt_checkpoint`
   matches the selected EBT ckpt.
4. Writing: thesis sections from `thesis_findings/` + diary, then paper.
   Don't cite post-resume val readings.
5. Music-quality evaluation (branch `music-quality-eval`, worktree
   `.claude/worktrees/music-quality`): `eval/music_quality.py` scores decoded
   MIDI (scale consistency, sharp dissonance, groove, bar self-similarity, …)
   against real windows by KDE overlap. Next: generate unguided + guided MIDI
   from each final model and score; calibrate against blind listening ratings;
   small A/B listening test. Treat `ebt_energy`/`repetition_ratio` as diagnostics only.
   Also has prompt coherence + diversity. Smoke tests 24938101/2 pending
   (CPU quota); then 6 × 100-sample runs. Tokens-matched loss done:
   `thesis_findings/2026-10-05_tokens_matched_validation_loss.md` (branch).
6. Repo tidy: project README (currently the upstream EBT one), final tag.

## Decisions
- 2026-10-05: no further pretraining. Final EBT ckpts = REMI 33,732 and
  Ant s1 88,800 (better val than stab+). Medium-Ant-AT dropped.
