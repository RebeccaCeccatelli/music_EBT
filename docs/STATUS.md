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

Outputs: `~/orcd/scratch/rebcecca/music_EBT_logs/attr_control/<attr>_regressor_<tok>_<timestamp>/best.pt`.

## Open TODOs (in order)
1. Check the regressor jobs: no NaN (the new guard would now fail fast),
   sensible val loss; for Ant pitch_register re-run
   `attribute_control/eval_pitch_register_regressor.py` to confirm the
   predicted range is no longer collapsed.
2. Final experiments on the paired checkpoints only: λ sweeps + attribute
   hit-rate for EBT (REMI, Ant) vs the GPT-2/Llama baselines.
3. Demo freeze: point it at REMI 33,732 / Ant s1 88,800; consider making
   `_find_attribute_regressor()` check the regressor's `ebt_checkpoint`
   matches the selected EBT ckpt.
4. Writing: thesis sections from `thesis_findings/` + diary, then paper.
   Don't cite post-resume val readings.
5. Repo tidy: project README (currently the upstream EBT one), final tag.

## Decisions
- 2026-10-05: no further pretraining. Final EBT ckpts = REMI 33,732 and
  Ant s1 88,800 (better val than stab+). Medium-Ant-AT dropped.
