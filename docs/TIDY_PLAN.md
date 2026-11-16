# Tidy-up and backup plan

Written 2026-11-16. Goal: nothing important lives only on the cluster, and the repo contains only what
Tonic needs. Space is not a concern. Tick items off here and log them in the diary.

Ground rules for every step:
- The submitted state is tagged `thesis-submission` (music_EBT, eth-mit-master-thesis, anticipation-musicEBT);
  anything removed from `main` stays reachable there. Thesis/README paths point at that tag's code.
- Code changes: compare old and new before committing (see "How to verify" below).
- Cluster rules (`/etc/claude-code/CLAUDE.md`): no recursive scans of `/orcd/scratch` or `/orcd/pool`;
  list known folders only.

## Done (2026-11-16)

- [x] Tags `thesis-submission` in the three repos.
- [x] Removed image/text/video code from the original EBT codebase (d5bbf69), verified old vs new.
- [x] REMI tokenizer config shipped with the repo + loader fallback (be3c534).
- [x] Backup folder `~/transfer/backup_2026-11-16/`: survey, 7 thesis checkpoints, results, git bundles
      (all refs), extras (tokenizer, Anticipation split indexes, signed declaration). SHA256SUMS verified.
- [x] Deleted resume snapshots, old Slurm logs, superseded zips and test leftovers (~145 GB).

## 1. Backup: finish

- [x] Copy `~/transfer/backup_2026-11-16/` to the Mac and run `shasum -a 256 -c SHA256SUMS` (all `OK`).
- [ ] Second copy somewhere else (external drive or ETH/MIT storage). Keep `survey_PRIVATE.tar` private.
- [ ] When the survey closes: export the responses CSV again (the backup has the 13 Nov export, n = 66),
      rerun `eval/plot_survey.py` if numbers change, add the CSV to the backup.
- [ ] Optional: add the 5 demo-only checkpoints (later Tonic saves, fully trained baselines; ~6 GB) to the
      backup if the demo should run elsewhere. Patterns are in `demo/app.py` (~line 204).
- [ ] Optional: `chmod 600 ~/music-EBT/.env`; revoke the HF/W&B tokens if no longer used.

## 2. Git leftovers: done 2026-11-16

All of these are inside `music_EBT_all_refs.bundle`.

```bash
cd ~/music-EBT
git worktree remove .claude/worktrees/music-quality     # unmerged branch music-quality-eval (139 commits, Oct 6)
git branch -D music-quality-eval backup-before-rewrite
git stash drop stash@{0}                                # "inference density control flags", Aug 19
git update-ref -d refs/original/refs/heads/main         # left over from the history rewrite
git -C data/mus/symbolic/tokenization/anticipation branch -D backup-before-rewrite
```

Check first: `git stash list` still shows only that entry (other sessions share the stash), and
`git worktree list` shows nothing else in use.

## 3. Code: remaining leftovers from the original EBT codebase

Same procedure as d5bbf69 (remove, then old-vs-new regression test).

- [ ] `model/model_utils.py`: image/video helpers (`save_frames`, `denormalize`, `load_image_encoder`,
      `call_style_gan_fvd`, `center_crop_arr`, the DiT/bidirectional-EBT builders around lines 600–650).
- [ ] Files only those helpers use: `model/diffusion/`, `model/diffusion_transformer.py`, `model/bi_ebt_adaln.py`.
- [ ] `base_model_trainer.py`: video branch in `log_metrics`; `self.to_pil` if unused.
- [ ] `train_model.py`: argparse flags that only applied to image/video/text (`--image_task`, `--backbone_type`,
      `--infer_generate_video`, `--debug_dataloader`, ...). **Keep them parseable**: old checkpoints store these
      hparams, so either keep the flags or make loading tolerate unknown keys. Low value, moderate risk.
- [ ] `docs/CODE_INFO.md`: rewrite for the music-only repo (or fold into README) instead of the inherited note.
- [ ] Decide on `MUS_NEUR` / `model/mus/ebt_neural.py` (neural-audio EBT, never used in the thesis): keep as
      future work or remove.

## 4. Code: optional reorganisation

- [ ] `attribute_control/`: move JSON/HTML data and `sweep_tables/` into `attribute_control/data/`; update paths.
- [ ] `eval/`: split into `eval/plots/` and `eval/analysis/`. Thesis cites `eval/...` paths, so only do it if
      those references may point at the tag.
- [ ] `demo/app.py` (4.5k lines): split into modules only if the demo moves (see 6). Not worth it otherwise.

Other repos (tonic, listening-study, listening-archive, thesis, anticipation-musicEBT): nothing to do.
Keep anticipation-musicEBT close to upstream.

## 5. Cluster: optional further tidy

- [ ] `checkpoints/` (179 GB, 404 files): keep the 7 thesis checkpoints, the 5 demo extras and the saves
      listed under "Also on disk" in STATUS.md (regressor pairings, stab+ 44,700, token-matched runs);
      review the rest one by one before deleting. Never pick by `valid_loss` in the filename.
- [ ] W&B local folders (`logs/wandb`, `wandb_demo`, `wandb_local`, `pool/music-EBT_wandb_runs`, `pool/wandb_cache`):
      check the runs are synced online, then delete.
- [ ] After the Mac copy is verified: delete `~/transfer/backup_2026-11-16/` if home space is needed.

## 6. Longevity (if cluster access ends)

- Websites (tonic, showcase, survey results, listening archive) are on GitHub Pages: unaffected.
- The hosted demo runs as a cluster job: it would go offline. To keep it, move it to Hugging Face Spaces
  (needs the demo checkpoints, the tokenizer config, `demo_cache`, the soundfont).
- Optional: publish the thesis checkpoints on Hugging Face Hub (with the tokenizer config) so others can use them.

## How to verify a code change

1. Scan checkpoint pickles for project-module references (only `torch`/`collections` expected).
2. `py_compile` all files; import `base_model_trainer` and `train_model` and diff `dir()` before/after.
3. CPU Slurm job (`mit_normal`, 4 CPUs, 96 GB) running old code (git worktree of the previous commit, with the
   anticipation submodule and `data/mus/symbolic/datasets` symlinked) and new code:
   - load all demo checkpoints with the demo loaders; compare weights hash + 32 seeded generated tokens;
   - fresh `ModelTrainer` from final-run hparams: optimizer groups, dataset sizes, seeded training loss
     (seed `random`, `numpy` and `torch`: training windows use Python `random`).
   Put the test folder on shared scratch (not `/tmp`: compute nodes can't see it).
