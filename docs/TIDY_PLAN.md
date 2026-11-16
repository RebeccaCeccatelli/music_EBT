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

## 3. Code: remaining leftovers from the original EBT codebase (done 2026-11-16, except where noted)

Same procedure as d5bbf69 (remove, then old-vs-new regression test: identical on all 12 demo checkpoints
and 4 training setups).

- [x] `model/model_utils.py`: image/video/text helpers removed (`modulate` kept: used by `ar_ebt_adaln.py`);
      `diffusers` import dropped (still in requirements.txt, harmless).
- [x] `model/diffusion/`, `model/diffusion_transformer.py`, `model/bi_ebt_adaln.py` removed.
- [x] `base_model_trainer.py`: video branch in `log_metrics` removed (`self.to_pil` kept for the image branch).
- [x] `docs/CODE_INFO.md` and `docs/CODE_STRUCTURE.md` updated for the music-only repo.
- [ ] **Left as is on purpose:** `train_model.py` argparse flags for image/video/text. Old checkpoints store
      these hparams; removing them gains nothing and risks breaking loading.
- [ ] **Your decision:** `MUS_NEUR` / `model/mus/ebt_neural.py` (neural-audio EBT, never used in the thesis):
      keep as future work or remove.

## 4. Code: optional reorganisation (decided against, 2026-11-16)

- `attribute_control/` data files: the thesis (experiments.tex) and the demo cite
  `attribute_control/corpus_stats.json` by path; moving it would break the submitted thesis's reference.
- `eval/` split: the thesis cites `eval/...` script paths. Same reason.
- `demo/app.py` split: only worth it if the demo moves (see 6).

Other repos (tonic, listening-study, listening-archive, thesis, anticipation-musicEBT): nothing to do.
Keep anticipation-musicEBT close to upstream.

## 5. Cluster: further tidy (2026-11-16)

- [x] `checkpoints/`: removed 196 run folders that held no checkpoint, only `wandb_run_id.txt`
      (from auto-requeued jobs). Their folder → W&B run ID mapping is saved in
      `logs/empty_ckpt_folders_wandb_ids_2026-11-16.tsv`. 56 folders / 155 checkpoints remain, none deleted.
- [ ] **Kept on purpose:** the 155 checkpoints (thesis runs' intermediate saves, token-matched runs, early
      baselines, probes). Deleting is irreversible and space is not a concern. If ever needed: keep the 7 thesis
      checkpoints, the 5 demo extras and the "Also on disk" saves in STATUS.md; never pick by `valid_loss`.
- [ ] **Kept on purpose:** W&B local folders (993 runs, all online `run-*`, none offline). Confirming each
      upload completed would need per-run W&B queries; the local copies are the fallback if one didn't.
- [ ] Optional: delete `~/transfer/backup_2026-11-16/` (verified copy on the Mac since 2026-11-16).

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
