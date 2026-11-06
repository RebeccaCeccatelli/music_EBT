# Blind listening survey

A static survey page served with the showcase on GitHub Pages at
`/music_EBT/survey/` (not linked from the showcase, so participants aren't
primed by the hand-picked examples). Source: `demo/showcase/site/survey/`.

An Italian version lives at `/music_EBT/survey/it/` (`site/survey/it/`). It
reuses the same `survey.json`, audio and `config.js` endpoint, so responses go
to the same sheet; they carry `"lang": "it"` and show as `v4 (it)` in the
`version` column (after redeploying `apps_script.gs`). Background answers are
stored with the English values, so the analysis treats both languages alike.
`it/survey.js` is a translated copy of `survey.js`: mirror any logic change.

## v5 (2026-11-06): curated by the author
Built with `demo/survey/build_survey_v5.py` (`sbatch job_scripts/mus/demo/build_survey_v5.sh`) from the clips
picked by ear in the picker pages (snapshot `music_EBT_logs/survey/picks_v5/`). Baselines are the EBT-recipe
Llama / GPT-2. Part 1: picked sets (original + EBT / Llama / GPT-2). Part 2: every picked steered clip (any
strength or target size) vs its system's own unguided reference, scored on direction. Part 3: per prompt,
attribute and direction, the picked clips of each system (2–4 versions). This is a qualitative study of
curated examples, not a controlled method comparison (the v4 design below). Trials sharing a `ukey` (same
prompt, or same prompt + attribute + direction) are never given to one participant twice. Key:
`music_EBT_logs/survey/key_v5.json` (keep `key.json` for v4 responses).

## Design (v4)
- **Part 1 · Unguided:** 4 blinded continuations of one prompt (EBT, Llama,
  GPT-2, and the human original as a hidden anchor), shown as Version A–D
  in random order. Listeners pick the most and the least musical.
- **Part 2 · Perceived change:** a system's own unguided output (reference)
  vs. the same system steered ±2 SD at its operating point. Five-point
  answer: "clearly softer … no difference … clearly louder" (note length /
  pitch for the other attributes). Measures whether the steering is audible.
- **Part 3 · Comparing versions:** the steered outputs of all systems for one
  prompt, attribute and direction. Listeners pick the most and the least musical.
- **Optional cacophony ticks:** every question ends with "did any version sound
  cacophonous?". These are per-version ticks, stored as `answer.cacophonous`.
- **Navigation:** Back to any earlier question (the answer is restored and replaced on
  resubmit). Every question can be skipped (`answer.skipped`).
- **Attention checks:** Part 1 includes a near-random EBT clip (λ far past the
  operating point), which should be picked as least musical. Part 2 includes a
  reference compared with itself, which should get "no difference".
- Systems (v5, EBT-recipe baselines): REMI has EBT, Llama PPLM, Llama tilt, Llama best-of-N,
  GPT-2 tilt and GPT-2 best-of-N; Anticipation has EBT, Llama PPLM, Llama best-of-N and GPT-2
  best-of-N (tilt is REMI-only). GPT-2 added 2026-11-06 at the user's request, for every attribute.
- Each participant gets a random subset (`PLAN` in `build_survey.py`):
  6 + 1 unguided, 12 + 1 change, 5 compare. That is ~25 trials, ~20 min.
  The sampling is round-robin over tokenizer × attribute × system.
- Answers unlock only after ≥70% of every clip has been heard. Listening time
  and time per trial are recorded.

## Blinding and selection
- Audio files are named by a salted hash. The key (hash → system, prompt,
  settings, metrics) lives at
  `~/orcd/scratch/rebcecca/music_EBT_logs/survey/key.json`, outside the public
  repo. **Keep it**: without it the responses can't be decoded.
- Prompts are picked by a fixed rule, without listening: the sweep prompts in
  `Random(0)` order, keeping those whose unguided continuations are ≥3 s, have
  ≥10 pitched notes, and differ across EBT / Llama / GPT-2. Every system uses
  its first draw (r0) at its operating point.

## Build
```bash
python demo/survey/build_survey.py pool      # inspect the pool (no rendering)
sbatch job_scripts/mus/demo/build_survey.sh  # render ~200 clips + survey.json + key
```
To publish a changed survey, rebuild, bump `--version` if trials changed
(stored with each response), then commit and push `demo/showcase/site/survey/`.

## Collecting responses
1. Create a Google Sheet, then **Extensions → Apps Script**. Paste
   `apps_script.gs` and save.
2. **Deploy → New deployment → Web app**: execute as *Me*, access *Anyone*.
   Copy the web-app URL.
3. Put the URL in `demo/showcase/site/survey/config.js` (`endpoint`). Optionally
   add a contact email there. Commit and push.
4. Each submission becomes one row in the `responses` sheet and is emailed
   to the account the script runs as (set `OWNER_EMAIL` to send elsewhere).
   It includes a readable summary and the raw JSON as an attachment. On the first
   deploy, Google asks to authorize sending email: allow it.
5. Participants can tick "Email me a copy". Their address is sent along only for
   that email: the script removes it before storing, so the sheet stays anonymous.
   Gmail accounts can send ~100 emails/day via Apps Script; each submission
   uses 1 (2 with a copy).
6. After editing the script: **Deploy → Manage deployments → Edit → New version**
   (the URL stays the same).

Without an endpoint the survey runs in test mode: the last screen downloads
the answers as JSON. Use this for piloting.

## Analysis
Export the sheet as CSV, then:
```bash
python demo/survey/analyze_survey.py responses.csv
```
This prints best-worst scores (Parts 1 and 3) and perceived change (Part 2) per
system with participant-bootstrap CIs. It excludes participants who fail an
attention check (`--keep-failed-catch` keeps them).

## Before sending the link
- Ethics: check with your supervisor whether MIT COUHES / ETH ethics review
  (likely an exemption for an anonymous listening study) is needed.
- Pilot with 2–3 people in test mode; check timing (~20 min) and wording.
