# 2026-11-11 — By-ear EBT operating points from the showcase picks

**Question.** Which EBT (Tonic) guidance strength λ sounds best for each
attribute, according to the clips the author picked by ear for the showcase?

**Method.** Tallied the λ of every EBT clip in the Single-Attribute Control
section of `demo/showcase/selection.json` (all `ladder` blocks, i.e. by-ear
picks, one per cell). The clip ids encode tokenizer, attribute, prompt, λ and target
(`cur_<tok>_ebt_<attr>_<prompt>_steered_<λ>_<u|d><SD>_…`). Cross-checked with
the earlier by-ear notes (diary 2026-09-23, 2026-11-06, 2026-11-09;
`demo/survey/ear_operating_points.json`).

## Result

| Tokenizer · attribute | Showcase picks (λ: count) | Prompts | Best operating point | Earlier note | Held-out rule (thesis) |
|---|---|---|---|---|---|
| REMI · velocity | 0.02: 4, 0.03: 5, **0.04: 8**, 0.05: 2 | 5 | **0.04** (band 0.03–0.04) | 0.03–0.04 | 0.05 |
| REMI · duration | 0.015: 7, **0.02: 10**, 0.025: 8, 0.03: 5 | 6 | **0.02** (band 0.02–0.025) | 0.02–0.025 | 0.055 (per-attribute) |
| REMI · pitch register | 0.01: 10, 0.02: 11 | 5 | **0.01 lower / 0.02 higher** | 0.02 | 0.01 |
| Ant · duration | 0.004: 7, **0.005: 11**, 0.006: 5 | 7 | **0.005** | 0.005 | 0.005 |
| Ant · pitch register | **0.005: 13**, 0.006: 9, 0.008: 6, 0.01: 2, 0.012: 4 | 8 | **0.005 higher / 0.006 lower** (up to 0.012 for −2 SD) | 0.005–0.006 | 0.012 |

Held-out column from diary 2026-11-05/09/11; the source is
`music_EBT_logs/music_quality/sweeps/tune_{remi,ant_ar}.json` (in the logs dir, not the repo).

By direction (picks per target):
- **REMI pitch register** depends on direction: lower pitch (−1/−2 SD) 0.01 in
  7 of 10 picks; higher pitch (+0.5/+1/+2 SD) 0.02 in 8 of 11.
- **Ant pitch register**: higher pitch median 0.005 (8 of 13 picks at 0.005); lower
  pitch median 0.006, and −2 SD pulls further (3 of 9 at 0.012).
- **REMI velocity**: louder median 0.04 (5 of 10 at 0.04); softer (−2 SD) median 0.03,
  spread 0.02–0.05. Kept as a single 0.04.
- **Ant duration**: longer notes (+2 SD) lean to 0.004 (5 of 8); shorter around
  0.005. Kept as a single 0.005.
- **REMI duration** is the least settled: every λ 0.015–0.03 picked; 0.02
  most frequent, 0.025 close behind.

## Use

- `demo/survey/ear_operating_points.json` updated (REMI pitch adds 0.01; new
  `demo_defaults` block).
- Interactive demo (`demo/app.py`, `_LAMBDA_RANGES_BY_TOK`) defaults set to the
  bold values; for pitch register the EBT λ follows the target's direction
  (below the unguided reference → "down" value) unless the user moved λ by hand.

## Caveats

Picks were made on the same few prompts the showcase shows (5–8 per
attribute), by one listener. Listening reference only: the thesis accuracy
numbers keep the held-out operating points.
