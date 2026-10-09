# Static showcase (listening examples)

A non-interactive page with selected outputs: EBT vs GPT-2 vs Llama
continuations, EBT vs Llama PPLM attribute steering, and an over-steering
example. Published to GitHub Pages by `.github/workflows/showcase-pages.yml`
on every push to `main` that touches `site/`.

All clips are existing outputs of the final guidance sweeps (same 16 REMI
validation prompts for every model); nothing is generated here.

```bash
PY=~/.conda/envs/music_EBT/bin/python
$PY demo/showcase/build_showcase.py candidates   # per-prompt table to choose clips from
# edit demo/showcase/selection.json (prompts, clips, captions)
$PY demo/showcase/build_showcase.py build        # MP3s + site/data.json
cd demo/showcase/site && $PY -m http.server 8000 # preview at localhost:8000
```

Page structure (`selection.json` is a tree: section → tokenizer tab →
attribute/combination tab → blocks):
1. Unguided Generation: REMI / Anticipation; original vs EBT, Llama, GPT-2.
2. Single-Attribute Control: REMI / Anticipation → velocity, duration,
   pitch register. A `steer` block (tilt, best-of-16, PPLM on Llama:
   down / unguided / up) and an `intensity` block (EBT: λ rows × target columns).
3. Combined Control: same skeleton, all placeholders for now.
4. Pushing Too Hard: EBT pitch register at operating vs high λ, per tokenizer.

Block types: `clips` (explicit list), `steer` (`rows` of method/model/strength),
`intensity` (`lambdas` × `sds`), `note` (text card). Clip specs:
`kind` = ground_truth | unguided (`model`) | guided (`method`, `model`,
`attribute`, `sd`, `strength`; for EBT the strength is λ). Strengths are
written as in the file names (`pplm16`, `tilt1`, `best_of_n16`, `r30.02`).
`"placeholder": true` on a clip, row or block plays a stand-in clip tagged
PLACEHOLDER. A slot whose output doesn't exist also falls back to it and is
listed at the end of `build`. Run locations are in `RUNS` in
`build_showcase.py`.

- `build` renders with FluidSynth + MuseScore General at a fixed gain
  (no loudness normalisation, so velocity steering stays audible) and
  re-renders a clip only when its source MIDI changes (`site/audio/sources.json`).
  A full fresh render (~120 clips) takes ~10 min.
- Displayed metrics are computed on the continuation only: mean velocity,
  mean pitch, mean note length (beats); harsh dissonance comes from the
  sweep's `quality_scores.csv`.
