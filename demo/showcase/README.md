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

- `build` renders with FluidSynth + MuseScore General at a fixed gain
  (no loudness normalisation, so velocity steering stays audible) and
  re-renders a clip only when its source MIDI changes (`site/audio/sources.json`).
- Clip kinds in `selection.json`: `ground_truth`, `baseline` (`model`:
  ebt/gpt2/llama, `draw` 0-2), `ebt` and `pplm` (`attribute`, `sd` in
  ±0.5/1/2; `ebt` takes an optional `lambda`, default = operating point).
- Displayed metrics are computed on the continuation only: mean velocity,
  mean pitch, mean note length (beats); harsh dissonance comes from the
  sweep's `quality_scores.csv`.
