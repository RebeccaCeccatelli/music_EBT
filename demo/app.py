#!/usr/bin/env python3
"""
Gradio demo for symbolic music generation — EBT, Llama, and GPT-2 side by side,
plus EBT attribute control (single-attribute and composed).

Launch (interactive GPU session):
    srun --gpus=1 --mem=40GB --time=2:00:00 --partition=mit_normal_gpu \
         --account=mit_general --qos=normal --pty bash
    cd ~/music-EBT && conda activate music_EBT
    python demo/app.py [--share]   # --share creates a public HuggingFace tunnel

Local access (from laptop, replace NODE with the compute node name):
    ssh -L 7860:NODE:7860 rebcecca@eosloan.mit.edu
    open http://localhost:7860
"""

import os

# This HPC node reports os.cpu_count() as the whole physical node (hundreds
# of cores), but each user's own cgroup is capped at a small CPU quota by
# cluster policy (see `systemctl cat user-$(id -u).slice` -> CPUQuota) —
# left unset, PyTorch/OMP/MKL each default their own thread pool to the
# node's full core count and try to spawn one thread per core, all of which
# then get squeezed into that tiny quota. That's not just slower, it's
# actual thrashing (confirmed via a "libgomp: Thread creation failed:
# Resource temporarily unavailable" error while profiling) — the fix is
# telling every thread-pool library the real number of cores this process
# actually gets, which must happen before those libraries are imported
# since several of them size their pool at import/first-use time. Override
# via DEMO_NUM_THREADS for a differently-provisioned environment.
_NUM_THREADS = int(os.environ.get("DEMO_NUM_THREADS", "4"))
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, str(_NUM_THREADS))

import re
import sys
import json
import time
import random
import tempfile
import traceback
from pathlib import Path
from argparse import Namespace

import torch
torch.set_num_threads(_NUM_THREADS)
import gradio as gr
from gradio_clickaudio import ClickAudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRATCH_DIR = Path.home() / "orcd/scratch/rebcecca/music_EBT_logs"
PROMPTS_FILE = ROOT / "demo/selected_prompts.json"
WANDB_PROJECT = "music_inference_ebt"
WANDB_ENTITY  = "rceccatelli-eth-z-rich"

RANDOM_SONG_LABEL = "🎲 Random song"
PROMPT_LENGTH = 256
MAX_STEPS = 100_000

REMI_TOKENIZER_CONFIG = "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json"

# ── Tokenizer format ─────────────────────────────────────────────────────────

TOKENIZER_CHOICES = ["REMI", "Anticipation-Arrival-Time"]
_TOK_SLUG = {"REMI": "remi", "Anticipation-Arrival-Time": "ant-at-full"}


def _tokenizer_config_path(tokenizer_type: str) -> str | None:
    return REMI_TOKENIZER_CONFIG if tokenizer_type == "REMI" else None


# ── Model registry ────────────────────────────────────────────────────────────

_MODEL_KEYS = ["EBT", "Llama", "GPT-2"]
_MODEL_INFO = {
    "EBT":    {"model_name": "ebt",                          "ckpt_pattern_tmpl": "ebt-symb-small-{slug}-s1*"},
    "Llama":  {"model_name": "baseline_llama_transformer",   "ckpt_pattern_tmpl": "baseline-llama-small-{slug}*"},
    "GPT-2":  {"model_name": "baseline_hf_gpt2_transformer", "ckpt_pattern_tmpl": "baseline-hf-gpt2-small-{slug}*"},
}


def _ckpt_pattern(model_key: str, tokenizer_type: str) -> str:
    return _MODEL_INFO[model_key]["ckpt_pattern_tmpl"].format(slug=_TOK_SLUG[tokenizer_type])


# ── Song list ─────────────────────────────────────────────────────────────────

def _clean_song_title(raw: str) -> str:
    """Song titles come straight from this crowd-sourced MIDI archive's raw
    filenames, which mix wildly different conventions in the same list —
    ALL CAPS, "Title * Artist", whole-title wrapped in quotes, stray tabs/
    runs of spaces, decorative "===" markers — reported as looking
    inconsistent in the dropdown. This is deliberately conservative: it only
    touches whitespace, a leading wrapped quote, and the "*"-as-separator
    convention. The data is too inconsistent (some use " - ", some "by
    Artist", some neither) to safely parse into one strict "Title — Artist"
    format without risking mangling titles that don't follow that pattern."""
    title = re.sub(r'\s+', ' ', raw).strip()
    title = title.strip('=').strip()
    # A title that OPENS with a quote almost always uses it to wrap just the
    # title itself, even when trailing text follows (e.g. a "(theme)" or a
    # translated alternate title) — strip that pair, keep everything else.
    m = re.match(r'^(["\'])(.+?)\1(.*)$', title)
    if m:
        _, inner, rest = m.groups()
        title = (inner + (' ' + rest.strip() if rest.strip() else '')).strip()
    title = re.sub(r'\s*\*\s*', ' — ', title)
    return title


_song_display_to_raw: dict[str, str] = {}


def _load_song_choices() -> list[str]:
    global _song_display_to_raw
    try:
        with open(PROMPTS_FILE) as f:
            songs = json.load(f)
    except Exception:
        return [RANDOM_SONG_LABEL]
    cleaned: dict[str, str] = {}
    for raw_key in songs.keys():
        display = _clean_song_title(raw_key)
        if display in cleaned:  # collision between two raw keys — rare, but
            display = f"{display} ({raw_key})"  # keep both reachable
        cleaned[display] = raw_key
    _song_display_to_raw = cleaned
    return [RANDOM_SONG_LABEL] + sorted(cleaned.keys())


def _song_midi_path(song_title: str) -> str | None:
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    entry = prompts.get(_song_display_to_raw.get(song_title, song_title))
    if entry is None:
        return None
    return entry["path"] if isinstance(entry, dict) else entry


_anticipation_compound_cache: dict[str, list[int]] = {}


def _midi_to_compound_cached(midi_path: str) -> list[int]:
    """MIDI -> Anticipation's raw 'compound' representation: flat
    (time, duration, note, instrument, velocity) quintuples in absolute
    ticks at TIME_RESOLUTION bins/sec from the file's start. Unlike the
    final vocab-mapped 'events' format, compound has no upper bound on
    absolute time — it's the safe intermediate for a whole song of any
    length (see _anticipation_window_tokens for why "events" alone isn't)."""
    if midi_path not in _anticipation_compound_cache:
        import sys
        anticipation_root = str(ROOT / "data/mus/symbolic/tokenization/anticipation")
        if anticipation_root not in sys.path:
            sys.path.insert(0, anticipation_root)
        from anticipation.convert import midi_to_compound
        _anticipation_compound_cache[midi_path] = midi_to_compound(midi_path)
    return _anticipation_compound_cache[midi_path]


def _anticipation_compound_time_map(midi_path: str) -> list[tuple[int, float]]:
    """[(token_index, cumulative_seconds)] for Anticipation, built directly
    from compound's own absolute time field for every note onset — exact,
    no decode() calls, and immune to the "events" vocab format's 100-second
    absolute-time cap (MAX_TIME_IN_SECONDS) since compound never offsets or
    caps at all. token_index uses *3 to stay in the same triplet-based units
    as the rest of the pipeline (_align_to_triplet, the offset slider's
    step=3, etc.), even though this map is built pre-triplet-conversion."""
    compound = _midi_to_compound_cached(midi_path)
    _, time_resolution = _anticipation_vocab()
    n_notes = len(compound) // 5
    return [(i * 3, compound[i * 5] / time_resolution) for i in range(n_notes)]


def _anticipation_window_tokens(midi_path: str, start_seconds: float, window_seconds: float = 60.0) -> list[int]:
    """A freshly-tokenized, locally-rebased Anticipation 'events' token
    window starting at start_seconds.

    Anticipation's vocabulary has a hard 100-second cap on absolute arrival
    time (MAX_TIME_IN_SECONDS in anticipation/config.py) — any single event
    beyond that overflows into the next token sub-range's numeric space,
    which doesn't just truncate cleanly: it silently corrupts every triplet
    after the overflow point (confirmed on a real 104-second song — filtered
    as "malformed" by the decoder's own range check). A whole song longer
    than 100s can't safely be tokenized as one absolute-time stream, which
    is exactly what naive whole-song tokenization did before this.

    The fix mirrors how real (<100s) training windows are built: pull out
    just the notes in [start_seconds, start_seconds + window_seconds) at the
    uncapped compound level, rebase their times to start at 0 (anticipation's
    own ops.clip()+translate() combination, applied here directly since we
    already have compound in hand), THEN convert just that short, rebased
    excerpt to vocab-mapped events — which, starting near 0, comfortably
    stays under the cap regardless of where in the song it began."""
    import sys
    anticipation_root = str(ROOT / "data/mus/symbolic/tokenization/anticipation")
    if anticipation_root not in sys.path:
        sys.path.insert(0, anticipation_root)
    from anticipation.convert import compound_to_events

    compound = _midi_to_compound_cached(midi_path)
    _, time_resolution = _anticipation_vocab()
    start_ticks = round(start_seconds * time_resolution)
    end_ticks = start_ticks + round(window_seconds * time_resolution)

    window = []
    for i in range(0, len(compound), 5):
        t, dur, note, instr, vel = compound[i:i + 5]
        if t < start_ticks or t >= end_ticks:
            continue
        window.extend([t - start_ticks, dur, note, instr, vel])

    if not window:
        return []
    return compound_to_events(window)


def _tokens_for_song(song_title: str, prompts: dict) -> list[int]:
    """Load the full token id list for a named song from its pre-tokenized
    REMI JSON — the only tokenizer with a per-song pre-tokenized cache.
    Anticipation songs are tokenized fresh, locally, around the requested
    start time instead (see _anticipation_window_tokens); a whole-song
    tokenization isn't safe for it (see that function's docstring)."""
    entry = prompts[_song_display_to_raw.get(song_title, song_title)]
    midi_path = entry["path"] if isinstance(entry, dict) else entry
    token_path = midi_path.replace("/midi/", "/tokens/miditok/").replace(".mid", ".json")
    with open(token_path) as f:
        return json.load(f)["ids"]


def _song_start_hint(song_title: str, prompts: dict) -> int:
    entry = prompts.get(_song_display_to_raw.get(song_title, song_title), {})
    return entry.get("start_token", 0) if isinstance(entry, dict) else 0


_SONG_CHOICES = _load_song_choices()

# ── Checkpoint discovery (hardened: highest step wins, not lowest loss) ──────

def _find_checkpoints(ckpt_pattern: str) -> list[tuple[str, str, int, float]]:
    """Return [(display_label, ckpt_path, step, loss)] sorted best-first by
    file modification time (most recent wins), NOT by loss or raw step count.

    Two failure modes were confirmed this session and both rule out the more
    "obvious" sort keys:
      - Loss: a checkpoint's filename loss can be a first-post-resume-
        validation artifact (every restart across GPT-2/Llama/EBT produced
        one spurious, anomalously-good reading) — "lowest loss anywhere"
        could silently default to exactly one of those.
      - Raw step count: this codebase has many abandoned experimental
        lineages sharing the same run-name pattern (e.g. a pre-stabilization
        EBT run with no Langevin noise/replay buffer trained further, by
        step count, than the actively-worked-on stabilized run before being
        abandoned) — "highest step anywhere" can resurface a stale,
        unrelated lineage instead of the run actually being iterated on.
    Most-recently-modified directly answers "what am I actually working on
    right now," which is what a demo's default should show.
    """
    ckpt_root = SCRATCH_DIR / "checkpoints"
    if not ckpt_root.exists():
        return []

    best: dict[str, tuple[float, int, float, str]] = {}  # run_dir -> (mtime, step, loss, ckpt_path)
    for run_dir in sorted(ckpt_root.glob(ckpt_pattern)):
        if not run_dir.is_dir():
            continue
        for ckpt in run_dir.glob("epoch=*.ckpt"):
            step_m = re.search(r'step=(\d+)', str(ckpt))
            if not step_m:
                continue
            loss_m = re.search(r'valid_loss=(\d+\.\d+)', str(ckpt))
            step = int(step_m.group(1))
            loss = float(loss_m.group(1)) if loss_m else float('nan')
            mtime = ckpt.stat().st_mtime
            key = str(run_dir)
            if key not in best or mtime > best[key][0]:
                best[key] = (mtime, step, loss, str(ckpt))

    results = sorted(best.values(), key=lambda x: x[0], reverse=True)
    return [(_ckpt_label(path, step, loss), path, step, loss) for mtime, step, loss, path in results]


def _ckpt_label(ckpt_path: str, step: int, loss: float) -> str:
    run_dir = Path(ckpt_path).parent.name
    date_m = re.search(r'(\d{4}-\d{2}-\d{2})', run_dir)
    date = f" · {date_m.group(1)}" if date_m else ""
    pct = step / MAX_STEPS * 100
    loss_str = f"{loss:.4f}" if loss == loss else "n/a"  # loss != loss iff NaN
    return f"step {step:,} ({pct:.1f}%) · val_loss {loss_str}{date}"


def _ckpt_progress_html(step: int, loss: float, tokenizer_type: str = "") -> str:
    # Naming the tokenizer format in the title itself (not just relying on
    # the numbers changing) is what actually makes a format switch visible —
    # without it, "Stats at this checkpoint" read identically before and
    # after switching tokenizer, so a successful switch and a silently
    # failed one looked the same at a glance.
    pct = min(step / MAX_STEPS * 100, 100)
    loss_str = f"{loss:.4f}" if loss == loss else "n/a"  # loss != loss iff NaN
    title = f"Stats at this {tokenizer_type} checkpoint:" if tokenizer_type else "Stats at this checkpoint:"
    return f"""
<div class="reference-box">
  <div style="font-weight:600;margin-bottom:6px">{title}</div>
  <div style="display:flex;flex-direction:column;gap:4px;font-size:.95em">
    <div>Training progress &nbsp;<b style="color:var(--primary-500)">{pct:.1f}%</b></div>
    <div>Validation loss &nbsp;<b style="color:var(--primary-500)">{loss_str}</b></div>
  </div>
</div>"""


def _ckpt_loading_html(tokenizer_type: str) -> str:
    return (f'<div class="reference-box ckpt-loading">'
            f'<span class="ckpt-spin">↻</span> Loading {tokenizer_type} checkpoints…</div>')


def _ckpt_stats_for_selection(label: str, tokenizer_type: str) -> str:
    """Recompute the stats card for whichever checkpoint is currently picked
    in the dropdown. Previously this card only ever showed the auto-picked
    checkpoint's stats from the last rescan — manually choosing a different
    one (see "Manually choose checkpoints" below) didn't update it, so it
    kept showing numbers for a checkpoint you weren't actually about to use."""
    if not label:
        return ""
    step_m = re.search(r'step ([\d,]+)', label)
    loss_m = re.search(r'val_loss ([\d.]+)', label)
    step = int(step_m.group(1).replace(',', '')) if step_m else 0
    loss = float(loss_m.group(1)) if loss_m else float('nan')
    return _ckpt_progress_html(step, loss, tokenizer_type)


def refresh_checkpoints(model_key: str, tokenizer_type: str, current_selection: str | None = None) -> tuple:
    """The checkpoint dropdown itself is tucked away as a manual-override
    option (see the "Manually choose checkpoints" accordion) — by default
    the most recent checkpoint is used automatically, and this instead
    surfaces just enough for the user to gauge training progress: percent of
    max steps reached and the checkpoint's validation loss.

    Re-scans disk for checkpoints (picking up new ones saved since the last
    look) and keeps whatever is currently selected if it's still a valid
    choice — only falling back to the most recent checkpoint when there's no
    prior selection to keep, or that selection no longer exists (e.g. after
    switching tokenizer, where the old label belongs to a different
    checkpoint set entirely). A genuine rescan should update the list
    without silently discarding a deliberate manual pick — always jumping
    back to the most recent checkpoint would really be a reset, not a
    rescan."""
    pairs = _find_checkpoints(_ckpt_pattern(model_key, tokenizer_type))
    if not pairs:
        return (
            gr.update(choices=[], value=None, label=f"{model_key} {tokenizer_type} checkpoint", interactive=True),
            "",
            f'<div class="reference-box">No {model_key} checkpoints found for {tokenizer_type} yet.</div>',
        )
    labels = [lbl for lbl, _, _, _ in pairs]
    paths  = {lbl: path for lbl, path, _, _ in pairs}
    by_label = {lbl: (step, loss) for lbl, _, step, loss in pairs}
    if current_selection in by_label:
        chosen = current_selection
    else:
        chosen = labels[0]
    chosen_step, chosen_loss = by_label[chosen]
    return (
        gr.update(choices=labels, value=chosen,
                  label=f"{model_key} {tokenizer_type} checkpoint ({len(labels)} found)",
                  interactive=True),
        json.dumps(paths),
        _ckpt_progress_html(chosen_step, chosen_loss, tokenizer_type),
    )


# ── Attribute regressor discovery ────────────────────────────────────────────

_ATTRIBUTE_UI = {
    "density":  {"max": 0.30, "default_target": 0.15,
                 "label": "Target density T  (0 = sparse · 0.30 = dense)"},
    "velocity": {"max": 1.00, "default_target": 0.70,
                 "label": "Target velocity T  (0 = soft · 1.0 = loud)"},
    "duration": {"max": 1.00, "default_target": 0.08,
                 "label": "Target note length T  (0 = staccato · 1.0 = sustained)"},
    "pitch_register": {"max": 1.00, "default_target": 0.50,
                        "label": "Target pitch register T  (0 = low/bass · 1.0 = high/treble)"},
}
# Only velocity/duration/pitch_register are exposed in the UI — density's
# solo control was confirmed unreliable this session. Left in the dict in
# case it's ever needed again, just not offered as a choice below.
_SOLO_ATTRIBUTES = ["velocity", "duration", "pitch_register"]


def _solo_attribute_choices(tokenizer_type: str) -> list[str]:
    """Velocity has no meaning under Anticipation — its vocabulary strips
    velocity out before the model ever sees it (see compute_velocity's
    docstring), a structural limitation, not a missing regressor — so it
    isn't offered as a pickable choice there at all, rather than being
    pickable and then failing with an error after the fact. Pitch register,
    unlike velocity, IS recoverable under Anticipation (its note field packs
    pitch alongside instrument — see compute_pitch_register), so it's offered
    for every tokenizer."""
    if tokenizer_type == "REMI":
        return _SOLO_ATTRIBUTES
    return [a for a in _SOLO_ATTRIBUTES if a != "velocity"]


def _attr_choice_tuples(attrs: list[str]) -> list[tuple[str, str]]:
    """(display_label, value) pairs for a Radio's `choices` — the value stays
    the raw attribute name (matching _ATTRIBUTE_UI/_LAMBDA_RANGES/ATTRIBUTES
    keys everywhere else), only the on-screen label swaps underscores for
    spaces (e.g. "pitch_register" -> "pitch register")."""
    return [(a.replace('_', ' '), a) for a in attrs]

# Working λ ranges from the comprehensive REMI listening sweeps (16 prompts,
# 768 guided samples/attribute — see docs/thesis_findings and the "REMI
# Guidance Sweeps" artifact for the full accuracy/error/coherence curves),
# refined further by ear. Duration needs roughly 2-3x smaller λ than
# velocity/pitch_register since it guides a different, non-overlapping gate
# in this vocabulary (velocity on Pitch→Velocity steps, duration on
# Velocity→Duration steps, pitch_register on Program→Pitch steps).
_LAMBDA_RANGES = {
    "velocity": {"min": 0.02, "max": 0.07, "default": 0.03, "step": 0.005},
    "duration": {"min": 0.015, "max": 0.04, "default": 0.025, "step": 0.005},
    "pitch_register": {"min": 0.01, "max": 0.07, "default": 0.02, "step": 0.005},
}


def _card_classes(visible: bool) -> list[str]:
    """elem_classes for a "model-card" gr.Group whose show/hide is driven by
    a CSS class rather than Gradio's own `visible` prop — that prop is
    unreliable on gr.Group in this Gradio version (hiding leaves an empty,
    still-styled box behind; confirmed via an isolated repro with zero app
    logic), and on gr.Column too (reveal delayed by a full click; also
    confirmed in isolation). Toggling elem_classes instead sidesteps both."""
    return ["model-card"] if visible else ["model-card", "hidden-variant"]


def _target_default_value(attribute: str, ref_velocity: float | None, ref_duration: float | None,
                           ref_pitch_register: float | None = None) -> float:
    """Before the user deliberately moves the target slider, it should read
    as "no push" — i.e. wherever EBT's own unguided generation already sits
    for this attribute (from the last main Generate run), not an arbitrary
    fixed number unrelated to what the model actually produces. Falls back
    to the old fixed default only if no reference exists yet (no EBT
    generation has run this session)."""
    ref = {"velocity": ref_velocity, "duration": ref_duration,
           "pitch_register": ref_pitch_register}.get(attribute)
    return ref if ref is not None else _ATTRIBUTE_UI[attribute]["default_target"]


_CORPUS_STATS_PATH = Path(__file__).parent.parent / "attribute_control" / "corpus_stats.json"
_corpus_stats_cache: dict | None = None


def _corpus_stats() -> dict:
    """Cached load of attribute_control/corpus_stats.json — mean/stdev of
    each attribute over a real training-data sample (see
    docs/thesis_findings/2026-09-24_corpus_attribute_distributions.md).
    Not tokenizer-specific (the file has no such split); used as-is for
    whichever tokenizer is active, same simplification the rest of the
    codebase already makes with this file."""
    global _corpus_stats_cache
    if _corpus_stats_cache is None:
        try:
            with open(_CORPUS_STATS_PATH) as f:
                _corpus_stats_cache = json.load(f)
        except Exception:
            _corpus_stats_cache = {}
    return _corpus_stats_cache


def _sigma_annotation_html(attribute: str, target: float) -> str:
    """How many corpus standard deviations a target sits from the real
    training-data mean for this attribute — the σ-based framing is already
    how the project's own sweep scripts (compose_attributes.py's
    --target_deltas_std) reason about targets internally; this surfaces the
    same thing live in the UI instead of requiring a separate lookup.
    Color tiers match _attribute_gauge_html's existing green/amber/red
    achieved-vs-target convention, reused here for "how extreme is this ask"
    instead of "how close did generation get." Duration and pitch_register
    are both meaningfully skewed (see the corpus-distributions finding), so
    this is a rule-of-thumb flag, not a precise boundary — the note says so
    explicitly rather than implying false precision."""
    stats = _corpus_stats().get(attribute)
    if not stats or not stats.get("stdev"):
        return ""
    z = (target - stats["mean"]) / stats["stdev"]
    az = abs(z)
    sign = "+" if z >= 0 else "−"
    if az < 1.0:
        color, note = "#22c55e", "within the training corpus"
    elif az < 2.0:
        color, note = "#f59e0b", "at the edge of the training corpus"
    else:
        color, note = "#ef4444", "far outside the training corpus"
    return (f'<div style="font-size:.76em;color:{color};margin-top:2px">'
            f'≈ {sign}{az:.1f}σ from corpus mean — {note}</div>')


def _find_attribute_regressor(attribute: str, tokenizer_type: str = "REMI") -> tuple[str | None, str]:
    """Return (ckpt_path, status) for the most recently trained EBT-specific
    regressor for this attribute AND tokenizer — a regressor is trained
    against one specific EBT checkpoint's embedding space (see
    train_density_regressor.py), and REMI vs. Anticipation EBT checkpoints
    have entirely different embedding spaces, so a regressor trained on one
    can't meaningfully guide the other. Filters out regressors trained on a
    baseline model's embeddings (e.g. the Llama velocity regressor built for
    this session's PPLM pilot) by checking each candidate's own saved
    ebt_checkpoint metadata — a regressor trained on Llama's embedding space
    would produce meaningless guidance applied to EBT's, even though the
    file naming pattern alone can't tell them apart."""
    if attribute == "velocity" and tokenizer_type != "REMI":
        # Not "hasn't been trained yet" — Anticipation's vocabulary strips
        # velocity out before the model ever sees it (every note gets a
        # fixed default velocity — see compute_velocity's docstring), so no
        # amount of training can produce a working velocity regressor here.
        return None, (f"Velocity control isn't available for {tokenizer_type}: its "
                       "tokenizer has no velocity information at all, so this can't be "
                       "trained regardless (not just missing yet).")
    slug = _TOK_SLUG.get(tokenizer_type, tokenizer_type.lower())
    regressor_root = SCRATCH_DIR / "attr_control"
    if not regressor_root.exists():
        return None, f"No {attribute} regressor found for {tokenizer_type} — train one first."
    candidates = sorted(
        regressor_root.glob(f"{attribute}_regressor_{slug}_*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in candidates:
        try:
            meta = torch.load(cand, map_location='cpu', weights_only=False)
        except Exception:
            continue
        src_ckpt = meta.get('ebt_checkpoint', '') or ''
        if 'ebt-symb' in src_ckpt:
            return str(cand), f"Regressor: {cand.parent.name}"
    if candidates:
        return None, (f"Found {attribute} regressor(s) for {tokenizer_type} but none were "
                       f"trained on an EBT checkpoint (only baseline-model regressors "
                       f"present) — train one on EBT.")
    return None, f"No {attribute} regressor found for {tokenizer_type} — train one first."


def _attribute_gauge_html(title: str, rows: list[tuple[str, float, float, float]]) -> str:
    """rows: list of (attribute_name, target, achieved, max_value)."""
    row_html = []
    for name, target, achieved, max_v in rows:
        t_pct = min(target / max_v * 100, 100) if max_v else 0
        a_pct = min(achieved / max_v * 100, 100) if max_v else 0
        diff = achieved - target
        tol = 0.02 * max_v
        color = "#22c55e" if abs(diff) < tol else "#f59e0b" if abs(diff) < 3 * tol else "#ef4444"
        sign = "+" if diff >= 0 else ""
        row_html.append(f"""
    <div style="margin-bottom:10px">
      <div style="font-size:.82em;color:var(--body-text-color-subdued,#888);margin-bottom:3px">{name.replace('_', ' ').capitalize()}</div>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="flex:1;background:#e5e7eb22;border-radius:6px;height:16px;position:relative;overflow:hidden">
          <div style="width:{t_pct:.1f}%;background:#6366f1;height:16px;position:absolute;opacity:.35"></div>
          <div style="width:{a_pct:.1f}%;background:{color};height:16px;position:absolute"></div>
        </div>
        <div style="font-size:.8em;white-space:nowrap;min-width:150px">
          target {target:.3f} → <span style="color:{color};font-weight:600">{achieved:.3f}</span> ({sign}{diff:.3f})
        </div>
      </div>
    </div>""")
    return f"""
<div style="font-family:inherit;padding:8px 4px">
  <b style="font-size:.9em">{title}</b>
  <div style="margin-top:10px">{''.join(row_html)}</div>
</div>"""



def _render_compose_reference_html(ref_velocity: float | None, ref_duration: float | None,
                                    ref_pitch_register: float | None = None) -> str:
    """Reference display shared by both the Single attribute (batch can mix
    velocity/duration/pitch_register rows, so all three are shown when
    available) and Compose sections (velocity + duration only): EBT's own
    unguided values from the last main Generate run. ref_pitch_register is
    optional and omitted from the text entirely under REMI-vs-Anticipation
    mixes where it wasn't measured, rather than showing a misleading 0."""
    if ref_duration is None:
        return ('<div class="reference-box">No reference yet — run <b>Generate</b> above '
                'with EBT included to compare against its unguided default.</div>')
    parts = []
    if ref_velocity is not None:
        parts.append(f'velocity: <b>{ref_velocity:.3f}</b>')
    parts.append(f'duration: <b>{ref_duration:.3f}</b>')
    if ref_pitch_register is not None:
        parts.append(f'pitch register: <b>{ref_pitch_register:.3f}</b>')
    return f'<div class="reference-box">Reference (unguided EBT) — {" · ".join(parts)}</div>'


# ── Model + tokenizer + dataset caches ───────────────────────────────────────

_model_cache: dict[str, tuple] = {}    # ckpt_path -> (model, hparams, tokenizer)
_dataset_cache: dict[str, object] = {} # tokenizer_type -> dataset


def _load(model_key: str, ckpt_path: str, device: str):
    if ckpt_path in _model_cache:
        return _model_cache[ckpt_path]

    model_name = _MODEL_INFO[model_key]["model_name"]
    if model_name == "ebt":
        from inference.mus.infer_ebt import load_checkpoint
        model, hparams = load_checkpoint(ckpt_path, device)
    else:
        from inference.mus.infer_baselines_interactive import load_checkpoint
        model, hparams = load_checkpoint(ckpt_path, model_name, device)

    from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
    tokenizer, _, pad_id = load_tokenizer(
        tokenizer_type=hparams.tokenizer_type,
        tokenizer_config_path=getattr(hparams, 'tokenizer_config_path', None),
        dataset_name=getattr(hparams, 'dataset_name', 'giga_midi'),
    )
    _model_cache[ckpt_path] = (model, hparams, tokenizer)
    return model, hparams, tokenizer


def _get_dataset(hparams):
    from inference.mus.infer_ebt import load_dataset as ebt_load_dataset
    key = hparams.tokenizer_type
    if key not in _dataset_cache:
        _dataset_cache[key] = ebt_load_dataset(hparams, split="validation")
    return _dataset_cache[key]


def _minimal_hparams(tokenizer_type: str) -> Namespace:
    return Namespace(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=_tokenizer_config_path(tokenizer_type),
        dataset_name="giga_midi",
        context_length=512,
        validation_split_pct=0.05,
    )


def _get_tokenizer(tokenizer_type: str):
    from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
    tokenizer, _, _ = load_tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=_tokenizer_config_path(tokenizer_type),
        dataset_name="giga_midi",
    )
    return tokenizer


# ── Token-index <-> seconds mapping (REMI named songs only) ─────────────────
#
# REMI has no fixed tokens-per-second rate (a dense passage packs more music
# into fewer tokens than a sparse one), so "seconds" can't be a simple linear
# relabeling of the token slider — it needs a real per-song mapping. Built by
# decoding growing token prefixes and reading each prefix's tempo-aware tick
# duration (score.end() / tpq * 60/qpm). That formula is internally correct —
# verified it exactly matches a from-scratch tempo calculation — but it
# doesn't match the REAL rendered audio: confirmed a real song where the tick
# math gives 62.75s while the actual FluidSynth-rendered WAV is 72.75s,
# almost certainly a release/reverb tail FluidSynth adds past the last note
# event, which no tick-based formula can predict. Since a trim UI needs the
# map to line up with what's actually heard, the raw tick-based map gets
# rescaled to match the real rendered duration (one WAV render per song,
# shared with render_full_song() below so it's not synthesized twice).
# Only applies to named REMI songs — the random-validation-sample path has no
# fixed sequence to build a map for, and keeps using a raw token offset.

_TIME_MAP_STEP = 32  # tokens between duration samples.
# NOTE: the slow part of building this map is NOT the decode() loop — measured
# individual decode() calls at ~5ms even at full song length, negligible even
# summed over ~160 calls. The real cost (confirmed ~11s on a ~73s song) is the
# one full-song FluidSynth render in _get_full_song_render, likely the same
# thread-contention issue as the audio-quality problem — both specific to this
# shared login node, not this loop. Don't "optimize" this step size again
# without re-profiling; it isn't the bottleneck.
_time_map_cache: dict[str, list[tuple[int, float]]] = {}
_full_song_render_cache: dict[str, tuple[str, float]] = {}  # "tokenizer::song" -> (wav_path, real_duration_s)


def _anticipation_vocab():
    import sys
    anticipation_root = str(ROOT / "data/mus/symbolic/tokenization/anticipation")
    if anticipation_root not in sys.path:
        sys.path.insert(0, anticipation_root)
    from anticipation.vocab_ant import TIME_OFFSET
    from anticipation.config import TIME_RESOLUTION
    return TIME_OFFSET, TIME_RESOLUTION


def _build_time_map(tokens: list[int], tokenizer) -> list[tuple[int, float]]:
    """[(token_index, cumulative_seconds)], monotonically increasing, tempo-aware
    but NOT yet corrected for FluidSynth's rendering tail — see _get_time_map.
    REMI-only: it has no directly-encoded absolute time, so this decodes
    growing token prefixes and reads each one's tempo-aware duration
    (sampled every _TIME_MAP_STEP tokens, not every token — decode() isn't
    free at full-song length). Anticipation's equivalent is
    _anticipation_compound_time_map, built completely differently (see its
    docstring for why a decode()-sampling approach like this one doesn't
    work for it)."""
    time_map = [(0, 0.0)]
    for k in range(_TIME_MAP_STEP, len(tokens) + 1, _TIME_MAP_STEP):
        try:
            score = tokenizer.decode(tokens[:k])
            qpm = score.tempos[0].qpm if score.tempos else 120.0
            time_map.append((k, score.end() / score.tpq * (60.0 / qpm)))
        except Exception:
            break
    return time_map


def _get_full_song_render(tokenizer_type: str, song_choice: str) -> tuple[str, float] | None:
    """Render a named song's full duration to WAV once, cached per
    (tokenizer_type, song_choice). Returns (wav_path, real_duration_seconds),
    or None if not applicable.

    Anticipation renders straight from the ORIGINAL MIDI file, not a
    tokenize-then-decode round trip — the round trip is exactly what
    _anticipation_window_tokens's docstring explains is unsafe for a whole
    song longer than 100 seconds (silently corrupts past that point), and
    the original file is already sitting right there, so there's no reason
    to risk it for a full-song render at all. REMI keeps the round trip
    since its own encoding has no such length cap and this also keeps the
    render consistent with what _build_time_map/_refine_token_index decode."""
    cache_key = f"{tokenizer_type}::{song_choice}"
    if cache_key in _full_song_render_cache:
        return _full_song_render_cache[cache_key]

    from demo.convert_midi_simple import simple_synth
    out_dir = Path(tempfile.mkdtemp(prefix="demo_fullsong_"))
    wav_path = out_dir / "full.wav"

    if _is_triplet_tokenizer(tokenizer_type):
        midi_path = _song_midi_path(song_choice)
        if midi_path is None:
            return None
        simple_synth(midi_path, str(wav_path))
    else:
        full_tokens, _, _ = _full_tokens_for_offset_range(tokenizer_type, song_choice)
        if full_tokens is None:
            return None
        tokenizer = _get_tokenizer(tokenizer_type)
        from inference.mus.tokens_to_midi import tokens_to_midi
        midi_path = out_dir / "full.mid"
        midi_path.write_bytes(tokens_to_midi(full_tokens, tokenizer))
        simple_synth(str(midi_path), str(wav_path))

    if not wav_path.exists():
        return None
    result = (str(wav_path), _wav_duration_seconds(str(wav_path)))
    _full_song_render_cache[cache_key] = result
    return result


def _get_time_map(tokenizer_type: str, song_choice: str) -> list[tuple[int, float]] | None:
    """[(token_index, seconds)] for the song, used AS-IS — no rescaling
    against the rendered WAV's duration.

    An earlier version stretched every entry by a constant
    `real_wav_duration / last_note_time` factor, on the theory that
    FluidSynth's render ran a bit long relative to the raw tick/tempo math
    (confirmed true — e.g. a real ~163s worth of notes rendering as a
    ~189s WAV). But that gap is a release/reverb TAIL after the last note
    finishes, not a uniform tempo drift across the whole piece — FluidSynth
    is a real synthesizer that respects the MIDI's own tempo throughout, it
    doesn't play the piece itself at some different, wrong speed. Stretching
    every note's time by that end-loaded ratio doesn't correct a real
    mismatch at each note — it manufactures one, and it grows with how far
    into the song you look: confirmed a click at 90s was landing the prompt
    at a note actually ~77s in (rescaled 1.16x), and at 150s landing ~21s
    early — a real, reported "prompt starts 1-2s before where I clicked"
    complaint, worse than that here since this song's tail happened to be
    especially long. Using raw note times directly means a click at second
    T lands on whatever note is really encoded at T, which is what
    FluidSynth actually plays there."""
    cache_key = f"{tokenizer_type}::{song_choice}"
    if cache_key in _time_map_cache:
        return _time_map_cache[cache_key]

    if _is_triplet_tokenizer(tokenizer_type):
        midi_path = _song_midi_path(song_choice)
        if midi_path is None:
            return None
        tmap = _anticipation_compound_time_map(midi_path)
    else:
        full_tokens, _, _ = _full_tokens_for_offset_range(tokenizer_type, song_choice)
        if full_tokens is None:
            return None
        tmap = _build_time_map(full_tokens, _get_tokenizer(tokenizer_type))

    _time_map_cache[cache_key] = tmap
    return tmap


def _seconds_to_token_index(time_map: list[tuple[int, float]], seconds: float) -> int:
    """Interpolate between the two time_map samples bracketing `seconds`,
    rather than flooring to the lower one. The map is only sampled every
    _TIME_MAP_STEP tokens (for performance — see its definition), so flooring
    made every click land up to _TIME_MAP_STEP tokens' worth of time (a
    fraction of a second, more in sparse passages) before where it was
    actually clicked — confirmed by direct user report."""
    prev_idx, prev_t = time_map[0]
    for idx, t in time_map[1:]:
        if t >= seconds:
            if t == prev_t:
                return idx
            frac = (seconds - prev_t) / (t - prev_t)
            return round(prev_idx + frac * (idx - prev_idx))
        prev_idx, prev_t = idx, t
    return prev_idx


def _refine_token_index(tokenizer_type: str, song_choice: str, coarse_idx: int, seconds: float) -> int:
    """Binary-search the exact token index over the whole song using live
    decode() calls, instead of the interpolated estimate. Interpolation
    assumes a constant token rate between the two bracketing samples, which
    isn't true in passages with uneven note density — the residual error is
    small but was still perceptible as a slight, consistent click offset.
    decode() costs ~5ms even at full song length (profiled earlier), so a
    ~13-call binary search here is cheap enough to do live, on click, for
    exact precision. `coarse_idx` is accepted but unused for REMI: it was an
    earlier, incorrect attempt to search only a window around it — several
    consecutive tokens can share the exact same cumulative time (one note
    event spans multiple tokens), forming a "plateau," and confirmed via
    direct testing that a window-local search could return a
    different-but-equally-valid representative of the same plateau depending
    on where the window started, breaking round-trip consistency (click →
    displayed offset → re-resolving that same displayed offset landing on a
    different token).

    Canonical rule: always return the FIRST token whose cumulative time
    reaches or exceeds `seconds` (a bisect_left over the monotonic
    real_time_at function, searched over the whole song so the result depends
    only on `seconds`, never on where the search started). This also matches
    the musically correct behavior — starting exactly at a note's onset
    rather than mid-note or at the tail of the previous one.

    Anticipation needs none of this: _get_time_map already built an EXACT
    per-triplet map (straight from each event's own encoded time, not
    interpolated decode() samples), so `coarse_idx` has no decode()-level
    error to correct — it just needs snapping to a clean triplet boundary."""
    if _is_triplet_tokenizer(tokenizer_type):
        return _align_to_triplet(coarse_idx)

    full_tokens, _, _ = _full_tokens_for_offset_range(tokenizer_type, song_choice)
    if full_tokens is None:
        return coarse_idx
    tokenizer = _get_tokenizer(tokenizer_type)

    def real_time_at(k: int) -> float:
        if k <= 0:
            return 0.0
        score = tokenizer.decode(full_tokens[:k])
        qpm = score.tempos[0].qpm if score.tempos else 120.0
        return score.end() / score.tpq * (60.0 / qpm)

    lo, hi = 0, len(full_tokens)
    while lo < hi:
        mid = (lo + hi) // 2
        if real_time_at(mid) < seconds:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _token_index_to_seconds(tokenizer_type: str, song_choice: str, token_idx: int) -> float:
    """Inverse of _refine_token_index: the real time a given token index
    actually starts at (no rescaling — see _get_time_map for why). Symbolic
    tokenization only has note-onset-level time granularity — a click can't
    land at an arbitrary continuous timestamp, only at whichever real
    note-onset is closest. Used to snap the displayed start-offset to what
    will actually be used, so the two never disagree (that mismatch, not the
    granularity itself, is what read as click position always being "a bit
    off").

    Anticipation reads the note's own encoded arrival time directly out of
    the uncapped compound representation — exact, no decode() needed,
    matching how _anticipation_compound_time_map itself is built (and
    critically, NOT sourced from a whole-song "events" tokenization — see
    _anticipation_window_tokens for why that isn't safe past 100 seconds)."""
    if token_idx <= 0:
        return 0.0
    if _is_triplet_tokenizer(tokenizer_type):
        midi_path = _song_midi_path(song_choice)
        if midi_path is None:
            return 0.0
        compound = _midi_to_compound_cached(midi_path)
        note_idx = min(_align_to_triplet(token_idx) // 3, len(compound) // 5 - 1)
        if note_idx < 0:
            return 0.0
        _, time_resolution = _anticipation_vocab()
        return compound[note_idx * 5] / time_resolution

    full_tokens, _, _ = _full_tokens_for_offset_range(tokenizer_type, song_choice)
    if full_tokens is None:
        return 0.0
    tokenizer = _get_tokenizer(tokenizer_type)
    score = tokenizer.decode(full_tokens[:token_idx])
    qpm = score.tempos[0].qpm if score.tempos else 120.0
    return score.end() / score.tpq * (60.0 / qpm)


def _is_seconds_mode(tokenizer_type: str, song_choice: str) -> bool:
    """A named song has a real, fixed sequence to build a time map over,
    under any tokenizer (REMI via decode()-sampled tempo, Anticipation via
    each triplet's own encoded time) — only the random-sample path (no fixed
    sequence at all) stays in raw tokens."""
    return bool(song_choice and song_choice != RANDOM_SONG_LABEL)


def _offset_to_tokens(tokenizer_type: str, song_choice: str, offset_value: float) -> int:
    """Convert the UI's start_offset slider value to a real token index —
    seconds for a named song under any tokenizer, already-tokens for the
    random-sample path."""
    if _is_seconds_mode(tokenizer_type, song_choice):
        tmap = _get_time_map(tokenizer_type, song_choice)
        if tmap:
            seconds = float(offset_value)
            coarse_idx = _seconds_to_token_index(tmap, seconds)
            return _refine_token_index(tokenizer_type, song_choice, coarse_idx, seconds)
    return int(offset_value)


# ── Prompt resolution (song + start offset, or random validation window) ────

def _full_tokens_for_offset_range(tokenizer_type: str, song_choice: str) -> tuple[list[int] | None, int, str]:
    """Return (full_tokens or None, default_start_offset, label).
    full_tokens=None means 'no single flat sequence to slice a window from' —
    the random-sample path (REMI or Anticipation), AND Anticipation named
    songs, which resolve through _anticipation_window_tokens's fresh, local,
    rebased tokenization instead (see its docstring for why a whole-song
    Anticipation tokenization isn't safe). Only REMI named songs have a real
    precomputed full-song array to return here."""
    if tokenizer_type == "REMI" and song_choice and song_choice != RANDOM_SONG_LABEL:
        with open(PROMPTS_FILE) as f:
            prompts = json.load(f)
        full_tokens = _tokens_for_song(song_choice, prompts)
        return full_tokens, _song_start_hint(song_choice, prompts), song_choice
    return None, 0, RANDOM_SONG_LABEL


def _is_triplet_tokenizer(tokenizer_type: str) -> bool:
    """Anticipation-family tokenizers encode every event as a fixed
    (time, duration, note) triplet — slicing a token window at an arbitrary
    offset can land inside a triplet instead of on its boundary, which
    doesn't just truncate cleanly: every following triplet reads shifted by
    1 or 2 positions, so a duration value gets interpreted as a time, a note
    as a duration, etc. Confirmed this crashes decode() outright once enough
    of the window is corrupted (`_filter_malformed_triplets` drops every
    triplet as out-of-range, leaving nothing for events_to_compound). REMI
    has no such constraint — each token stands alone."""
    return tokenizer_type.startswith("Anticipation")


def _align_to_triplet(n: int) -> int:
    return (n // 3) * 3


def _resolve_prompt(tokenizer_type: str, song_choice: str, start_offset: int,
                     seed_sample_idx: int | None = None) -> tuple[list[int], list[int], str, int]:
    """Return (prompt_tokens, ground_truth_continuation_tokens, label, sample_idx_used).
    sample_idx_used is only meaningful for the random path (else -1)."""
    triplet = _is_triplet_tokenizer(tokenizer_type)
    prompt_len = _align_to_triplet(PROMPT_LENGTH) if triplet else PROMPT_LENGTH

    if triplet and song_choice and song_choice != RANDOM_SONG_LABEL:
        # A named Anticipation song has no single flat full-song array to
        # slice a window from (see _anticipation_window_tokens's docstring
        # for why) — start_offset is a token index into the song's time
        # map (see _offset_to_tokens/_get_time_map), so it's converted back
        # to the real second it represents (both in the same, un-rescaled
        # time domain — see _get_time_map), and a fresh, correctly-rebased
        # local window is tokenized starting exactly there.
        midi_path = _song_midi_path(song_choice)
        start_seconds = _token_index_to_seconds(tokenizer_type, song_choice, start_offset)
        window_tokens = _anticipation_window_tokens(midi_path, start_seconds) if midi_path else []
        prompt = window_tokens[:prompt_len]
        gt = window_tokens[prompt_len:prompt_len + prompt_len]
        # Every training sequence this model has ever seen starts with a mode
        # marker (AUTOREGRESS or ANTICIPATE — see tokenize.py's seq.insert(0,
        # z), confirmed with no exceptions) — this window is built fresh from
        # raw MIDI and had no such marker, unlike listen_density_sweep.py's
        # prompts (which reuse real dataset sequences that already carry one).
        # Measured directly: the model's own loss on real validation
        # continuations is ~9% higher (worse) without it. AUTOREGRESS since
        # this is plain generation, no anticipated controls involved.
        if prompt:
            import sys
            anticipation_root = str(ROOT / "data/mus/symbolic/tokenization/anticipation")
            if anticipation_root not in sys.path:
                sys.path.insert(0, anticipation_root)
            from anticipation.vocab_selector import AUTOREGRESS
            prompt = [AUTOREGRESS] + prompt
        return prompt, gt, f"{song_choice} (offset {start_offset})", -1

    full_tokens, _, _ = _full_tokens_for_offset_range(tokenizer_type, song_choice)
    if full_tokens is not None:
        start = max(0, min(start_offset, max(0, len(full_tokens) - PROMPT_LENGTH)))
        if triplet:
            start = _align_to_triplet(start)
        window = full_tokens[start:]
        prompt = window[:prompt_len]
        gt = window[prompt_len:prompt_len + prompt_len]
        return prompt, gt, f"{song_choice} (offset {start})", -1

    # Random validation sample (REMI random-song choice, or any Anticipation prompt)
    hparams = _minimal_hparams(tokenizer_type)
    dataset = _get_dataset(hparams)
    idx = seed_sample_idx if seed_sample_idx is not None else random.randint(0, len(dataset) - 1)
    if hasattr(dataset, 'get_full_tokens'):
        full = dataset.get_full_tokens(idx)
    else:
        full = dataset[idx]['input_ids'].tolist()
    start = max(0, min(start_offset, max(0, len(full) - PROMPT_LENGTH)))
    if triplet:
        start = _align_to_triplet(start)
    window = full[start:]
    prompt = window[:prompt_len]
    gt = window[prompt_len:prompt_len + prompt_len]
    return prompt, gt, f"random validation sample {idx} (offset {start})", idx


def max_start_offset(tokenizer_type: str, song_choice: str) -> tuple[float, float, str]:
    """Return (slider_max, default_offset, unit) for the current song/tokenizer
    choice. A named song gets a real seconds range under any tokenizer (via
    the time map — REMI's decode()-sampled, Anticipation's exact-per-triplet);
    the random-sample path (no fixed sequence to map) stays in raw tokens."""
    if _is_seconds_mode(tokenizer_type, song_choice):
        tmap = _get_time_map(tokenizer_type, song_choice)
        if tmap:
            total_seconds = tmap[-1][1]
            # The saved start_token hint only exists (and only means
            # anything) for REMI — see _song_start_hint — so there's no
            # hint lookup to do for Anticipation at all.
            hint_seconds = 0.0
            if not _is_triplet_tokenizer(tokenizer_type):
                _, hint_tokens, _ = _full_tokens_for_offset_range(tokenizer_type, song_choice)
                for idx, t in tmap:
                    if idx >= hint_tokens:
                        hint_seconds = t
                        break
            # Leave a few seconds of headroom so the window doesn't run past
            # the end — _resolve_prompt's own token-level clamp is the real
            # safety net if this coarse estimate is slightly off.
            return max(total_seconds - 5.0, 0.0), hint_seconds, "seconds"
    # Random path: a generous, generic token ceiling — the actual sampled
    # sequence's real length is only known once one is drawn.
    return 2000, 0, "tokens"


def on_song_or_tokenizer_change(tokenizer_type: str, song_choice: str):
    """Only ever touches start_offset's bounds/label. It used to also
    re-emit song_dd's own value (as a no-op meant to say "leave the current
    pick alone") — but writing to song_dd from a chain that song_dd.change()
    itself triggers, self-triggering regardless of whether the value
    actually differed. That produced a runaway reload loop (dropdown, full
    song, and prompt preview all re-rendering nonstop). Song choice was
    always meant to just persist across a tokenizer switch untouched, which
    is what simply never writing to it here achieves directly."""
    slider_max, default_offset, unit = max_start_offset(tokenizer_type, song_choice)
    if unit == "seconds":
        label, step = "Prompt start (seconds into the song)", 0.5
    else:
        # Anticipation's triplet structure means only every 3rd token
        # position is valid (see _is_triplet_tokenizer) — stepping by 3
        # keeps every slider position valid, instead of relying solely on
        # _resolve_prompt's silent snap-down to save it. Only reachable here
        # by the random-sample path now — a named song always gets seconds
        # mode under any tokenizer (see _is_seconds_mode).
        token_step = 3 if _is_triplet_tokenizer(tokenizer_type) else 1
        label, step = "Prompt start offset (tokens, random validation sample)", token_step
    return gr.update(minimum=0, maximum=max(slider_max, 1), value=min(default_offset, slider_max),
                      step=step, label=label)


def _clear_prompt_audio():
    """Immediately clear the full-song and preview players when the song or
    tokenizer changes, run as the first step before the (slower) actual
    re-render. Without this, the previous song's audio kept playing/showing
    until the new one finished loading — confirmed reported as sounding like
    the demo was still playing the old song while it "reloaded"."""
    return gr.update(value=None), gr.update(value=None), "Loading new song…"


def _clear_generated_results():
    """Wipes every section's previously-generated results (baseline Generate,
    Single Attribute batch, Compose) the instant the prompt itself changes —
    song, tokenizer, or start offset. Without this, switching to a new
    prompt left the OLD prompt's audio/gauges sitting on screen until the
    next Generate click overwrote them, which reads as if they belong to the
    prompt currently selected. Reset only, no re-generation — the user still
    has to click Generate again for the new prompt, same as before any of
    this existed."""
    main_slots = []
    for key in _SLOT_KEYS:
        main_slots.extend(_slot_done(None, key))
    model_cols = tuple(gr.update(elem_classes=_card_classes(False)) for _ in _MODEL_KEYS)
    solo_reset = []
    for _ in range(_MAX_SOLO_BATCH):
        solo_reset.extend((gr.update(), "", gr.update(visible=True, value=_idle_placeholder_html()),
                            gr.update(visible=False, value=None), ""))
    empty_ref_html = _render_compose_reference_html(None, None)
    return (
        *main_slots,
        gr.update(visible=False, value=None), None, None, None,
        gr.update(visible=False, value=None),
        *model_cols,
        "",
        empty_ref_html, empty_ref_html,
        *solo_reset, "",
        gr.update(visible=False, value=None), "", "",
    )


def preview_prompt(tokenizer_type: str, song_choice: str, start_offset: float):
    try:
        token_offset = _offset_to_tokens(tokenizer_type, song_choice, start_offset)
        tokens, _, label, _ = _resolve_prompt(tokenizer_type, song_choice, token_offset)
        tokenizer = _get_tokenizer(tokenizer_type)

        from inference.mus.tokens_to_midi import tokens_to_midi
        from demo.convert_midi_simple import simple_synth

        out_dir = Path(tempfile.mkdtemp(prefix="demo_preview_"))
        midi_path = out_dir / "preview.mid"
        wav_path  = out_dir / "preview.wav"
        midi_path.write_bytes(tokens_to_midi(tokens, tokenizer))
        simple_synth(str(midi_path), str(wav_path))

        if wav_path.exists():
            # Naming the format explicitly matters here: song choice and
            # offset both persist across a tokenizer switch by design (see
            # on_song_or_tokenizer_change), so without this, the status text
            # can read as completely unchanged after switching format even
            # though the preview WAS re-tokenized and re-rendered from
            # scratch — confirmed reported as "I don't see a difference".
            # Written as plain prose rather than "(tokenizer): song (offset
            # N)" — that read poorly, and a raw token/triplet index ("offset
            # 2001") means nothing to a listener. Recomputing the actual
            # start time from token_offset (rather than reusing the slider's
            # raw start_offset) matches exactly what got rendered, including
            # any snap-to-nearest-valid-position adjustment.
            if _is_seconds_mode(tokenizer_type, song_choice):
                actual_seconds = _token_index_to_seconds(tokenizer_type, song_choice, token_offset)
                status = (f'Previewing: "{song_choice}", starting at {actual_seconds:.1f}s. '
                          f'Tokenization format chosen: {tokenizer_type}')
            else:
                status = f"Previewing: {label}. Tokenization format chosen: {tokenizer_type}"
            return str(wav_path), status
        return None, "⚠️ WAV synthesis failed."
    except Exception as e:
        return None, f"❌ Preview failed:\n{e}\n{traceback.format_exc()}"


def _wav_duration_seconds(path: str) -> float:
    import wave
    with wave.open(path, 'rb') as w:
        return w.getnframes() / w.getframerate()


def render_full_song(tokenizer_type: str, song_choice: str):
    """Render the entire song (not just a windowed prompt) so it can be
    clicked directly on its waveform to pick a start point. Any named song
    under any tokenizer — hidden only for the random-sample path. Shares the
    cached render (and its real duration) with _get_time_map so the two
    never disagree.

    Returns (group_update, audio_update): visibility is toggled on the
    wrapping gr.Group, not on the ClickAudio component itself — see the
    comment where full_song_audio is constructed for why."""
    if not _is_seconds_mode(tokenizer_type, song_choice):
        return gr.update(visible=False), gr.update(value=None)
    rendered = _get_full_song_render(tokenizer_type, song_choice)
    if not rendered:
        return gr.update(visible=False), gr.update(value=None)
    wav_path, _ = rendered
    return gr.update(visible=True), gr.update(value=wav_path)


def apply_seek_as_start(tokenizer_type: str, song_choice: str, evt: gr.EventData):
    """Fired directly by ClickAudio's `seek` event — the clicked time (in
    seconds) is exact, unlike the old trim-based back-solve, since the custom
    component reports it straight from the waveform's own click handler.

    The displayed offset is then snapped to the real time of whichever token
    _offset_to_tokens would actually resolve the click to (via the same
    coarse-interpolate + refine path), rather than showing the raw clicked
    second value. Symbolic tokenization only has note-onset-level time
    granularity, so the two can differ by a fraction of a second — left
    unsnapped, the slider showed one number while generation started from a
    different, nearby one, which is what read as the click always landing
    "a bit off"."""
    clicked_seconds = float(evt.time)
    if not _is_seconds_mode(tokenizer_type, song_choice):
        return gr.update(), "⚠️ Can't set a start point for this song/tokenizer."
    tmap = _get_time_map(tokenizer_type, song_choice)
    if not tmap:
        return gr.update(), "⚠️ Could not resolve this song's duration."
    clicked_seconds = max(0.0, min(tmap[-1][1], clicked_seconds))
    coarse_idx = _seconds_to_token_index(tmap, clicked_seconds)
    token_idx = _refine_token_index(tokenizer_type, song_choice, coarse_idx, clicked_seconds)
    offset_seconds = _token_index_to_seconds(tokenizer_type, song_choice, token_idx)
    return gr.update(value=offset_seconds), f"Start point set to {offset_seconds:.1f}s from your click."


def _synth_with_retry(tokens: list[int], tokenizer, out_dir: Path, name: str, attempts: int = 2):
    """tokens_to_midi + simple_synth, retried once on any failure before
    giving up. Isolated failures here (a slow/starved FluidSynth subprocess
    on a shared, contended compute node; a transient write hiccup) were
    reported as audio boxes intermittently and unpredictably missing across
    every section of the demo, even though the exact same conversion
    reproduces cleanly in isolation outside the live, loaded demo process —
    consistent with a transient resource issue rather than a deterministic
    bug in the conversion itself, so retrying the whole attempt (fresh MIDI
    write, fresh synth call) is the appropriate fix rather than debugging a
    single non-reproducing failure.

    Returns (wav_path_or_None, error_message_or_None)."""
    from inference.mus.tokens_to_midi import tokens_to_midi
    from demo.convert_midi_simple import simple_synth

    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            midi_path = out_dir / f"{name}.mid"
            wav_path = out_dir / f"{name}.wav"
            midi_path.write_bytes(tokens_to_midi(tokens, tokenizer))
            simple_synth(str(midi_path), str(wav_path))
            # Also guard against a truncated/zero-byte file from a render
            # that was killed or interrupted mid-write under load —
            # wav_path.exists() alone doesn't catch that.
            if wav_path.exists() and wav_path.stat().st_size > 0:
                return str(wav_path), None
            last_err = "synthesis produced an empty file"
        except Exception as e:
            last_err = str(e)
        if attempt < attempts:
            time.sleep(1.5)
    return None, last_err


# ── Core single-model generation (shared by the baseline row and both EBT
#    attribute-control windows) ───────────────────────────────────────────────

def _run_generation(model_key: str, ckpt_path: str, tokenizer_type: str,
                     prompt_tokens: list[int], gt_tokens: list[int],
                     temperature: float, top_p: float, generation_length: int,
                     attribute_overrides: dict | None = None):
    """Returns dict with keys: wav_paths (prompt/generated/combined/gt_combined),
    generated_tokens, tokenizer, hparams, status_lines. Raises on failure."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    status_lines = [f"{model_key} · device={device} · checkpoint={Path(ckpt_path).parent.name}"]

    model, hparams, tokenizer = _load(model_key, ckpt_path, device)
    status_lines.append(f"{model_key}: model loaded" + (" (cached)" if True else ""))

    hparams.infer_temp        = temperature
    hparams.infer_topp        = top_p
    hparams.infer_max_gen_len = generation_length
    hparams.infer_logprobs    = False
    hparams.infer_echo        = False

    hparams.attribute_target = None
    hparams.lambda_attribute = 0.0
    hparams.attribute_regressor_ckpt = None
    hparams.attribute_regressor_ckpts = None
    hparams.attribute_targets = None
    hparams.attribute_weights = None
    if attribute_overrides:
        for k, v in attribute_overrides.items():
            setattr(hparams, k, v)

    prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)
    batch = {'input_ids': prompt_tensor}
    model_name = _MODEL_INFO[model_key]["model_name"]

    with torch.no_grad():
        if model_name == "ebt":
            from inference.mus.generate_music import generate_music
            outputs = generate_music(model, batch, hparams)
        else:
            from inference.mus.generate_music import generate_remi, generate_anticipation
            if "Anticipation" in hparams.tokenizer_type:
                outputs = generate_anticipation(model, batch, hparams)
            else:
                outputs = generate_remi(model, batch, hparams)

    generated_tokens = [int(t) for t in outputs['generation_tokens'][0]]
    status_lines.append(f"{model_key}: generated {len(generated_tokens)} tokens")

    out_dir = Path(tempfile.mkdtemp(prefix=f"demo_{model_key}_"))
    combined = prompt_tokens + generated_tokens

    # Only "generated" and "combined" here — the prompt and prompt+ground-truth
    # are identical across every model (same prompt, same tokenizer), so
    # generate_all() renders those once via _render_prompt_and_gt() instead of
    # every model redundantly re-synthesizing the same audio.
    wav_paths = {}
    for name, toks in [("generated", generated_tokens), ("combined", combined)]:
        wav_path, err = _synth_with_retry(toks, tokenizer, out_dir, name)
        wav_paths[name] = wav_path
        if err:
            status_lines.append(f"⚠️ {model_key} {name} conversion failed: {err}")

    return {
        "wav_paths": wav_paths, "generated_tokens": generated_tokens,
        "tokenizer": tokenizer, "hparams": hparams, "status_lines": status_lines,
        # Only EBT's own generate_music() populates this (see its diagnostics
        # collection) — Llama/GPT-2 have no comparable "energy" to trace.
        "diagnostics": outputs.get('diagnostics') if model_name == "ebt" else None,
    }


def _render_prompt_and_gt(tokenizer_type: str, prompt_tokens: list[int], gt_tokens: list[int]) -> dict:
    """Render the prompt and prompt+ground-truth once, shared across all models
    in a single generate_all() call (they don't depend on which model is used)."""
    tokenizer = _get_tokenizer(tokenizer_type)
    out_dir = Path(tempfile.mkdtemp(prefix="demo_shared_"))
    gt_combined = prompt_tokens + gt_tokens
    wav_paths = {}
    for name, toks in [("prompt", prompt_tokens), ("gt_combined", gt_combined)]:
        wav_path, _err = _synth_with_retry(toks, tokenizer, out_dir, name)
        wav_paths[name] = wav_path
    return wav_paths


def _log_to_wandb(tag: str, wav_paths: dict, meta: dict):
    try:
        import wandb
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=tag, reinit=True)
        for name, path in wav_paths.items():
            if path:
                wandb.log({f"audio/{name}": wandb.Audio(path, sample_rate=44100)})
        wandb.log(meta)
        wandb.finish()
        return "✅ Logged to WandB"
    except Exception as e:
        return f"⚠️ WandB logging failed: {e}"


# ── Baseline (multi-model) generation ────────────────────────────────────────
#
# Each output "slot" is a (placeholder, audio) pair — see _audio_slot() in the
# UI section — so generation-in-progress can show a big, unmistakable
# "Generating…" panel in the exact spot the audio will appear, instead of
# relying on Gradio's own small default spinner. Only one of the pair is
# visible at a time; _slot_loading()/_slot_done() build the (placeholder,
# audio) gr.update() tuple for each state.

_SLOT_KEYS = [
    "shared_prompt", "shared_gt",
    "EBT_generated", "EBT_combined",
    "Llama_generated", "Llama_combined",
    "GPT-2_generated", "GPT-2_combined",
]


def _slot_label(key: str) -> str:
    if key == "shared_prompt":
        return "Prompt"
    if key == "shared_gt":
        return "Prompt + Original Continuation"
    if key.endswith("_combined"):
        return "Prompt + Generated Continuation"
    return "Generated Continuation"


def _slot_idle_text(key: str) -> str:
    # The "Prompt"/"Prompt + Original Continuation" boxes are waiting on the
    # prompt itself being resolved (which only happens once Generate runs),
    # not on a model producing something — "prompt not chosen yet" says that
    # directly instead of the generic "nothing to display" wording used for
    # the per-model output boxes.
    if key in ("shared_prompt", "shared_gt"):
        return "Prompt not chosen yet — click Generate above."
    return _DEFAULT_IDLE_TEXT


def _slot_loading(key: str):
    # Explicitly re-supplies the placeholder's value (not just visible=True):
    # _slot_done() below has to clear this same value to actually hide the
    # placeholder (see its comment), so on the *next* run this restores the
    # "Generating…" text rather than showing a blank box.
    return gr.update(visible=True, value=_placeholder_html(_slot_label(key))), gr.update(visible=False, value=None)


def _slot_done(wav_path, key: str = ""):
    # visible=False alone, with no accompanying value change, doesn't reliably
    # take effect on a gr.HTML component across successive generator yields —
    # confirmed via a minimal repro (the placeholder stayed visible forever
    # once shown). Clearing the value alongside visible=False is what actually
    # hides it.
    if wav_path:
        return gr.update(visible=False, value=""), gr.update(visible=True, value=wav_path)
    # Nothing was produced (skipped model, missing checkpoint, or an error) —
    # show the idle message instead of an empty audio player.
    return gr.update(visible=True, value=_idle_placeholder_html(_slot_idle_text(key))), gr.update(visible=False, value=None)


def generate_all(
    tokenizer_type: str, song_choice: str, start_offset: int,
    temperature: float, top_p: float, generation_length: int,
    ebt_on: bool, ebt_ckpt: str, ebt_paths_json: str,
    llama_on: bool, llama_ckpt: str, llama_paths_json: str,
    gpt2_on: bool, gpt2_ckpt: str, gpt2_paths_json: str,
):
    def flat(slot_values: dict) -> tuple:
        out = []
        for key in _SLOT_KEYS:
            out.extend(slot_values[key])
        return tuple(out)

    enabled = {"EBT": (ebt_on, ebt_ckpt, ebt_paths_json),
               "Llama": (llama_on, llama_ckpt, llama_paths_json),
               "GPT-2": (gpt2_on, gpt2_ckpt, gpt2_paths_json)}

    slots = {key: _slot_loading(key) for key in _SLOT_KEYS}
    # Models that aren't even checked shouldn't sit in "generating" forever —
    # mark them done-but-empty immediately.
    for model_key, (on, _, _) in enabled.items():
        if not on:
            slots[f"{model_key}_generated"] = _slot_done(None, f"{model_key}_generated")
            slots[f"{model_key}_combined"] = _slot_done(None, f"{model_key}_combined")

    # EBT's "prompt + generated" from this run, surfaced as a reference in the
    # Attribute Control section below so guided output can be compared against
    # the model's own unguided default without re-running generation. Velocity
    # and duration are computed the same rule-based way _make_solo_row_step/
    # _compose_generate measure "achieved" values, so the reference numbers are
    # directly comparable to a guided run's gauge.
    ebt_reference_wav = None
    ebt_reference_velocity = None
    ebt_reference_duration = None
    ebt_reference_pitch_register = None

    # Reveals all three model-comparison columns the moment Generate is
    # clicked (constant for every yield in this run) — they start hidden
    # (see their _card_classes(False) at layout time) so a fresh page load
    # doesn't show three empty "nothing yet" boxes before you've asked for
    # anything, unlike the shared prompt/ground-truth boxes above them,
    # which stay visible since "waiting on a prompt choice" is real state.
    model_col_reveal = tuple(gr.update(elem_classes=_card_classes(True)) for _ in _MODEL_KEYS)

    def ref_tuple():
        # Plain values here leave the Audio component's own `visible` prop
        # untouched, so before EBT has ever produced a reference it sat there
        # as an empty player next to the "No reference yet" text below it —
        # gr.update() lets us hide it until there's actually something to
        # play, matching that same "no reference yet" state visually.
        # Same wav shown twice — once at the top of section 4, once again in
        # the Compose column so it's visible right next to its own reference
        # text without scrolling back up to the shared copy above. Two
        # separate gr.update() calls (not one object reused) so each output
        # component gets its own update instance.
        return (gr.update(visible=bool(ebt_reference_wav), value=ebt_reference_wav),
                ebt_reference_velocity, ebt_reference_duration, ebt_reference_pitch_register,
                gr.update(visible=bool(ebt_reference_wav), value=ebt_reference_wav))

    # Only two yields total: this immediate "Starting…" for instant feedback,
    # then one single final yield with the complete result. Used to yield
    # once per model instead — under the real network/resource conditions
    # this demo actually runs under (a shared, contended compute node), each
    # of those intermediate yields was a chance for Gradio's streaming to
    # drop, reorder, or redundantly reapply a partial state, surfacing as
    # boxes intermittently missing their content or visibly resizing/
    # flickering mid-run. Fewer yields means fewer chances for that —
    # reliability over a progressive per-model reveal.
    yield (*flat(slots), *ref_tuple(), *model_col_reveal, "Starting…")

    try:
        token_offset = _offset_to_tokens(tokenizer_type, song_choice, start_offset)
        prompt_tokens, gt_tokens, prompt_label, sample_idx = _resolve_prompt(
            tokenizer_type, song_choice, token_offset)
    except Exception as e:
        msg = f"❌ Could not resolve prompt:\n{e}\n{traceback.format_exc()}"
        slots = {key: _slot_done(None, key) for key in _SLOT_KEYS}
        yield (*flat(slots), *ref_tuple(), *model_col_reveal, msg)
        return

    status_all = [f"Prompt: {prompt_label}"]

    try:
        shared_wavs = _render_prompt_and_gt(tokenizer_type, prompt_tokens, gt_tokens)
        slots["shared_prompt"] = _slot_done(shared_wavs.get("prompt"), "shared_prompt")
        slots["shared_gt"] = _slot_done(shared_wavs.get("gt_combined"), "shared_gt")
    except Exception as e:
        status_all.append(f"⚠️ Shared prompt/ground-truth render failed: {e}")
        shared_wavs = {}
        slots["shared_prompt"] = _slot_done(None, "shared_prompt")
        slots["shared_gt"] = _slot_done(None, "shared_gt")

    for model_key, (on, ckpt_display, paths_json) in enabled.items():
        if not on:
            status_all.append(f"{model_key}: skipped")
            continue
        if not ckpt_display or not paths_json:
            status_all.append(f"{model_key}: ❌ no checkpoint selected")
            slots[f"{model_key}_generated"] = _slot_done(None, f"{model_key}_generated")
            slots[f"{model_key}_combined"] = _slot_done(None, f"{model_key}_combined")
            continue
        try:
            ckpt_path = json.loads(paths_json).get(ckpt_display)
            out = _run_generation(model_key, ckpt_path, tokenizer_type,
                                   prompt_tokens, gt_tokens, temperature, top_p, generation_length)
            status_all.extend(out["status_lines"])
            wp = out["wav_paths"]
            slots[f"{model_key}_generated"] = _slot_done(wp.get("generated"), f"{model_key}_generated")
            slots[f"{model_key}_combined"] = _slot_done(wp.get("combined"), f"{model_key}_combined")
            if model_key == "EBT":
                ebt_reference_wav = wp.get("combined")
                from attribute_control.attributes import ATTRIBUTES
                # Each measured independently — velocity raising
                # NotImplementedError under Anticipation (expected, see
                # compute_velocity) used to abort this whole block before
                # duration ever got computed, so duration's reference silently
                # never showed up there either even though it works fine.
                try:
                    ebt_reference_velocity = ATTRIBUTES["velocity"](out["generated_tokens"], tokenizer_type)
                except Exception:
                    pass
                try:
                    ebt_reference_duration = ATTRIBUTES["duration"](out["generated_tokens"], tokenizer_type)
                except Exception as e:
                    status_all.append(f"⚠️ Could not measure duration reference: {e}")
                try:
                    ebt_reference_pitch_register = ATTRIBUTES["pitch_register"](out["generated_tokens"], tokenizer_type)
                except Exception as e:
                    status_all.append(f"⚠️ Could not measure pitch register reference: {e}")
            _log_to_wandb(f"demo_{model_key}_{tokenizer_type}",
                          {**wp, **shared_wavs},
                          {"model": model_key, "checkpoint": ckpt_display,
                           "tokenizer": tokenizer_type, "prompt": prompt_label,
                           "temperature": temperature, "top_p": top_p,
                           "generation_length": generation_length})
        except Exception as e:
            status_all.append(f"{model_key}: ❌ {e}\n{traceback.format_exc()}")
            slots[f"{model_key}_generated"] = _slot_done(None, f"{model_key}_generated")
            slots[f"{model_key}_combined"] = _slot_done(None, f"{model_key}_combined")

    yield (*flat(slots), *ref_tuple(), *model_col_reveal, "\n".join(status_all))


# ── EBT single-attribute generation (batch: N independent runs at once) ─────

_MAX_SOLO_BATCH = 6  # fixed pool of output slots in the UI; extra rows are ignored
_MAX_COMPOSE_ROWS = len(_SOLO_ATTRIBUTES)  # composing the same attribute twice is meaningless


def _card_classes_slot_helpers():
    """Shared placeholder/hidden-slot builders for the Single Attribute batch
    — split out so both _solo_batch_start and _make_solo_row_step build
    identical tuples without duplicating the elem_classes/visibility dance."""
    def hidden_slot():
        # The slot's own container is a gr.Group, whose `visible` prop is
        # unreliable in this Gradio version (see _card_classes) — toggle the
        # elem_classes-driven CSS hide instead, same as the input rows above.
        return (gr.update(elem_classes=_card_classes(False)), "", gr.update(visible=False),
                gr.update(visible=False, value=None), "")

    def pending_slot(title):
        # Explicitly restores the placeholder's text (not just visible=True) —
        # the "done"/"failed" builders below clear this same value to
        # actually hide it (see their comment), so a later run needs it put
        # back rather than staying blank.
        return (gr.update(elem_classes=_card_classes(True)), title,
                gr.update(visible=True, value=_placeholder_html("Generated Continuation")),
                gr.update(visible=False, value=None), "")

    def failed_slot(title, gauge=""):
        # WAV synthesis (or setup) genuinely failed — show the idle
        # placeholder instead of an Audio component with no source, which
        # otherwise renders as an empty, broken-looking player.
        return (gr.update(elem_classes=_card_classes(True)), title,
                gr.update(visible=True, value=_idle_placeholder_html("Audio unavailable — see Status below.")),
                gr.update(visible=False, value=None), gauge)

    def done_slot(title, wav_path, gauge):
        # visible=False alone, with no accompanying value change, doesn't
        # reliably take effect on a gr.HTML component across successive
        # updates — confirmed via a minimal repro (the placeholder stayed
        # visible forever once shown). Clearing the value alongside
        # visible=False is what actually hides it.
        if wav_path:
            return (gr.update(elem_classes=_card_classes(True)), title, gr.update(visible=False, value=""),
                    gr.update(visible=True, value=wav_path), gauge)
        return failed_slot(title, gauge)

    return hidden_slot, pending_slot, failed_slot, done_slot


def _solo_batch_start(
    tokenizer_type: str, song_choice: str, start_offset: int,
    temperature: float, top_p: float, generation_length: int,
    ebt_ckpt: str, ebt_paths_json: str,
    row_count: int, *row_values,
):
    """First step of the Single Attribute batch — click handler. Builds the
    row list, shows every active row's pending placeholder, resolves the
    prompt once (shared by every row), and stashes everything the per-row
    steps below need into one gr.State.

    This used to be one generator covering the whole batch, yielding a
    single update for all _MAX_SOLO_BATCH * 5 components at once. That
    single oversized update turned out to be the actual cause of cards
    flickering and — worse — one card (consistently the last one in the
    flattened tuple) sometimes never getting its audio player at all, even
    over a direct SSH tunnel with no public relay involved. Splitting the
    batch into one chained `.then()` step per row (see _make_solo_row_step)
    means no single Gradio update ever touches more than one row's 5
    components, so there's nothing left for a large-payload bug to drop."""
    hidden_slot, pending_slot, failed_slot, _ = _card_classes_slot_helpers()

    def flat(slots):
        out = []
        for i in range(_MAX_SOLO_BATCH):
            out.extend(slots[i])
        return tuple(out)

    rows = []
    for i in range(min(int(row_count), _MAX_SOLO_BATCH)):
        attr, lam, target = row_values[i * 3], row_values[i * 3 + 1], row_values[i * 3 + 2]
        if attr in _SOLO_ATTRIBUTES:
            rows.append((attr, float(target), float(lam)))

    if not rows:
        slots = [hidden_slot() for _ in range(_MAX_SOLO_BATCH)]
        return (*flat(slots),
                "❌ No valid rows — each needs attribute ('velocity' or 'duration'), a numeric target, and a numeric λ.",
                {"rows": []})
    if not ebt_ckpt or not ebt_paths_json:
        slots = [hidden_slot() for _ in range(_MAX_SOLO_BATCH)]
        return (*flat(slots), "❌ Select an EBT checkpoint in the panel above first.", {"rows": []})

    n = len(rows)
    # "Variant N" matches the numbering on the input row it came from, so the
    # output can be tracked back to its config at a glance.
    titles = [f"<b>Variant {i + 1}</b><br>Attribute: {attr.replace('_', ' ')}<br>Parameters: λ = {lam:.3f}, T = {target:.3f}"
              for i, (attr, target, lam) in enumerate(rows)]

    try:
        ckpt_path = json.loads(ebt_paths_json).get(ebt_ckpt)
        token_offset = _offset_to_tokens(tokenizer_type, song_choice, start_offset)
        prompt_tokens, gt_tokens, prompt_label, _ = _resolve_prompt(tokenizer_type, song_choice, token_offset)
    except Exception as e:
        msg = f"❌ Could not resolve prompt:\n{e}\n{traceback.format_exc()}"
        slots = [failed_slot(titles[i]) if i < n else hidden_slot() for i in range(_MAX_SOLO_BATCH)]
        # rows=[] tells every per-row step below there's nothing left for it
        # to do — the failure above already finalized every active slot.
        return (*flat(slots), msg, {"rows": []})

    slots = [pending_slot(titles[i]) if i < n else hidden_slot() for i in range(_MAX_SOLO_BATCH)]
    state = {
        "rows": rows, "titles": titles,
        "ckpt_path": ckpt_path, "tokenizer_type": tokenizer_type,
        "prompt_tokens": prompt_tokens, "gt_tokens": gt_tokens, "prompt_label": prompt_label,
        "temperature": temperature, "top_p": top_p, "generation_length": generation_length,
    }
    return (*flat(slots), f"Prompt: {prompt_label}", state)


def _make_solo_row_step(i: int):
    """One chained `.then()` step, scoped to exactly row `i`'s own 5 output
    components (plus the shared status textbox) — see _solo_batch_start's
    comment for why this replaced one big end-of-batch update."""
    _, _, _, done_slot = _card_classes_slot_helpers()

    def _step(state: dict, status_text: str):
        rows = state.get("rows", [])
        if i >= len(rows):
            # Not an active row this run, or _solo_batch_start already
            # finalized every slot itself (e.g. a prompt-resolution error) —
            # this step's only outputs are row i's own components, so a true
            # no-op here can never clobber a different row's real update.
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), status_text

        attr, target, lam = rows[i]
        title = state["titles"][i]
        tokenizer_type = state["tokenizer_type"]
        try:
            reg_path, reg_status = _find_attribute_regressor(attr, tokenizer_type)
            if not reg_path:
                status_text = status_text + f"\nRow {i + 1} ({attr}): ❌ {reg_status}"
                return (*done_slot(title, None, ""), status_text)

            out = _run_generation(
                "EBT", state["ckpt_path"], tokenizer_type,
                state["prompt_tokens"], state["gt_tokens"],
                state["temperature"], state["top_p"], state["generation_length"],
                attribute_overrides={
                    "attribute_target": target,
                    "lambda_attribute": lam,
                    "attribute_regressor_ckpt": reg_path,
                },
            )
            from attribute_control.attributes import ATTRIBUTES
            achieved = ATTRIBUTES[attr](out["generated_tokens"], tokenizer_type)
            gauge = _attribute_gauge_html(f"{attr.replace('_', ' ').capitalize()} guidance (λ={lam})",
                                           [(attr, target, achieved, _ATTRIBUTE_UI[attr]["max"])])
            status_text = status_text + "\n" + "\n".join(out["status_lines"])
            _log_to_wandb(f"demo_EBT_{attr}_{tokenizer_type}", out["wav_paths"],
                          {"attribute": attr, "target": target, "lambda": lam,
                           "achieved": achieved, "prompt": state.get("prompt_label", "")})
            return (*done_slot(title, out["wav_paths"].get("generated"), gauge), status_text)
        except Exception as e:
            status_text = status_text + f"\nRow {i + 1} ({attr}): ❌ {e}"
            return (*done_slot(title, None, ""), status_text)

    return _step


# ── EBT compose (velocity + duration) generation ─────────────────────────────

def _compose_start():
    """First step of Compose — immediate "Generating…" feedback. Split out
    from the actual generation (see _compose_generate) and chained via
    `.then()` rather than done as a second yield from one generator on the
    same click event: a single generator's *later* yields were already
    established (see _solo_batch_start's comment) as unreliable in this
    environment for actually landing on the frontend, independent of how
    small the payload is — which is exactly what was happening here (the
    audio player's final visible=True update was the one being lost)."""
    return gr.update(visible=False, value=None), "⏳ Generating…", ""


def _compose_generate(
    tokenizer_type: str, song_choice: str, start_offset: int,
    temperature: float, top_p: float, generation_length: int,
    ebt_ckpt: str, ebt_paths_json: str,
    lam: float,
    row_count: int, *row_values,
):
    """One EBT generation combining every active row's attribute into a
    SINGLE guided run — their energies are summed (each scaled by its own
    row's relative weight), then the combined term is scaled by the one
    shared λ above. Unlike Single attribute's batch (N independent
    generations, one per row), Compose always produces exactly one output.

    The backend (generate_remi's attribute_regressor_ckpts/targets/weights
    lists) already supported composing any number of attributes — see
    compose_attributes.py, which exercises exactly this — only the demo UI
    was ever hardcoded to velocity+duration specifically."""
    rows = []
    for i in range(min(int(row_count), _MAX_COMPOSE_ROWS)):
        attr, target, weight = row_values[i * 3], row_values[i * 3 + 1], row_values[i * 3 + 2]
        if attr in _SOLO_ATTRIBUTES:
            rows.append((attr, float(target), float(weight)))

    if not rows:
        return gr.update(visible=False, value=None), "❌ No valid rows — add at least one attribute.", ""
    if len({a for a, _, _ in rows}) != len(rows):
        return gr.update(visible=False, value=None), "❌ Each row must use a different attribute.", ""
    if not ebt_ckpt or not ebt_paths_json:
        return gr.update(visible=False, value=None), "❌ Select an EBT checkpoint in the panel above first.", ""

    try:
        ckpt_path = json.loads(ebt_paths_json).get(ebt_ckpt)
        token_offset = _offset_to_tokens(tokenizer_type, song_choice, start_offset)
        prompt_tokens, gt_tokens, prompt_label, _ = _resolve_prompt(tokenizer_type, song_choice, token_offset)

        reg_paths, reg_statuses = [], []
        for attr, _, _ in rows:
            reg_path, reg_status = _find_attribute_regressor(attr, tokenizer_type)
            reg_statuses.append(reg_status)
            if not reg_path:
                return gr.update(visible=False, value=None), "❌ " + "\n".join(reg_statuses), ""
            reg_paths.append(reg_path)

        out = _run_generation(
            "EBT", ckpt_path, tokenizer_type, prompt_tokens, gt_tokens,
            temperature, top_p, generation_length,
            attribute_overrides={
                "attribute_regressor_ckpts": reg_paths,
                "attribute_targets": [t for _, t, _ in rows],
                "attribute_weights": [w for _, _, w in rows],
                "lambda_attribute": lam,
            },
        )
        from attribute_control.attributes import ATTRIBUTES
        gen = out["generated_tokens"]
        gauge_rows = []
        wandb_meta = {"lambda": lam, "prompt": prompt_label}
        for attr, target, weight in rows:
            achieved = ATTRIBUTES[attr](gen, tokenizer_type)
            gauge_rows.append((attr, target, achieved, _ATTRIBUTE_UI[attr]["max"]))
            wandb_meta[f"{attr}_target"] = target
            wandb_meta[f"{attr}_achieved"] = achieved
            wandb_meta[f"{attr}_weight"] = weight
        title = " + ".join(a.replace('_', ' ') for a, _, _ in rows)
        gauge = _attribute_gauge_html(f"{title} compose (λ={lam})", gauge_rows)
        status = "\n".join(out["status_lines"] + [f"Prompt: {prompt_label}", *reg_statuses])
        wandb_status = _log_to_wandb(f"demo_EBT_compose_{tokenizer_type}", out["wav_paths"], wandb_meta)
        return gr.update(visible=True, value=out["wav_paths"].get("generated")), status + "\n" + wandb_status, gauge
    except Exception as e:
        return gr.update(visible=False, value=None), f"❌ {e}\n{traceback.format_exc()}", ""


# ── Theme ─────────────────────────────────────────────────────────────────────

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    # background_fill_primary/secondary are the base tokens body_background_fill
    # etc. derive from — but overriding only the derived tokens (as a first pass
    # here did) leaves anything that reads the base tokens directly, like the
    # Dropdown popup/listbox (which uses background_fill_secondary, "items
    # placed on top of another item"), still on the light default. That's what
    # caused white text on a white popup — set the base tokens explicitly too.
    background_fill_primary="*neutral_950",
    background_fill_primary_dark="*neutral_950",
    background_fill_secondary="*neutral_800",
    background_fill_secondary_dark="*neutral_800",
    border_color_primary="*neutral_700",
    border_color_primary_dark="*neutral_700",
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_800",
    block_border_color_dark="*neutral_800",
    block_radius="16px",
    block_shadow="0 4px 24px rgba(0,0,0,0.25)",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_400",
    button_primary_text_color="white",
    # button_secondary_* also drives the Radio/Checkbox "pill" background
    # (checkbox_label_background_fill defaults to *button_secondary_background_fill)
    # — left at the Base theme's light default, this is what made the
    # velocity/duration attribute picker and any non-primary button render
    # white instead of matching the rest of the dark UI.
    button_secondary_background_fill="*neutral_800",
    button_secondary_background_fill_dark="*neutral_800",
    button_secondary_background_fill_hover="*neutral_700",
    button_secondary_background_fill_hover_dark="*neutral_700",
    button_secondary_text_color="*neutral_100",
    button_secondary_text_color_dark="*neutral_100",
    button_secondary_border_color="*neutral_700",
    button_secondary_border_color_dark="*neutral_700",
    checkbox_label_background_fill_selected="*primary_500",
    checkbox_label_background_fill_selected_dark="*primary_500",
    checkbox_label_text_color_selected="white",
    checkbox_label_text_color_selected_dark="white",
    # table_even/odd_background_fill default to literal "white"/near-white
    # (Base theme's light-mode default) — same class of bug as the button/
    # checkbox one above, surfaced by the batch-generation Dataframe table.
    table_even_background_fill="*neutral_900",
    table_even_background_fill_dark="*neutral_900",
    table_odd_background_fill="*neutral_800",
    table_odd_background_fill_dark="*neutral_800",
    table_border_color="*neutral_700",
    table_border_color_dark="*neutral_700",
    table_text_color="*neutral_100",
    table_text_color_dark="*neutral_100",
    input_background_fill="*neutral_800",
    input_background_fill_dark="*neutral_800",
    body_text_color="*neutral_100",
    body_text_color_dark="*neutral_100",
    body_text_color_subdued="*neutral_400",
    body_text_color_subdued_dark="*neutral_400",
)

_CSS = """
.model-card { border-radius: 16px !important; padding: 14px !important; }
.attr-window { border-radius: 16px !important; padding: 16px !important; margin-top: 8px; }
.section-title { font-size: 1.15em; font-weight: 700; margin-bottom: 4px; }
.section-sub { color: var(--body-text-color-subdued); font-size: .88em; margin-bottom: 12px; }
footer { display: none !important; }

.gen-placeholder {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 84px; border-radius: 12px;
    background: var(--block-background-fill); border: 1px dashed var(--primary-500);
    animation: gen-pulse 1.4s ease-in-out infinite;
}

/* This was `min-height` before, which let each real audio player grow
   taller once its waveform finished decoding — that decode is async and
   per-element (fetch + wavesurfer render), so the 6 variant boxes in a row
   each pop to full height at a different real moment. In a wrapping flex
   row, one item growing reflows every sibling *to its right*, which is
   exactly the repeated, cascading flicker reported (not a one-time settle,
   and worst on the right-hand boxes in each row). Fixing the height
   (instead of just floor-ing it) and clipping overflow means no
   waveform-decode timing can ever change a box's footprint again, so nothing
   downstream has anything to reflow around. */
/* NOT overflow:hidden + a hard height — that clipped the real Audio
   component's own container the instant it went from `display:none`
   (hidden at layout time) to visible, and its internal waveform renderer
   (wavesurfer) sizes its canvas off that container's box at mount time.
   Clipping it before/while it measures is a plausible way to get a canvas
   sized to ~0 and nothing drawn at all — worse than the original flicker,
   and consistent with what was reported (audio not appearing at all,
   rather than intermittently). min-height only reserves space without ever
   constraining the real player once it mounts. */
/* transition (not overflow:hidden — that broke the waveform renderer
   entirely, see the comment this replaced) smooths the one-time jump each
   box makes when its real content mounts, instead of snapping instantly —
   with up to 6 rows completing one at a time (sequential generation, not
   parallel), each box's own jump lands at a different real moment, which
   read as ongoing "flickering" across the whole batch even though no single
   box was looping. */
.gen-audio-slot, .gen-audio-slot > div { min-height: 148px; transition: min-height .2s ease; }
.gen-placeholder { min-height: 148px; transition: min-height .2s ease; }

/* Same problem, different component: the attribute gauge is a gr.HTML that
   sits empty (value="") while a row is pending, then jumps to its full
   rendered height only once a result exists — reserving that height up
   front means the card never has to resize when the gauge fills in. */
.attr-gauge-slot { min-height: 74px; transition: min-height .15s ease; }

/* The Single Attribute output cards were a multi-column flex row that wraps
   onto additional lines once cards no longer fit one row — that wrapping
   was the actual mechanism behind a "flickering" bug that survived many
   rounds of other fixes: any card's size changing (e.g. its waveform
   finishing an async render) can shift where LATER cards land, especially
   the first card of a wrapped line, since its position is most exposed to
   the row above settling. A single-column vertical stack removes wrapping
   entirely — one card's size changing can only push cards below it down
   (ordinary, expected block-layout behavior, not a reflow cascade), so
   there's no mechanism left for a size change anywhere to visually disturb
   a DIFFERENT card. Trades compactness (more scrolling for 6 variants) for
   actually eliminating the bug's cause instead of patching its symptoms. */
.solo-output-row { flex-direction: column !important; }
.solo-output-row > * { width: 100% !important; }
.gen-placeholder .gen-icon { font-size: 1.8em; line-height: 1; }
.gen-placeholder .gen-text { font-weight: 600; margin-top: 4px; font-size: .85em; }
@keyframes gen-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .45; } }

/* Idle state — nothing generated yet, nothing in progress either. No pulse
   (that reads as "working"), muted border/text so it doesn't compete with
   the busier "in progress" panel for attention. */
.gen-placeholder-idle {
    animation: none; border: 1px dashed var(--border-color-primary);
    opacity: .8;
}
.gen-placeholder-idle .gen-text { font-weight: 500; color: var(--body-text-color-subdued); }

/* Checkpoint scan in progress — a spinning icon so the (previously silent,
   show_progress="hidden") scan doesn't read as "nothing is happening" or
   "stuck" while the dropdown is disabled. */
@keyframes ckpt-spin { to { transform: rotate(360deg); } }
.ckpt-loading { color: var(--body-text-color-subdued); font-style: italic; }
.ckpt-loading .ckpt-spin {
    display: inline-block; margin-right: 4px;
    animation: ckpt-spin 0.9s linear infinite;
}

.centered-btn-row { display: flex !important; justify-content: center !important; }
.centered-btn-row button {
    min-width: 320px; font-size: 1.2em !important; padding: 16px 32px !important;
}

.reference-box {
    font-size: .9em; padding: 8px 12px; border-radius: 10px;
    background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
    margin-top: 6px;
}

.variants-bundle {
    border: 2px solid var(--primary-500); border-radius: 16px;
    padding: 12px; margin: 10px 0; gap: 10px !important;
}

/* Gradio's own `visible` prop toggling is unreliable on both gr.Column
   (reveal delayed by a click) and gr.Group (hide leaves an empty
   card-styled box behind) in this Gradio version. Toggling this class via
   elem_classes instead sidesteps both bugs entirely. */
.hidden-variant { display: none !important; }

/* Accordion's collapsed-state arrow rotates to point left by default;
   point it right (toward the "expand" direction) instead. Open state
   (rotate(0deg), pointing down) is untouched. */
.label-wrap:not(.open) .icon { transform: rotate(-90deg) !important; }

/* Lines the rescan button up with the dropdown's value box rather than its
   label row — see the checkpoint-picker's own comment for why. */
.ckpt-refresh-row { align-items: flex-end !important; gap: 10px !important; }
.ckpt-refresh-row button {
    min-width: fit-content !important;
    padding: 10px 16px !important;
    font-size: .9em !important;
    white-space: nowrap !important;
    border-radius: 8px !important;
}

/* Hero header: previously used the exact same plain-text style as every
   subsection heading below it, so the top of the page had no visual
   identity of its own. A soft violet-tinted panel plus a gradient title
   (using the theme's own primary/secondary tokens, not arbitrary colors)
   gives it one without introducing a new palette. */
.hero-header {
    background: radial-gradient(circle at 15% 20%, rgba(167,139,250,0.22) 0%, transparent 55%),
                linear-gradient(135deg, rgba(139,92,246,0.16) 0%, rgba(139,92,246,0.03) 65%, transparent 100%);
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 20px !important;
    padding: 28px 32px 22px !important;
    margin-bottom: 10px;
    text-align: center;
}
/* The gradient/clip has to land on the actual text-bearing element — the
   markdown wrapper Gradio attaches elem_classes to only contains the real
   <h1>, so applying background-clip:text up on the wrapper clips against
   an empty box (no direct text of its own) and silently no-ops, leaving
   the h1's own inherited white color showing instead. */
.app-title h1 {
    font-size: 2.2em !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    margin-bottom: 6px !important;
    background-image: linear-gradient(90deg, var(--primary-300) 0%, var(--primary-500) 55%, var(--secondary-400) 100%) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    color: transparent !important;
    display: inline-block !important;
}
.app-subtitle { color: var(--body-text-color-subdued); font-size: .95em; }
"""


# ── UI ────────────────────────────────────────────────────────────────────────

def _placeholder_html(label: str) -> str:
    # "{label} — in progress…" reads cleanly for every label this is used
    # with ("Prompt", "Prompt + Original Continuation", "Generated
    # Continuation", "Prompt + Generated Continuation") — "Generating
    # {label}…" read as "Generating Generated…" when label was just
    # "Generated".
    return (f'<div class="gen-placeholder"><div class="gen-icon">🎵</div>'
            f'<div class="gen-text">{label} — in progress…</div></div>')


_DEFAULT_IDLE_TEXT = "Nothing to display yet — click Generate above."


def _idle_placeholder_html(text: str = _DEFAULT_IDLE_TEXT) -> str:
    return f'<div class="gen-placeholder gen-placeholder-idle"><div class="gen-text">{text}</div></div>'


def _audio_slot(label: str, idle_text: str = _DEFAULT_IDLE_TEXT):
    """One output audio slot: an idle "nothing yet" placeholder before
    Generate has ever run, a big pulsing "Generating…" placeholder while it's
    in flight, and the real audio player once a result exists — only one of
    the three visible at a time. Swapping visibility (not Gradio's small
    default spinner) so it's unmistakable when something in that exact spot
    is being generated, and so the audio player never sits there empty."""
    placeholder = gr.HTML(value=_idle_placeholder_html(idle_text), visible=True)
    audio = gr.Audio(label=label, type="filepath", visible=False, elem_classes="gen-audio-slot")
    return placeholder, audio


def build_ui():
    with gr.Blocks(title="Symbolic Music Demo") as demo:
        with gr.Column(elem_classes="hero-header"):
            gr.Markdown("# 🎷 Symbolic Music Generation 🎷", elem_classes="app-title")
            gr.Markdown(
                "Compare EBT, Llama, and GPT-2 on the same prompt, then explore EBT's "
                "attribute-guided generation. Models load once and are cached.",
                elem_classes="app-subtitle",
            )

        gr.Markdown("## 1. Choose format and models", elem_classes="section-title")
        gr.Markdown(
            "Tokenization format affects how the prompt itself gets read (its offset "
            "units, start-point precision, etc.), so it's chosen first — picking a "
            "prompt below won't need to get re-resolved if you change format afterward.",
            elem_classes="section-sub",
        )
        tokenizer_dd = gr.Dropdown(choices=TOKENIZER_CHOICES, value="REMI", label="Tokenization format")
        with gr.Accordion("⚙️ Advanced generation settings", open=False):
            with gr.Row():
                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                gen_length = gr.Slider(64, 1024, value=512, step=64, label="Generation length (tokens)")

        # By default no checkpoint dropdown is shown at all — the most recent
        # checkpoint is used automatically (see refresh_checkpoints/_find_
        # checkpoints). Each model card just shows training progress (% of
        # max steps, best checkpoint's validation loss) so the user can gauge
        # how far along it is without needing to pick anything. Explicit
        # checkpoint selection is tucked away below as a manual override.
        ckpt_states = {}
        ckpt_progress_htmls = {}
        model_controls = {}
        with gr.Row():
            for model_key in _MODEL_KEYS:
                with gr.Column(elem_classes="model-card"):
                    gr.Markdown(f"### {model_key}")
                    on_cb = gr.Checkbox(value=True, label=f"Include {model_key}")
                    ckpt_progress_html = gr.HTML(value=_ckpt_loading_html("REMI"))
                    ckpt_state = gr.State("{}")
                    model_controls[model_key] = [on_cb, None, None]
                    ckpt_states[model_key] = ckpt_state
                    ckpt_progress_htmls[model_key] = ckpt_progress_html

        with gr.Accordion("⚙️ Manually choose checkpoints (optional)", open=False):
            for model_key in _MODEL_KEYS:
                # The dropdown's own label sits above its value box, but a
                # plain gr.Button has no label — left as a default gr.Row
                # (which aligns children to the top), the button floated up
                # next to the dropdown's LABEL instead of its value box,
                # reading as misaligned/disconnected from the control it
                # actually acts on. align-items: flex-end lines it up with
                # the dropdown's value box instead.
                with gr.Row(elem_classes="ckpt-refresh-row"):
                    ckpt_dd = gr.Dropdown(choices=[], label=f"{model_key} REMI checkpoint", scale=4)
                    # Re-scans disk for checkpoints and reselects the most
                    # recent one only if nothing was already picked — see
                    # refresh_checkpoints. Spelled out since "↻" alone reads
                    # as "reset to some fixed default," which this isn't:
                    # there's no fixed default, and a manual pick survives
                    # a rescan as long as it's still on disk.
                    refresh_btn = gr.Button("↻ Rescan", scale=0, size="sm")
                model_controls[model_key][1] = ckpt_dd
                model_controls[model_key][2] = refresh_btn
        model_controls = {k: tuple(v) for k, v in model_controls.items()}

        gr.Markdown("## 2. Choose your prompt", elem_classes="section-title")
        gr.Markdown(
            "Pick a song and mark where the prompt should start by ear — the model "
            "generates a continuation from whatever's playing at that point.",
            elem_classes="section-sub",
        )
        song_dd = gr.Dropdown(choices=_SONG_CHOICES,
                               value=_SONG_CHOICES[1] if len(_SONG_CHOICES) > 1 else RANDOM_SONG_LABEL,
                               label="Song / Prompt")

        # ClickAudio's own `visible` prop can't be toggled after mount — a bug
        # in this custom component's build (confirmed via a headless-browser
        # repro: toggling `visible` on the component itself throws a Svelte
        # "effect_orphan" error that kills its whole reactive tree, while
        # toggling *value* alone, or toggling visibility on a wrapping
        # gr.Group, both work fine). So the component itself stays
        # permanently visible=True, and the group around it is what actually
        # hides/shows the section.
        with gr.Group(visible=False) as full_song_group:
            full_song_audio = ClickAudio(
                label="Full song — click anywhere on the waveform to set the prompt's start point",
                type="filepath", interactive=False, visible=True,
            )

        start_offset = gr.Slider(0, 2000, value=0, step=1, label="Prompt start offset (tokens)")
        preview_audio = gr.Audio(label="Prompt preview", type="filepath")
        preview_status = gr.Markdown("")

        generate_btn = gr.Button("🎹 Generate", variant="primary", size="lg")

        gr.Markdown("## 3. Listen to generated outputs", elem_classes="section-title")
        status_box = gr.Textbox(label="Status", lines=8, interactive=False)
        gr.Markdown(
            "Prompt and ground-truth continuation are the same across every model, "
            "shown once here — each model's card below only shows what it generated.",
            elem_classes="section-sub",
        )
        with gr.Row():
            with gr.Column(elem_classes="model-card"):
                shared_prompt_ph, shared_prompt_audio = _audio_slot(
                    "Prompt", idle_text=_slot_idle_text("shared_prompt"))
            with gr.Column(elem_classes="model-card"):
                shared_gt_ph, shared_gt_audio = _audio_slot(
                    "Prompt + Original Continuation", idle_text=_slot_idle_text("shared_gt"))

        output_slots = {"shared_prompt": (shared_prompt_ph, shared_prompt_audio),
                         "shared_gt": (shared_gt_ph, shared_gt_audio)}
        model_output_cols: dict[str, gr.Column] = {}
        with gr.Row():
            for model_key in _MODEL_KEYS:
                # Hidden until Generate is actually clicked (see generate_all's
                # initial reveal) — before that, an empty "Generated
                # Continuation" box for a model you haven't run yet is just
                # clutter, unlike the shared prompt/ground-truth boxes above
                # (those are legitimately "waiting on you to choose a prompt").
                with gr.Column(elem_classes=_card_classes(False)) as _model_col:
                    gr.Markdown(f"### {model_key}")
                    gen_ph, gen_audio = _audio_slot("Generated Continuation")
                    comb_ph, comb_audio = _audio_slot("Prompt + Generated Continuation")
                    output_slots[f"{model_key}_generated"] = (gen_ph, gen_audio)
                    output_slots[f"{model_key}_combined"] = (comb_ph, comb_audio)
                model_output_cols[model_key] = _model_col

        # ── EBT attribute control ────────────────────────────────────────────
        gr.Markdown("## 4. Guide EBT's attributes", elem_classes="section-title")
        ebt_reference_audio = gr.Audio(
            label="Reference: EBT default (unguided, from the last Generate run above)",
            type="filepath", visible=False,
        )
        solo_reference_html = gr.HTML(value=_render_compose_reference_html(None, None))
        ebt_ref_velocity_state = gr.State(None)
        ebt_ref_duration_state = gr.State(None)
        ebt_ref_pitch_register_state = gr.State(None)

        with gr.Column(elem_classes="attr-window"):
            gr.Markdown("### Single attribute — generate multiple at once")
            gr.Markdown(
                f"Each numbered box below is its own **separate variant** — its own "
                f"attribute, guidance strength λ, and target value. Start with one, use **+ Add "
                f"another variant** to queue more (up to {_MAX_SOLO_BATCH}), then hit Generate "
                "once to run all of them together.",
                elem_classes="section-sub",
            )

            solo_row_count_state = gr.State(1)
            solo_rows = []
            solo_sigma_htmls = []
            with gr.Column(elem_classes="variants-bundle"):
                for _i in range(_MAX_SOLO_BATCH):
                    with gr.Group(elem_classes=_card_classes(_i == 0)) as _row_group:
                        with gr.Row():
                            gr.Markdown(f"**Variant {_i + 1}**")
                            _row_copy_btn = gr.Button("⧉ Copy to new variant", size="sm", scale=0)
                        with gr.Row():
                            # Its own full-width row, not sharing one with the
                            # lambda/target sliders — three pill-style radio
                            # options need more horizontal room than a
                            # one-third-width column gives them, which was
                            # wrapping "pitch register" onto its own line.
                            _row_attr = gr.Radio(choices=_attr_choice_tuples(_SOLO_ATTRIBUTES),
                                                  value="velocity", label="Attribute")
                        with gr.Row():
                            _row_lambda = gr.Slider(_LAMBDA_RANGES["velocity"]["min"], _LAMBDA_RANGES["velocity"]["max"],
                                                     value=_LAMBDA_RANGES["velocity"]["default"],
                                                     step=_LAMBDA_RANGES["velocity"]["step"],
                                                     label="Guidance strength λ", scale=1)
                            _row_target = gr.Slider(0.0, _ATTRIBUTE_UI["velocity"]["max"],
                                                     value=_ATTRIBUTE_UI["velocity"]["default_target"],
                                                     step=0.01, label=_ATTRIBUTE_UI["velocity"]["label"], scale=1)
                        _row_sigma = gr.HTML(value=_sigma_annotation_html(
                            "velocity", _ATTRIBUTE_UI["velocity"]["default_target"]))
                    solo_rows.append((_row_group, _row_attr, _row_lambda, _row_target, _row_copy_btn))
                    solo_sigma_htmls.append(_row_sigma)

                    def _make_row_attr_change():
                        def _on_row_attr_change(attribute, ref_velocity, ref_duration, ref_pitch_register):
                            meta = _ATTRIBUTE_UI[attribute]
                            lam = _LAMBDA_RANGES[attribute]
                            target_update = gr.update(
                                minimum=0.0, maximum=meta["max"], label=meta["label"],
                                value=_target_default_value(attribute, ref_velocity, ref_duration, ref_pitch_register))
                            lambda_update = gr.update(minimum=lam["min"], maximum=lam["max"],
                                                       step=lam["step"], value=lam["default"])
                            return target_update, lambda_update
                        return _on_row_attr_change

                    _row_attr.change(_make_row_attr_change(),
                                      inputs=[_row_attr, ebt_ref_velocity_state, ebt_ref_duration_state,
                                              ebt_ref_pitch_register_state],
                                      outputs=[_row_target, _row_lambda]).then(
                        _sigma_annotation_html, inputs=[_row_attr, _row_target], outputs=[_row_sigma],
                    )
                    _row_target.change(_sigma_annotation_html, inputs=[_row_attr, _row_target],
                                        outputs=[_row_sigma])

                    def _make_row_tokenizer_change():
                        def _on_row_tokenizer_change(tokenizer_type, current_attr):
                            # Switching to Anticipation drops "velocity" from
                            # this row's choices (see _solo_attribute_choices)
                            # — if that row was on velocity, fall back to
                            # duration instead of leaving it selected on a
                            # choice that no longer exists.
                            choices = _solo_attribute_choices(tokenizer_type)
                            value = current_attr if current_attr in choices else choices[0]
                            return gr.update(choices=_attr_choice_tuples(choices), value=value)
                        return _on_row_tokenizer_change

                    tokenizer_dd.change(_make_row_tokenizer_change(),
                                         inputs=[tokenizer_dd, _row_attr], outputs=[_row_attr])

            with gr.Row():
                solo_add_row_btn = gr.Button("+ Add another variant", size="sm")
                solo_remove_row_btn = gr.Button("− Remove last variant", size="sm")

            def _set_solo_row_count(count: int, delta: int):
                count = max(1, min(_MAX_SOLO_BATCH, count + delta))
                return (count, *[gr.update(elem_classes=_card_classes(i < count)) for i in range(_MAX_SOLO_BATCH)])

            _solo_row_groups = [rg for rg, _, _, _, _ in solo_rows]
            # Disable the button for the duration of its own click handler —
            # a click that lands while a previous one is still in flight (the
            # round trip takes a moment, and nothing visually indicates
            # "processing" otherwise) was reported landing as two increments
            # instead of one, since a real HTML `disabled` attribute blocks
            # the second click at the browser level, not just cosmetically.
            # Changing the button's own TEXT (not just graying it out) while
            # its click is processing — a disabled-but-same-label button is
            # easy to read as "nothing happened" if the round trip has any
            # real latency, which is exactly what was reported.
            # "Add"/"Remove"/"Copy" all read-then-write the SAME shared
            # solo_row_count_state (current count -> new count). A shared
            # concurrency_id serializes every mutation of this counter
            # against every OTHER one, regardless of which of the three
            # controls triggered it, closing off a genuine (if narrower)
            # stale-read race between overlapping clicks. Note this is
            # NOT what caused the "every other click has no visible effect"
            # bug reported separately — that traced to a Gradio frontend
            # bug where gr.Column's `visible` prop doesn't reliably apply
            # when toggled repeatedly (reproduced in complete isolation,
            # with no app logic at all); the fix there was switching these
            # toggled containers from gr.Column to gr.Group, which doesn't
            # exhibit the bug.
            _ROW_MUTATION_CONCURRENCY = dict(concurrency_id="solo_row_mutation", concurrency_limit=1)
            solo_add_row_btn.click(
                lambda: gr.update(interactive=False, value="Adding…"), outputs=[solo_add_row_btn],
            ).then(
                lambda c: _set_solo_row_count(c, 1),
                inputs=[solo_row_count_state],
                outputs=[solo_row_count_state, *_solo_row_groups],
                **_ROW_MUTATION_CONCURRENCY,
            ).then(
                lambda: gr.update(interactive=True, value="+ Add another variant"), outputs=[solo_add_row_btn],
            )
            solo_remove_row_btn.click(
                lambda: gr.update(interactive=False, value="Removing…"), outputs=[solo_remove_row_btn],
            ).then(
                lambda c: _set_solo_row_count(c, -1),
                inputs=[solo_row_count_state],
                outputs=[solo_row_count_state, *_solo_row_groups],
                **_ROW_MUTATION_CONCURRENCY,
            ).then(
                lambda: gr.update(interactive=True, value="− Remove last variant"), outputs=[solo_remove_row_btn],
            )

            # "Copy" on variant i: reveal the next hidden variant (same as
            # "+ Add another variant") and pre-fill it with variant i's
            # current attribute/lambda/target, instead of the usual defaults.
            _solo_value_comps = []
            for _, _attr, _lam, _target, _ in solo_rows:
                _solo_value_comps += [_attr, _lam, _target]

            def _make_copy_handler(src_idx):
                def _copy(count, src_attr, src_lam, src_target):
                    count = int(count)
                    new_count = min(count + 1, _MAX_SOLO_BATCH)
                    dest_idx = new_count - 1
                    visibility_updates = [gr.update(elem_classes=_card_classes(i < new_count)) for i in range(_MAX_SOLO_BATCH)]
                    value_updates = []
                    for i in range(_MAX_SOLO_BATCH):
                        if i == dest_idx and dest_idx != src_idx:
                            value_updates.extend([gr.update(value=src_attr), gr.update(value=src_lam),
                                                   gr.update(value=src_target)])
                        else:
                            value_updates.extend([gr.update(), gr.update(), gr.update()])
                    return (new_count, *visibility_updates, *value_updates)
                return _copy

            for _i, (_, _row_attr, _row_lambda, _row_target, _row_copy_btn) in enumerate(solo_rows):
                _row_copy_btn.click(
                    lambda: gr.update(interactive=False, value="Copying…"), outputs=[_row_copy_btn],
                ).then(
                    _make_copy_handler(_i),
                    inputs=[solo_row_count_state, _row_attr, _row_lambda, _row_target],
                    outputs=[solo_row_count_state, *_solo_row_groups, *_solo_value_comps],
                    **_ROW_MUTATION_CONCURRENCY,
                ).then(
                    lambda: gr.update(interactive=True, value="⧉ Copy to new variant"), outputs=[_row_copy_btn],
                )

            solo_btn = gr.Button("🎹 Generate (Single Attribute)", variant="primary")
            solo_status = gr.Textbox(label="Status", lines=6, interactive=False)
            solo_slots = []
            with gr.Row(elem_classes="solo-output-row"):
                for _i in range(_MAX_SOLO_BATCH):
                    # Stays hidden until Generate is actually clicked — see
                    # _card_classes_slot_helpers()'s pending_slot/done_slot,
                    # which reveal each card only once its own result is
                    # ready. Without this, the first card showed an empty
                    # "Generated Continuation" player from the moment the
                    # page loads, before anything had been generated.
                    with gr.Group(elem_classes=_card_classes(False)) as _solo_col:
                        _solo_title = gr.HTML(value="")
                        _solo_ph, _solo_audio = _audio_slot("Generated Continuation")
                        _solo_gauge = gr.HTML(value="", elem_classes="attr-gauge-slot")
                        solo_slots.append((_solo_col, _solo_title, _solo_ph, _solo_audio, _solo_gauge))

        with gr.Column(elem_classes="attr-window"):
            gr.Markdown("### Compose — combine multiple attributes in one generation")
            gr.Markdown(
                "Different from **Single attribute** above: there, each row runs its "
                "own separate generation. Here, every active row below contributes to "
                "the **same** generation at once — their guidance is summed (each "
                "scaled by its own Relative weight), then the combined result is "
                "scaled by the one shared λ underneath.",
                elem_classes="section-sub",
            )
            compose_reference_html = gr.HTML(value=_render_compose_reference_html(None, None))
            compose_reference_audio = gr.Audio(
                label="Reference: EBT default (unguided, from the last Generate run above)",
                type="filepath", visible=False,
            )

            compose_row_count_state = gr.State(1)
            compose_rows = []
            compose_sigma_htmls = []
            with gr.Column(elem_classes="variants-bundle"):
                for _i in range(_MAX_COMPOSE_ROWS):
                    with gr.Group(elem_classes=_card_classes(_i == 0)) as _crow_group:
                        with gr.Row():
                            gr.Markdown(f"**Attribute {_i + 1}**")
                        with gr.Row():
                            _crow_attr = gr.Radio(
                                choices=_attr_choice_tuples(_SOLO_ATTRIBUTES),
                                value=_SOLO_ATTRIBUTES[min(_i, len(_SOLO_ATTRIBUTES) - 1)],
                                label="Attribute")
                        with gr.Row():
                            _crow_target = gr.Slider(0.0, _ATTRIBUTE_UI["velocity"]["max"],
                                                      value=_ATTRIBUTE_UI["velocity"]["default_target"],
                                                      step=0.01, label=_ATTRIBUTE_UI["velocity"]["label"], scale=2)
                            _crow_weight = gr.Slider(0.0, 2.0, value=1.0, step=0.1,
                                                      label="Relative weight (mix vs. the other rows)", scale=1)
                        _crow_sigma = gr.HTML(value=_sigma_annotation_html(
                            "velocity", _ATTRIBUTE_UI["velocity"]["default_target"]))
                    compose_rows.append((_crow_group, _crow_attr, _crow_target, _crow_weight))
                    compose_sigma_htmls.append(_crow_sigma)

                    def _make_crow_attr_change():
                        def _on_crow_attr_change(attribute, ref_velocity, ref_duration, ref_pitch_register):
                            meta = _ATTRIBUTE_UI[attribute]
                            return gr.update(
                                minimum=0.0, maximum=meta["max"], label=meta["label"],
                                value=_target_default_value(attribute, ref_velocity, ref_duration, ref_pitch_register))
                        return _on_crow_attr_change

                    _crow_attr.change(_make_crow_attr_change(),
                                       inputs=[_crow_attr, ebt_ref_velocity_state, ebt_ref_duration_state,
                                               ebt_ref_pitch_register_state],
                                       outputs=[_crow_target]).then(
                        _sigma_annotation_html, inputs=[_crow_attr, _crow_target], outputs=[_crow_sigma],
                    )
                    _crow_target.change(_sigma_annotation_html, inputs=[_crow_attr, _crow_target],
                                         outputs=[_crow_sigma])

                    def _make_crow_tokenizer_change():
                        def _on_crow_tokenizer_change(tokenizer_type, current_attr):
                            choices = _solo_attribute_choices(tokenizer_type)
                            value = current_attr if current_attr in choices else choices[0]
                            return gr.update(choices=_attr_choice_tuples(choices), value=value)
                        return _on_crow_tokenizer_change

                    tokenizer_dd.change(_make_crow_tokenizer_change(),
                                         inputs=[tokenizer_dd, _crow_attr], outputs=[_crow_attr])

            with gr.Row():
                compose_add_row_btn = gr.Button("+ Add another attribute", size="sm")
                compose_remove_row_btn = gr.Button("− Remove last attribute", size="sm")

            def _set_compose_row_count(count: int, delta: int):
                count = max(1, min(_MAX_COMPOSE_ROWS, count + delta))
                return (count, *[gr.update(elem_classes=_card_classes(i < count)) for i in range(_MAX_COMPOSE_ROWS)])

            _compose_row_groups = [rg for rg, _, _, _ in compose_rows]
            # Same button-disable + shared-concurrency pattern as the Single
            # attribute rows above — see _ROW_MUTATION_CONCURRENCY's comment
            # for why (a Gradio gr.Group visible-toggle bug, not app logic).
            _COMPOSE_ROW_CONCURRENCY = dict(concurrency_id="compose_row_mutation", concurrency_limit=1)
            compose_add_row_btn.click(
                lambda: gr.update(interactive=False, value="Adding…"), outputs=[compose_add_row_btn],
            ).then(
                lambda c: _set_compose_row_count(c, 1),
                inputs=[compose_row_count_state],
                outputs=[compose_row_count_state, *_compose_row_groups],
                **_COMPOSE_ROW_CONCURRENCY,
            ).then(
                lambda: gr.update(interactive=True, value="+ Add another attribute"), outputs=[compose_add_row_btn],
            )
            compose_remove_row_btn.click(
                lambda: gr.update(interactive=False, value="Removing…"), outputs=[compose_remove_row_btn],
            ).then(
                lambda c: _set_compose_row_count(c, -1),
                inputs=[compose_row_count_state],
                outputs=[compose_row_count_state, *_compose_row_groups],
                **_COMPOSE_ROW_CONCURRENCY,
            ).then(
                lambda: gr.update(interactive=True, value="− Remove last attribute"), outputs=[compose_remove_row_btn],
            )

            # One shared λ drives every active row (each row's own Relative
            # weight tunes its mix) — spans the union of every attribute's own
            # tuned range rather than any single one alone, since which
            # attributes are active can change row to row.
            compose_lambda = gr.Slider(
                min(r["min"] for r in _LAMBDA_RANGES.values()),
                max(r["max"] for r in _LAMBDA_RANGES.values()),
                value=0.04, step=0.005, label="Guidance strength λ (shared across all active rows)")
            compose_btn = gr.Button("🎹 Generate (Compose)", variant="primary")
            compose_status = gr.Textbox(label="Status", lines=4, interactive=False)
            # Hidden until Generate is clicked — see _compose_generate,
            # which reveals this once a real result (or an error) is ready.
            compose_audio = gr.Audio(label="Generated Continuation", type="filepath", visible=False)
            compose_gauge = gr.HTML(value="", elem_classes="attr-gauge-slot")

        # ── Event wiring ──────────────────────────────────────────────────────

        # Switching tokenizer fires all 3 of these at once (one per model) —
        # over a plain local connection that's harmless, but over the
        # --share gradio.live tunnel that burst of simultaneous requests was
        # reported causing "could not parse server response" (the tunnel
        # returning an HTML error page instead of JSON under the sudden
        # load). Serializing them against each other spreads the burst out
        # instead of firing all 3 through the tunnel in the same instant.
        _CKPT_REFRESH_CONCURRENCY = dict(concurrency_id="ckpt_refresh", concurrency_limit=1)
        for model_key in _MODEL_KEYS:
            on_cb, ckpt_dd, refresh_btn = model_controls[model_key]
            ckpt_state = ckpt_states[model_key]
            ckpt_progress_html = ckpt_progress_htmls[model_key]
            # show_progress="hidden": a closed accordion's status-tracker
            # widget only mounts once it's opened -- if that happens AFTER
            # this (sub-second) refresh already finished in the background,
            # the widget has no real "done" event left to receive and gets
            # stuck showing a live counter timing its own time-since-opened,
            # not the actual (already-finished) operation. Confirmed this
            # is timing-dependent, not caused by the concurrency_id above:
            # opening the accordion promptly, before the refresh completes,
            # never showed it; opening it later always did. Since the
            # refresh is fast and always correct regardless, there's no
            # real information the progress indicator was adding.
            # Disabling the dropdown for the duration of the refresh (instead
            # of firing refresh_checkpoints directly on click/change) closes
            # a race that was reported reverting a manual pick: refresh_
            # checkpoints captures ckpt_dd's value as `current_selection` the
            # MOMENT it's triggered, not when it finishes — so if the user
            # picked a different checkpoint while an earlier-triggered
            # refresh (e.g. from a tokenizer switch a moment before) was
            # still in flight, that refresh would later resolve using its
            # own stale snapshot and silently stomp the newer pick, which
            # looked like "the change is detected for a moment, then reverts
            # to default." Making the dropdown non-interactive until its own
            # refresh settles means a manual pick can only ever happen when
            # no refresh for this model is still pending.
            # The disable step also swaps the stats card to a spinning
            # "Loading {tokenizer} checkpoints…" message — with show_progress
            # ="hidden" on the actual refresh (see below), there was no
            # visible sign anything was happening at all, which read as
            # "stuck" during the (previously silent) scan, and gave no
            # confirmation a tokenizer switch had actually been picked up.
            # The disable step is ALSO grouped under _CKPT_REFRESH_CONCURRENCY
            # (limit=1, shared across all 3 models) — otherwise it fires 3
            # simultaneous requests the instant tokenizer_dd changes, which is
            # exactly the burst that broke the --share tunnel before (see the
            # concurrency_id's own original comment). Splitting the work into
            # two steps doesn't help if the FIRST step reintroduces the burst.
            refresh_btn.click(
                lambda tok: (gr.update(interactive=False), _ckpt_loading_html(tok)),
                inputs=[tokenizer_dd], outputs=[ckpt_dd, ckpt_progress_html],
                **_CKPT_REFRESH_CONCURRENCY,
            ).then(
                lambda tok, cur, mk=model_key: refresh_checkpoints(mk, tok, cur),
                inputs=[tokenizer_dd, ckpt_dd], outputs=[ckpt_dd, ckpt_state, ckpt_progress_html],
                show_progress="hidden",
                **_CKPT_REFRESH_CONCURRENCY,
            )
            tokenizer_dd.change(
                lambda tok: (gr.update(interactive=False), _ckpt_loading_html(tok)),
                inputs=[tokenizer_dd], outputs=[ckpt_dd, ckpt_progress_html],
                **_CKPT_REFRESH_CONCURRENCY,
            ).then(
                lambda tok, cur, mk=model_key: refresh_checkpoints(mk, tok, cur),
                inputs=[tokenizer_dd, ckpt_dd], outputs=[ckpt_dd, ckpt_state, ckpt_progress_html],
                show_progress="hidden",
                **_CKPT_REFRESH_CONCURRENCY,
            )
            # Manually picking a different checkpoint from this dropdown
            # (see the "Manually choose checkpoints" section) is what
            # actually gets used for generation — but without this, the
            # stats card above kept showing the auto-picked checkpoint's
            # numbers regardless, which looked like the manual pick hadn't
            # taken effect even though it had.
            ckpt_dd.change(_ckpt_stats_for_selection, inputs=[ckpt_dd, tokenizer_dd], outputs=[ckpt_progress_html],
                            show_progress="hidden")

        # Every chain below reads/writes the SAME shared prompt-selection
        # outputs (song_dd, start_offset, preview_audio, preview_status,
        # full_song_group/audio) from a DIFFERENT triggering event (tokenizer
        # switch, song pick, slider release, waveform click, plus the initial
        # demo.load()). Without serializing them, a slower call triggered
        # EARLIER (e.g. the initial page-load preview for the default REMI
        # song) can resolve LATER than a faster call triggered by a
        # subsequent user action (e.g. switching to Anticipation right
        # after page load), silently overwriting the newer, correct state
        # with stale results — confirmed exactly reproducing "switching
        # tokenizer breaks song choice/preview" when the tokenizer is
        # switched before the initial load's own preview finishes. A shared
        # concurrency_id forces them to run one at a time, in trigger order.
        _PROMPT_CHAIN_CONCURRENCY = dict(concurrency_id="prompt_chain", concurrency_limit=1)
        # Every previously-generated result (baseline Generate, Single
        # Attribute, Compose) depends on the prompt these three controls
        # define — changing any of them invalidates whatever's currently on
        # screen. Without this, the old prompt's audio/gauges just sat there
        # until the next Generate click quietly replaced them, which reads
        # as though they belong to the prompt now selected. A plain,
        # unconditional reset (not gated on _PROMPT_CHAIN_CONCURRENCY) — it
        # always produces the same idle output regardless of order, so it's
        # safe to run independently of the slower prompt-resolution chain.
        _clear_generated_outputs = [
            *(c for key in _SLOT_KEYS for c in output_slots[key]),
            ebt_reference_audio, ebt_ref_velocity_state, ebt_ref_duration_state, ebt_ref_pitch_register_state,
            compose_reference_audio,
            *(model_output_cols[k] for k in _MODEL_KEYS),
            status_box,
            solo_reference_html, compose_reference_html,
            *(c for slot in solo_slots for c in slot), solo_status,
            compose_audio, compose_status, compose_gauge,
        ]
        tokenizer_dd.change(_clear_generated_results, outputs=_clear_generated_outputs)
        song_dd.change(_clear_generated_results, outputs=_clear_generated_outputs)
        start_offset.release(_clear_generated_results, outputs=_clear_generated_outputs)

        tokenizer_dd.change(
            _clear_prompt_audio, outputs=[full_song_audio, preview_audio, preview_status],
            **_PROMPT_CHAIN_CONCURRENCY,
        ).then(
            on_song_or_tokenizer_change, inputs=[tokenizer_dd, song_dd], outputs=[start_offset],
            **_PROMPT_CHAIN_CONCURRENCY,
        ).then(
            render_full_song, inputs=[tokenizer_dd, song_dd], outputs=[full_song_group, full_song_audio],
            **_PROMPT_CHAIN_CONCURRENCY,
        ).then(
            preview_prompt, inputs=[tokenizer_dd, song_dd, start_offset], outputs=[preview_audio, preview_status],
            **_PROMPT_CHAIN_CONCURRENCY,
        )
        song_dd.change(
            _clear_prompt_audio, outputs=[full_song_audio, preview_audio, preview_status],
            **_PROMPT_CHAIN_CONCURRENCY,
        ).then(
            on_song_or_tokenizer_change, inputs=[tokenizer_dd, song_dd], outputs=[start_offset],
            **_PROMPT_CHAIN_CONCURRENCY,
        ).then(
            render_full_song, inputs=[tokenizer_dd, song_dd], outputs=[full_song_group, full_song_audio],
            **_PROMPT_CHAIN_CONCURRENCY,
        ).then(
            preview_prompt, inputs=[tokenizer_dd, song_dd, start_offset], outputs=[preview_audio, preview_status],
            **_PROMPT_CHAIN_CONCURRENCY,
        )

        start_offset.release(preview_prompt, inputs=[tokenizer_dd, song_dd, start_offset],
                              outputs=[preview_audio, preview_status], **_PROMPT_CHAIN_CONCURRENCY)

        full_song_audio.seek(
            apply_seek_as_start, inputs=[tokenizer_dd, song_dd],
            outputs=[start_offset, preview_status],
            **_PROMPT_CHAIN_CONCURRENCY,
        ).then(
            preview_prompt, inputs=[tokenizer_dd, song_dd, start_offset],
            outputs=[preview_audio, preview_status],
            **_PROMPT_CHAIN_CONCURRENCY,
        )

        all_ckpt_inputs = []
        for model_key in _MODEL_KEYS:
            on_cb, ckpt_dd, _ = model_controls[model_key]
            all_ckpt_inputs += [on_cb, ckpt_dd, ckpt_states[model_key]]

        all_outputs = []
        for key in _SLOT_KEYS:
            all_outputs += list(output_slots[key])

        generate_call = generate_btn.click(
            generate_all,
            inputs=[tokenizer_dd, song_dd, start_offset, temperature, top_p, gen_length, *all_ckpt_inputs],
            outputs=[*all_outputs, ebt_reference_audio, ebt_ref_velocity_state, ebt_ref_duration_state,
                     ebt_ref_pitch_register_state, compose_reference_audio,
                     *[model_output_cols[k] for k in _MODEL_KEYS], status_box],
            concurrency_limit=1,
        ).then(
            _render_compose_reference_html,
            inputs=[ebt_ref_velocity_state, ebt_ref_duration_state, ebt_ref_pitch_register_state],
            outputs=[solo_reference_html],
        ).then(
            _render_compose_reference_html,
            inputs=[ebt_ref_velocity_state, ebt_ref_duration_state, ebt_ref_pitch_register_state],
            outputs=[compose_reference_html],
        )
        # Once a reference becomes available, snap every target slider that
        # hasn't been deliberately moved yet to it too — before that, "no
        # adjustment" should mean "matches EBT's own unguided output", not an
        # arbitrary fixed number unrelated to what the model actually produces.
        for _rg, _r_attr, _r_lam, _r_target, _r_copy in solo_rows:
            generate_call = generate_call.then(
                lambda attribute, rv, rd, rp: gr.update(value=_target_default_value(attribute, rv, rd, rp)),
                inputs=[_r_attr, ebt_ref_velocity_state, ebt_ref_duration_state, ebt_ref_pitch_register_state],
                outputs=[_r_target],
            )
        for _crg, _cr_attr, _cr_target, _cr_weight in compose_rows:
            generate_call = generate_call.then(
                lambda attribute, rv, rd, rp: gr.update(value=_target_default_value(attribute, rv, rd, rp)),
                inputs=[_cr_attr, ebt_ref_velocity_state, ebt_ref_duration_state, ebt_ref_pitch_register_state],
                outputs=[_cr_target],
            )

        ebt_on_cb, ebt_ckpt_dd, _ = model_controls["EBT"]
        ebt_ckpt_state = ckpt_states["EBT"]

        solo_batch_outputs = []
        for _col, _title, _ph, _audio, _gauge in solo_slots:
            solo_batch_outputs += [_col, _title, _ph, _audio, _gauge]

        solo_row_inputs = []
        for _rg, _attr, _lam, _target, _copy_btn in solo_rows:
            solo_row_inputs += [_attr, _lam, _target]

        solo_batch_state = gr.State({})
        # One shared concurrency_id across the click AND every chained
        # per-row step, so a second click can't interleave its own rows with
        # an in-flight batch's (which would corrupt the shared status text
        # and could apply one run's row to another's slot).
        _SOLO_BATCH_CONCURRENCY = dict(concurrency_id="solo_batch_generation", concurrency_limit=1)
        _solo_chain = solo_btn.click(
            _solo_batch_start,
            inputs=[tokenizer_dd, song_dd, start_offset, temperature, top_p, gen_length,
                    ebt_ckpt_dd, ebt_ckpt_state, solo_row_count_state, *solo_row_inputs],
            outputs=[*solo_batch_outputs, solo_status, solo_batch_state],
            **_SOLO_BATCH_CONCURRENCY,
        )
        for _i, (_col, _title, _ph, _audio, _gauge) in enumerate(solo_slots):
            _solo_chain = _solo_chain.then(
                _make_solo_row_step(_i),
                inputs=[solo_batch_state, solo_status],
                outputs=[_col, _title, _ph, _audio, _gauge, solo_status],
                **_SOLO_BATCH_CONCURRENCY,
            )

        compose_row_inputs = []
        for _crg, _cr_attr, _cr_target, _cr_weight in compose_rows:
            compose_row_inputs += [_cr_attr, _cr_target, _cr_weight]

        _COMPOSE_GEN_CONCURRENCY = dict(concurrency_id="compose_generation", concurrency_limit=1)
        compose_btn.click(
            _compose_start,
            inputs=[],
            outputs=[compose_audio, compose_status, compose_gauge],
            **_COMPOSE_GEN_CONCURRENCY,
        ).then(
            _compose_generate,
            inputs=[tokenizer_dd, song_dd, start_offset, temperature, top_p, gen_length,
                    ebt_ckpt_dd, ebt_ckpt_state, compose_lambda,
                    compose_row_count_state, *compose_row_inputs],
            outputs=[compose_audio, compose_status, compose_gauge],
            **_COMPOSE_GEN_CONCURRENCY,
        )

        # Initial load: populate all checkpoints and preview the default song.
        # show_progress="hidden" here for the same reason as the
        # tokenizer-switch handlers above — these dropdowns sit inside a
        # closed accordion, and its status-tracker widget only mounts once
        # the user opens it. If that happens after this (sub-second)
        # refresh already finished in the background (the common case,
        # since it's an optional, initially-collapsed section), the widget
        # has no real "done" event left to receive and gets stuck showing a
        # live counter timing its own time-since-opened, not the actual
        # operation — confirmed this is about accordion-open timing, not
        # the concurrency_id grouping below. Keeping the grouping anyway:
        # it still protects the initial page load from the same
        # --share-tunnel burst risk the tokenizer-switch fix addresses.
        for model_key in _MODEL_KEYS:
            _, ckpt_dd, _ = model_controls[model_key]
            demo.load(lambda mk=model_key: refresh_checkpoints(mk, "REMI"),
                      outputs=[ckpt_dd, ckpt_states[model_key], ckpt_progress_htmls[model_key]],
                      show_progress="hidden", **_CKPT_REFRESH_CONCURRENCY)
        # on_song_or_tokenizer_change is otherwise only wired to .change() on
        # the dropdowns, which never fires for their own initial values — the
        # default song (a real REMI song) needs seconds-mode applied to the
        # start_offset slider (range/step/label) from the very first render,
        # or it's stuck showing the 0-2000/step=1 token defaults while
        # already being *interpreted* as seconds underneath — confirmed via
        # direct inspection to display "Prompt start offset (tokens)" on a
        # fresh load despite the default song being in seconds mode.
        demo.load(on_song_or_tokenizer_change, inputs=[tokenizer_dd, song_dd],
                  outputs=[start_offset], **_PROMPT_CHAIN_CONCURRENCY)
        demo.load(preview_prompt, inputs=[tokenizer_dd, song_dd, start_offset],
                  outputs=[preview_audio, preview_status], **_PROMPT_CHAIN_CONCURRENCY)
        demo.load(render_full_song, inputs=[tokenizer_dd, song_dd],
                  outputs=[full_song_group, full_song_audio], **_PROMPT_CHAIN_CONCURRENCY)

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        help="Create a public HuggingFace tunnel (no SSH forwarding needed)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_ui()
    # Gradio's own default (when .queue() is never called) caps the WHOLE
    # app at concurrency_limit=1 globally — meaning any lightweight UI event
    # (add/remove/copy variant, dropdown changes, etc.) silently queues
    # behind a slow generation still in flight, appearing to do nothing
    # until the generation finishes. Raised here so those aren't blocked;
    # the three generation buttons explicitly set their own concurrency_limit=1
    # (see below) to preserve "only one generation at a time" — this GPU is
    # shared and not meant to run multiple generations simultaneously.
    app.queue(default_concurrency_limit=8)
    # Every generated/preview WAV this demo produces is written via raw
    # tempfile.mkdtemp() calls (one per render, scattered across
    # _run_generation/preview_prompt/_get_full_song_render/etc.) — these land
    # directly under the system temp root, NOT under Gradio's own
    # self-managed temp subtree or the app's working directory, which are the
    # only roots Gradio serves files from by default. Without allowed_paths,
    # Gradio can silently refuse to serve an otherwise-valid gr.Audio path:
    # the component still renders (visible=True), but the browser's fetch for
    # the actual audio bytes fails, showing no player at all — reported as
    # "the gauge shows but there's no audio," intermittently, across every
    # section of the demo, exactly matching this failure mode.
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
               theme=_THEME, css=_CSS, allowed_paths=[tempfile.gettempdir()])
