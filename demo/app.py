#!/usr/bin/env python3
"""
Gradio demo for EBT symbolic music generation.

Launch (interactive GPU session):
    srun --gpus=1 --mem=40GB --time=2:00:00 --partition=mit_normal_gpu \
         --account=mit_amf_standard_gpu --qos=mit_amf_standard_gpu --pty bash
    cd ~/music-EBT && conda activate music_EBT
    python demo/app.py [--share]   # --share creates a public HuggingFace tunnel

Local access (from laptop, replace NODE with the compute node name):
    ssh -L 7860:NODE:7860 rebcecca@eosloan.mit.edu
    open http://localhost:7860
"""

import os
import re
import sys
import json
import random
import tempfile
import traceback
from pathlib import Path
from argparse import Namespace

import torch
import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRATCH_DIR = Path.home() / "orcd/scratch/rebcecca/music_EBT_logs"
PROMPTS_FILE = ROOT / "demo/selected_prompts.json"
WANDB_PROJECT = "music_inference_ebt"
WANDB_ENTITY  = "rceccatelli-eth-z-rich"

RANDOM_SONG_LABEL = "🎲 Random song"

# ── Song list ─────────────────────────────────────────────────────────────────

def _load_song_choices() -> list[str]:
    """Return [RANDOM_SONG_LABEL] + sorted song titles from selected_prompts.json."""
    try:
        with open(PROMPTS_FILE) as f:
            songs = json.load(f)
        return [RANDOM_SONG_LABEL] + sorted(songs.keys())
    except Exception:
        return [RANDOM_SONG_LABEL]


def _tokens_for_song(song_title: str, prompts: dict) -> list[int]:
    """Load token ids for a named song from its pre-tokenized JSON file."""
    entry = prompts[song_title]
    midi_path = entry["path"] if isinstance(entry, dict) else entry
    start_token = entry.get("start_token", 0) if isinstance(entry, dict) else 0
    token_path = midi_path.replace("/midi/", "/tokens/miditok/").replace(".mid", ".json")
    with open(token_path) as f:
        return json.load(f)["ids"][start_token:]


_SONG_CHOICES = _load_song_choices()

# ── Model registry ────────────────────────────────────────────────────────────

_MODELS = {
    "EBT small / REMI": {
        "ckpt_pattern":  "ebt-symb-small-remi*",
        "model_name":    "ebt",
        "tokenizer_type": "REMI",
    },
    "Llama small / REMI": {
        "ckpt_pattern":  "baseline-llama-small-remi*",
        "model_name":    "baseline_llama_transformer",
        "tokenizer_type": "REMI",
    },
    "GPT2 small / REMI": {
        "ckpt_pattern":  "baseline-hf-gpt2-small-remi*",
        "model_name":    "baseline_hf_gpt2_transformer",
        "tokenizer_type": "REMI",
    },
}

# ── Checkpoint discovery ──────────────────────────────────────────────────────

def _find_best_checkpoints(ckpt_pattern: str) -> list[tuple[str, str]]:
    """Return [(display_label, ckpt_path)] sorted best-first (lowest val loss)."""
    ckpt_root = SCRATCH_DIR / "checkpoints"
    if not ckpt_root.exists():
        return []

    best: dict[str, tuple[float, str]] = {}  # run_dir -> (loss, ckpt_path)
    for run_dir in sorted(ckpt_root.glob(ckpt_pattern)):
        if not run_dir.is_dir():
            continue
        for ckpt in run_dir.glob("epoch=*.ckpt"):
            m = re.search(r'valid_loss=(\d+\.\d+)', str(ckpt))
            if not m:
                continue
            loss = float(m.group(1))
            key = str(run_dir)
            if key not in best or loss < best[key][0]:
                best[key] = (loss, str(ckpt))

    results = sorted(best.values(), key=lambda x: x[0])
    return [(_ckpt_label(path, loss), path) for loss, path in results]


MAX_STEPS = 100_000

def _ckpt_label(ckpt_path: str, loss: float) -> str:
    stem = Path(ckpt_path).stem
    step_m = re.search(r'step=(\d+)', stem)
    step = int(step_m.group(1)) if step_m else None
    run_dir = Path(ckpt_path).parent.name
    date_m = re.search(r'(\d{4}-\d{2}-\d{2})', run_dir)
    date = f" · {date_m.group(1)}" if date_m else ""
    if step is not None:
        pct = step / MAX_STEPS * 100
        return f"step {step:,} ({pct:.1f}%) · val_loss {loss:.4f}{date}"
    return f"val_loss {loss:.4f}{date}"


def refresh_checkpoints(model_label: str) -> tuple:
    pairs = _find_best_checkpoints(_MODELS[model_label]["ckpt_pattern"])
    if not pairs:
        return (
            gr.update(choices=[], value=None, label="Checkpoint"),
            "",
            "No checkpoints found for this model yet — training is still running. "
            "Click ↻ to refresh once a checkpoint has been saved (every 100 steps for EBT, every 500 for baselines).",
        )
    labels = [lbl for lbl, _ in pairs]
    paths  = {lbl: path for lbl, path in pairs}
    return (
        gr.update(choices=labels, value=labels[0], label=f"Checkpoint ({len(labels)} found)"),
        json.dumps(paths),
        f"Loaded {len(labels)} checkpoint(s) for {model_label}.",
    )

# ── Model + tokenizer cache ───────────────────────────────────────────────────

_model_cache: dict[str, tuple] = {}  # ckpt_path -> (model, hparams, tokenizer)


def _load(model_label: str, ckpt_path: str, device: str):
    if ckpt_path in _model_cache:
        return _model_cache[ckpt_path]

    model_name = _MODELS[model_label]["model_name"]

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


# ── Dataset cache ─────────────────────────────────────────────────────────────

_dataset_cache: dict[str, object] = {}


def _get_dataset(hparams):
    from inference.mus.infer_baselines_interactive import prepare_dataset
    key = hparams.tokenizer_type
    if key not in _dataset_cache:
        _dataset_cache[key] = prepare_dataset(hparams, split="validation")
    return _dataset_cache[key]


def _get_preview_dataset_and_tokenizer(model_label: str, ckpt_path: str | None):
    """Return (dataset, tokenizer) for prompt preview without loading the full model.

    If the model is already cached we reuse its tokenizer. Otherwise we load
    only the tokenizer (fast, ~1 s) and the validation dataset.
    """
    from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
    from inference.mus.infer_ebt import load_dataset as ebt_load_dataset

    # Fast path: model already in GPU cache — reuse everything
    if ckpt_path and ckpt_path in _model_cache:
        _, hparams, tokenizer = _model_cache[ckpt_path]
        key = hparams.tokenizer_type
        if key not in _dataset_cache:
            _dataset_cache[key] = ebt_load_dataset(hparams, split="validation")
        return _dataset_cache[key], tokenizer

    # Slow path: build minimal hparams from registry (no GPU needed)
    info = _MODELS[model_label]
    hparams = Namespace(
        tokenizer_type=info["tokenizer_type"],
        tokenizer_config_path=(
            "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json"
        ),
        dataset_name="giga_midi",
        context_length=512,
        validation_split_pct=0.05,
    )
    key = hparams.tokenizer_type
    tokenizer, _, _ = load_tokenizer(
        tokenizer_type=hparams.tokenizer_type,
        tokenizer_config_path=hparams.tokenizer_config_path,
        dataset_name=hparams.dataset_name,
    )
    if key not in _dataset_cache:
        _dataset_cache[key] = ebt_load_dataset(hparams, split="validation")
    return _dataset_cache[key], tokenizer


# ── Prompt preview ───────────────────────────────────────────────────────────

PROMPT_LENGTH = 256  # tokens; ~30–60 s of REMI music


def preview_prompt(model_label: str, ckpt_display: str, ckpt_paths_json: str,
                   song_choice: str):
    """Convert a dataset sample to WAV without running the model."""
    try:
        ckpt_path = None
        if ckpt_display and ckpt_paths_json:
            try:
                ckpt_path = json.loads(ckpt_paths_json).get(ckpt_display)
            except Exception:
                pass

        if song_choice and song_choice != RANDOM_SONG_LABEL:
            with open(PROMPTS_FILE) as f:
                prompts = json.load(f)
            all_tokens = _tokens_for_song(song_choice, prompts)
            tokens = all_tokens[:PROMPT_LENGTH]
            from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
            info = _MODELS[model_label]
            tokenizer, _, _ = load_tokenizer(
                tokenizer_type=info["tokenizer_type"],
                tokenizer_config_path=(
                    "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json"
                ),
                dataset_name="giga_midi",
            )
            label = f"Previewing: {song_choice}"
        else:
            dataset, tokenizer = _get_preview_dataset_and_tokenizer(model_label, ckpt_path)
            sample_idx = random.randint(0, len(dataset) - 1)
            if hasattr(dataset, 'get_full_tokens'):
                tokens = dataset.get_full_tokens(sample_idx)[:PROMPT_LENGTH]
            else:
                tokens = [int(t) for t in dataset[sample_idx]['input_ids'][:PROMPT_LENGTH]]
            label = f"Previewing random validation sample {sample_idx}"

        from inference.mus.tokens_to_midi import tokens_to_midi
        from demo.convert_midi_simple import simple_synth

        out_dir = Path(tempfile.mkdtemp(prefix="ebt_preview_"))
        midi_path = out_dir / "preview.mid"
        wav_path  = out_dir / "preview.wav"

        midi_bytes = tokens_to_midi(tokens, tokenizer)
        midi_path.write_bytes(midi_bytes)
        simple_synth(str(midi_path), str(wav_path))

        if wav_path.exists():
            return str(wav_path), label
        return None, "⚠️ WAV synthesis failed (MIDI converted but synth produced no output)."
    except Exception as e:
        return None, f"❌ Preview failed:\n{e}\n{traceback.format_exc()}"


# ── Generation ────────────────────────────────────────────────────────────────

def generate(
    model_label: str,
    ckpt_display: str,
    ckpt_paths_json: str,
    temperature: float,
    top_p: float,
    generation_length: int,
    song_choice: str,
):
    _loading = lambda lbl: gr.update(value=None, label=f"⏳ {lbl}")
    _done    = lambda lbl, v: gr.update(value=v, label=lbl)

    # Clear outputs immediately so stale audio doesn't linger
    yield (
        _loading("Prompt (input)"),
        _loading("Generated continuation"),
        _loading("Prompt + generated"),
        _loading("Prompt + original continuation (ground truth)"),
        "Starting…",
    )

    if not ckpt_display or not ckpt_paths_json:
        yield None, None, None, None, "❌ Select a model and checkpoint first."
        return

    try:
        ckpt_paths = json.loads(ckpt_paths_json)
    except Exception:
        yield None, None, None, None, "❌ Checkpoint state corrupted — click Refresh."
        return

    ckpt_path = ckpt_paths.get(ckpt_display)
    if not ckpt_path:
        yield None, None, None, None, f"❌ Could not resolve checkpoint path for: {ckpt_display}"
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    status_lines = [f"Device: {device} | Checkpoint: {Path(ckpt_path).parent.name}"]

    try:
        cached = ckpt_path in _model_cache
        yield None, None, None, None, "\n".join(status_lines + [
            "⏳ Loading model (cached)…" if cached else "⏳ Loading model (first load — may take ~30s)…"
        ])
        model, hparams, tokenizer = _load(model_label, ckpt_path, device)
        status_lines.append("✅ Model loaded (cached)" if cached else "✅ Model loaded")
    except Exception as e:
        yield None, None, None, None, f"❌ Model load failed:\n{e}\n{traceback.format_exc()}"
        return

    # Generation hyperparams
    hparams.infer_temp      = temperature
    hparams.infer_topp      = top_p
    hparams.infer_max_gen_len = generation_length
    hparams.infer_logprobs  = False
    hparams.infer_echo      = False

    try:
        if song_choice and song_choice != RANDOM_SONG_LABEL:
            with open(PROMPTS_FILE) as f:
                prompts = json.load(f)
            full_tokens = _tokens_for_song(song_choice, prompts)
            status_lines.append(f"Prompt: {song_choice}")
        else:
            dataset = _get_dataset(hparams)
            sample_idx = random.randint(0, len(dataset) - 1)
            if hasattr(dataset, 'get_full_tokens'):
                full_tokens = dataset.get_full_tokens(sample_idx)
            else:
                full_tokens = dataset[sample_idx]['input_ids'].tolist()
            status_lines.append(f"Prompt: random validation sample {sample_idx}")
        prompt_tokens = torch.tensor(full_tokens[:PROMPT_LENGTH], dtype=torch.long).unsqueeze(0).to(device)
        gt_tokens = full_tokens[PROMPT_LENGTH:PROMPT_LENGTH + generation_length]
    except Exception as e:
        yield None, None, None, None, f"❌ Dataset error:\n{e}\n{traceback.format_exc()}"
        return

    try:
        status_lines.append("⏳ Generating tokens…")
        yield None, None, None, None, "\n".join(status_lines)
        model_name = _MODELS[model_label]["model_name"]
        batch = {'input_ids': prompt_tokens}

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

        generated_tokens = outputs['generation_tokens'][0]
        status_lines[-1] = f"✅ Generated {len(generated_tokens)} tokens"
        status_lines.append("⏳ Converting to audio…")
        yield None, None, None, None, "\n".join(status_lines)
    except Exception as e:
        yield None, None, None, None, f"❌ Generation failed:\n{e}\n{traceback.format_exc()}"
        return

    try:
        from inference.mus.tokens_to_midi import tokens_to_midi
        from demo.convert_midi_simple import simple_synth

        out_dir = Path(tempfile.mkdtemp(prefix="ebt_demo_"))
        prompt_list      = full_tokens[:PROMPT_LENGTH]
        gen_list         = [int(t) for t in generated_tokens]
        combined_list    = prompt_list + gen_list
        gt_combined_list = prompt_list + gt_tokens

        wav_paths = {}
        for name, tokens in [
            ("prompt",       prompt_list),
            ("generated",    gen_list),
            ("combined",     combined_list),
            ("gt_combined",  gt_combined_list),
        ]:
            midi_path = out_dir / f"{name}.mid"
            wav_path  = out_dir / f"{name}.wav"
            try:
                midi_bytes = tokens_to_midi(tokens, tokenizer)
                midi_path.write_bytes(midi_bytes)
                simple_synth(str(midi_path), str(wav_path))
                wav_paths[name] = str(wav_path) if wav_path.exists() else None
            except Exception as e:
                status_lines.append(f"⚠️ {name} conversion failed: {e}")
                wav_paths[name] = None
    except Exception as e:
        yield None, None, None, None, f"❌ MIDI/WAV conversion failed:\n{e}\n{traceback.format_exc()}"
        return

    try:
        import wandb
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                         name=f"demo_{model_label}_{seed}", reinit=True)
        for name, path in wav_paths.items():
            if path:
                wandb.log({f"audio/{name}": wandb.Audio(path, sample_rate=44100)})
        log_meta = {
            "model": model_label, "checkpoint": ckpt_display,
            "temperature": temperature, "top_p": top_p,
            "generation_length": generation_length,
        }
        if song_choice and song_choice != RANDOM_SONG_LABEL:
            log_meta["song"] = song_choice
        else:
            log_meta["sample_idx"] = sample_idx
        wandb.log(log_meta)
        wandb.finish()
        status_lines.append("✅ Logged to WandB")
    except Exception as e:
        status_lines.append(f"⚠️ WandB logging failed: {e}")

    yield (
        _done("Prompt (input)",                                 wav_paths.get("prompt")),
        _done("Generated continuation",                         wav_paths.get("generated")),
        _done("Prompt + generated",                             wav_paths.get("combined")),
        _done("Prompt + original continuation (ground truth)",  wav_paths.get("gt_combined")),
        "\n".join(status_lines),
    )


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="EBT Music Demo") as demo:
        gr.Markdown("# EBT Symbolic Music Generation")
        gr.Markdown(
            "Select a model, pick a checkpoint, adjust generation parameters, then hit **Generate**. "
            "The model is cached after the first load — subsequent generations are fast."
        )

        # Hidden state: JSON map of {display_label: ckpt_path}
        ckpt_paths_state = gr.State("{}")

        with gr.Row():
            # ── Left column: controls ─────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Model")
                model_dd = gr.Dropdown(
                    choices=list(_MODELS.keys()),
                    value=list(_MODELS.keys())[0],
                    label="Model",
                )
                with gr.Row():
                    ckpt_dd = gr.Dropdown(choices=[], label="Checkpoint", scale=4)
                    refresh_btn = gr.Button("↻", scale=1, size="sm")

                gr.Markdown("### Generation")
                song_dd = gr.Dropdown(
                    choices=_SONG_CHOICES,
                    value=_SONG_CHOICES[1] if len(_SONG_CHOICES) > 1 else RANDOM_SONG_LABEL,
                    label="Song / Prompt",
                )
                preview_audio = gr.Audio(label="Prompt preview", type="filepath")

                temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="Temperature")
                top_p       = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                gen_length  = gr.Slider(64, 1024, value=512, step=64, label="Generation length (tokens)")

                generate_btn = gr.Button("Generate", variant="primary")

            # ── Right column: outputs ─────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Output")
                prompt_audio      = gr.Audio(label="Prompt (input)", type="filepath")
                generated_audio   = gr.Audio(label="Generated continuation", type="filepath")
                combined_audio    = gr.Audio(label="Prompt + generated", type="filepath")
                gt_combined_audio = gr.Audio(label="Prompt + original continuation (ground truth)", type="filepath")
                status_box = gr.Textbox(label="Status", lines=6, interactive=False)

        # ── Event wiring ──────────────────────────────────────────────────────

        def on_refresh(model_label):
            return refresh_checkpoints(model_label)

        def on_model_change(model_label):
            return refresh_checkpoints(model_label)

        refresh_btn.click(on_refresh, inputs=[model_dd], outputs=[ckpt_dd, ckpt_paths_state, status_box])
        model_dd.change(on_model_change, inputs=[model_dd], outputs=[ckpt_dd, ckpt_paths_state, status_box])

        song_dd.change(
            preview_prompt,
            inputs=[model_dd, ckpt_dd, ckpt_paths_state, song_dd],
            outputs=[preview_audio, status_box],
        )

        generate_btn.click(
            generate,
            inputs=[model_dd, ckpt_dd, ckpt_paths_state,
                    temperature, top_p, gen_length, song_dd],
            outputs=[prompt_audio, generated_audio, combined_audio, gt_combined_audio, status_box],
        )

        # Load checkpoints and preview default song on startup
        demo.load(on_refresh, inputs=[model_dd], outputs=[ckpt_dd, ckpt_paths_state, status_box])
        demo.load(
            preview_prompt,
            inputs=[model_dd, ckpt_dd, ckpt_paths_state, song_dd],
            outputs=[preview_audio, status_box],
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        help="Create a public HuggingFace tunnel (no SSH forwarding needed)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
               theme=gr.themes.Soft())
