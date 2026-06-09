#!/usr/bin/env python3
"""Upload inference run outputs (piano rolls, audio, log) to wandb."""

import re
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


_TOKENIZER_LABELS = {
    "REMI": "remi",
    "Anticipation-Arrival-Time": "ant-full",
    "Anticipation-Vanilla": "ant-vanilla",
    "Anticipation-Interarrival": "ant-interarrival",
}


def _tokenizer_label(config: dict) -> str:
    for model_name in ("gpt2", "llama"):
        tt = config.get(model_name, {}).get("tokenizer_type")
        if tt:
            return _TOKENIZER_LABELS.get(tt, tt.lower().replace(" ", "-"))
    return "unknown-tok"


def _present_models(config: dict) -> list:
    return [m for m in ("gpt2", "llama") if config.get(m)]


def _checkpoint_label(ckpt_path: str) -> str:
    """Extract a short label from a checkpoint path.

    e.g. '...epoch=epoch=0-step=step=1700-valid_loss=valid_loss=2.1304.ckpt'
    -> 's1700-vl2.1304'
    """
    if not ckpt_path:
        return "unk"
    stem = Path(ckpt_path).stem
    step_match = re.search(r'step=(\d+)', stem)
    loss_match = re.search(r'valid_loss=([\d.]+)', stem)
    step_part = f"s{step_match.group(1)}" if step_match else "sunk"
    loss_part = f"-vl{loss_match.group(1)}" if loss_match else ""
    return step_part + loss_part


def upload_run(run_dir: Path, wandb_project: str, wandb_entity: str, log_file: Path = None, temperature: float = None):
    import wandb

    # Detect if this is an EBT run (flat structure) or baseline run (gpt2/llama subdirs)
    is_ebt_run = (run_dir / "midi").exists() and (run_dir / "wav").exists()

    if is_ebt_run:
        # EBT inference: flat structure with midi/, wav/, piano_rolls/, tokens/ subdirs
        timestamp_match = re.search(r'(\d{8}_\d{6})', run_dir.name)
        timestamp = timestamp_match.group(1) if timestamp_match else run_dir.name
        # e.g. "ebt_ant-at-full_20260608_163104" → tok_slug = "ant-at-full"
        tok_slug_match = re.match(r'ebt_(.+?)_\d{8}_\d{6}', run_dir.name)
        tok_prefix = tok_slug_match.group(1) if tok_slug_match else "unknown-tok"
        temp_tag = f"-t{temperature}" if temperature is not None else ""
        run_name = f"{tok_prefix}-ebt-infer{temp_tag}-{timestamp}"

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity or None,
            name=run_name,
            tags=["inference", "symbolic", "ebt"],
        )

        log_dict = {}
        midi_dir        = run_dir / "midi"
        wav_dir         = run_dir / "wav"
        piano_rolls_dir = run_dir / "piano_rolls"

        # (kind, audio_rank, roll_rank) — rank prefixes drive alphabetical sort in wandb
        UPLOAD_KINDS = [
            ("prompt",       "01", "02"),
            ("ground_truth", "03", "04"),
            ("generated",    "05", "06"),
        ]

        sample_indices = sorted(set(
            f.stem.split("_")[1]
            for f in midi_dir.glob("sample_*_generated.mid")
        ), key=int)

        _AUDIO_SIZE_LIMIT = 50 * 1024 * 1024   # 50 MB — wandb renders fail above this

        for sample_idx in sample_indices:
            for kind, audio_rank, roll_rank in UPLOAD_KINDS:
                wav_file = wav_dir / f"sample_{sample_idx}_{kind}.wav"
                if wav_file.exists():
                    wav_size = wav_file.stat().st_size
                    if wav_size > _AUDIO_SIZE_LIMIT:
                        print(f"  ⚠️  Skipping {wav_file.name} ({wav_size // 1024 // 1024} MB > {_AUDIO_SIZE_LIMIT // 1024 // 1024} MB limit)")
                    else:
                        key = f"sample_{sample_idx}/{audio_rank}_{kind}_audio"
                        log_dict[key] = wandb.Audio(str(wav_file), caption=f"sample {sample_idx} — {kind}")

                if piano_rolls_dir.exists():
                    png_file = piano_rolls_dir / f"sample_{sample_idx}_{kind}.png"
                    if png_file.exists():
                        key = f"sample_{sample_idx}/{roll_rank}_{kind}_piano_roll"
                        log_dict[key] = wandb.Image(str(png_file), caption=f"sample {sample_idx} — {kind}")

        if log_dict:
            wandb.log(log_dict)

        print(f"✅ Uploaded to wandb: {run.url}")
        wandb.finish()
        return

    # Original baseline logic for gpt2/llama
    # Gather config from both models' metadata
    config = {"run_id": run_dir.name}
    for model_name in ["gpt2", "llama"]:
        meta_path = run_dir / model_name / "midi" / "inference_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                config[model_name] = json.load(f)

    tok_label = _tokenizer_label(config)
    models = _present_models(config)
    if len(models) == 1:
        model_label = models[0]
        ckpt_label = _checkpoint_label(config[models[0]].get("checkpoint", ""))
    else:
        # Fallback for old combined runs
        model_label = "-".join(models) if models else "unknown"
        ckpt_label = "_".join(
            _checkpoint_label(config.get(m, {}).get("checkpoint", "")) for m in models
        )
    timestamp_match = re.search(r'(\d{8}_\d{6})', run_dir.name)
    timestamp = timestamp_match.group(1) if timestamp_match else run_dir.name
    run_name = f"{tok_label}_{model_label}_{ckpt_label}_{timestamp}"

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=run_name,
        config=config,
        tags=["inference", "symbolic", "baseline"],
    )

    # (file_kind, display_name, audio_rank, roll_rank) — rank prefixes drive alphabetical
    # sort in wandb: prompt audio → prompt roll → ground truth audio → ground truth roll
    # → generated audio → generated roll
    UPLOAD_KINDS = [
        ("prompt",                             "prompt",       "01", "02"),
        ("ground_truth",                       "ground_truth", "03", "04"),
        ("prompt_with_generated_continuation", "generated",    "05", "06"),
    ]

    log_dict = {}

    for model_name in ["gpt2", "llama"]:
        model_dir = run_dir / model_name
        piano_rolls_dir = model_dir / "piano_rolls"
        wav_dir = model_dir / "wav"

        if not piano_rolls_dir.exists():
            print(f"⚠️  No piano rolls found for {model_name}, skipping")
            continue

        sample_indices = sorted(set(
            f.stem.split("_")[1]
            for f in piano_rolls_dir.glob("sample_*_generated.png")
        ), key=int)

        for idx in sample_indices:
            for file_kind, display_name, audio_rank, roll_rank in UPLOAD_KINDS:
                wav = wav_dir / f"sample_{idx}_{file_kind}.wav"
                if wav.exists():
                    key = f"sample_{idx}/{audio_rank}_{display_name}_audio"
                    log_dict[key] = wandb.Audio(str(wav), caption=f"sample {idx} — {display_name}")

                png = piano_rolls_dir / f"sample_{idx}_{file_kind}.png"
                if png.exists():
                    key = f"sample_{idx}/{roll_rank}_{display_name}_piano_roll"
                    log_dict[key] = wandb.Image(str(png), caption=f"sample {idx} — {display_name}")

    if log_file and log_file.exists():
        wandb.save(str(log_file), base_path=str(log_file.parent.parent))

    wandb.log(log_dict)
    wandb.finish()
    print(f"✅ Uploaded to wandb: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload inference run to wandb")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--wandb_project", default="music_inference_baselines")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--log_file", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"❌ Run directory not found: {args.run_dir}")
        sys.exit(1)

    upload_run(args.run_dir, args.wandb_project, args.wandb_entity, args.log_file, args.temperature)
