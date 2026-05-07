#!/usr/bin/env python3
"""Upload inference run outputs (piano rolls, audio, log) to wandb."""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def upload_run(run_dir: Path, wandb_project: str, wandb_entity: str, log_file: Path = None):
    import wandb

    # Gather config from both models' metadata
    config = {"run_id": run_dir.name}
    for model_name in ["gpt2", "llama"]:
        meta_path = run_dir / model_name / "midi" / "inference_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                config[model_name] = json.load(f)

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=f"baselines_inference_{run_dir.name}",
        config=config,
        tags=["inference", "symbolic", "baseline"],
    )

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
            for kind in ["prompt", "generated", "ground_truth", "prompt_with_generated_continuation"]:
                png = piano_rolls_dir / f"sample_{idx}_{kind}.png"
                if png.exists():
                    key = f"{model_name}/sample_{idx}/{kind}_piano_roll"
                    log_dict[key] = wandb.Image(str(png), caption=f"{model_name} sample {idx} — {kind}")

                wav = wav_dir / f"sample_{idx}_{kind}.wav"
                if wav.exists():
                    key = f"{model_name}/sample_{idx}/{kind}_audio"
                    log_dict[key] = wandb.Audio(str(wav), caption=f"{model_name} sample {idx} — {kind}")

    if log_file and log_file.exists():
        wandb.save(str(log_file), base_path=str(log_file.parent))

    wandb.log(log_dict)
    wandb.finish()
    print(f"✅ Uploaded to wandb: {run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload inference run to wandb")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--wandb_project", default="music_inference_baselines")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--log_file", type=Path, default=None)
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"❌ Run directory not found: {args.run_dir}")
        sys.exit(1)

    upload_run(args.run_dir, args.wandb_project, args.wandb_entity, args.log_file)
