"""
Plot training curves directly from WandB API to verify what WandB shows in the browser.

Usage:
    conda activate music_EBT
    python tools/plot_training_curves.py

Saves plots to tools/training_curves/
"""

import os
import sys
from pathlib import Path

import wandb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

ENTITY  = "rceccatelli-eth-z-rich"
PROJECT = "mus_symb_ebt_pretrain"
OUT_DIR = Path(__file__).parent / "training_curves"

# Groups of runs to stitch together as a single continuous training curve.
# Each list is ordered from first to last job (by global_step).
RUN_GROUPS = {
    "EBT small — REMI": [
        "snqhp0us",   # job14747549
    ],
    "EBT small — Anticipation-AT": [
        "dzs5ot9c",   # job15871250
    ],
    "EBT medium — Anticipation-AT": [
        "p6r3w636",   # job15641636  (steps 0 – ~3600)
        "rg3ba28e",   # job15716080  (steps ~3600 – ~7200)
        "pm3vx9sm",   # job15875407  (steps ~7200 – ~10500)
    ],
}

METRICS = ["valid_loss", "train_loss", "valid_perplexity",
           "valid_final_loss", "valid_initial_loss", "valid_energy_improvement"]
# valid and train metrics are logged in separate WandB rows; fetch each group
# separately using scan_history key-filter (returns only rows that have those keys).
VALID_METRICS = ["valid_loss", "valid_perplexity", "valid_final_loss",
                 "valid_initial_loss", "valid_energy_improvement"]
TRAIN_METRICS = ["train_loss", "train_perplexity", "train_final_loss",
                 "train_initial_loss", "train_energy_improvement"]
# Use Lightning's global_step as x-axis.  WandB's internal _step is inflated by
# wandb_watch gradient/activation logging and is not a reliable training step proxy.
STEP_COL = "trainer/global_step"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan(run, keys: list[str]) -> pd.DataFrame:
    """Return rows that contain all of `keys`, as a DataFrame."""
    rows = list(run.scan_history(keys=keys))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=[STEP_COL])
    df[STEP_COL] = df[STEP_COL].astype(int)
    return df


def fetch_history(run_id: str, api: wandb.Api) -> pd.DataFrame:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    print(f"  Fetching {run.name} ({run_id})…", flush=True)

    frames = []
    for metric in VALID_METRICS + TRAIN_METRICS:
        df = _scan(run, [metric, STEP_COL])
        if not df.empty:
            frames.append(df[[STEP_COL, metric]])

    if not frames:
        print("    → (empty)")
        return pd.DataFrame()

    # Merge on STEP_COL so each unique step gets all its metric columns
    from functools import reduce
    merged = reduce(lambda a, b: pd.merge(a, b, on=STEP_COL, how="outer"), frames)
    merged = merged.sort_values(STEP_COL).reset_index(drop=True)
    step_min, step_max = merged[STEP_COL].min(), merged[STEP_COL].max()
    n_valid = merged["valid_loss"].notna().sum() if "valid_loss" in merged.columns else 0
    n_train = merged["train_loss"].notna().sum() if "train_loss" in merged.columns else 0
    print(f"    → {len(merged)} rows, step {step_min}–{step_max} "
          f"(valid_loss: {n_valid} pts, train_loss: {n_train} pts)")
    return merged


def stitch_runs(run_ids: list[str], api: wandb.Api) -> pd.DataFrame:
    """Fetch all runs in a group and concatenate, removing duplicate steps."""
    frames = [fetch_history(rid, api) for rid in run_ids]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(STEP_COL).drop_duplicates(subset=STEP_COL, keep="first")
    return df.reset_index(drop=True)


def plot_group(name: str, df: pd.DataFrame, ax_loss, ax_ppl):
    if "valid_loss" in df.columns:
        valid = df[[STEP_COL, "valid_loss"]].dropna()
        ax_loss.plot(valid[STEP_COL], valid["valid_loss"],
                     label=f"{name} — valid", linewidth=1.5)
        print(f"  valid_loss: {len(valid)} data points")

    if "train_loss" in df.columns:
        train = df[[STEP_COL, "train_loss"]].dropna()
        ax_loss.plot(train[STEP_COL], train["train_loss"],
                     label=f"{name} — train", linewidth=0.8, alpha=0.5, linestyle="--")
        print(f"  train_loss: {len(train)} data points")

    if "valid_perplexity" in df.columns:
        ppl = df[[STEP_COL, "valid_perplexity"]].dropna()
        ppl = ppl[ppl["valid_perplexity"] < 1e6]
        ax_ppl.plot(ppl[STEP_COL], ppl["valid_perplexity"],
                    label=name, linewidth=1.5)
        print(f"  valid_perplexity: {len(ppl)} data points")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    fig_ppl,  ax_ppl  = plt.subplots(figsize=(10, 5))

    for name, run_ids in RUN_GROUPS.items():
        print(f"\n{name}")
        df = stitch_runs(run_ids, api)

        # ── Per-group plot ─────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(name)

        valid = df[[STEP_COL, "valid_loss"]].dropna() if "valid_loss" in df.columns else None
        train = df[[STEP_COL, "train_loss"]].dropna() if "train_loss" in df.columns else None

        if valid is not None and not valid.empty:
            axes[0].plot(valid[STEP_COL], valid["valid_loss"],
                         label="valid_loss", linewidth=1.5)
            axes[0].scatter(valid[STEP_COL], valid["valid_loss"],
                            s=8, zorder=5)
        if train is not None and not train.empty:
            axes[0].plot(train[STEP_COL], train["train_loss"],
                         label="train_loss", linewidth=0.8, alpha=0.6, linestyle="--")
        axes[0].set_xlabel("trainer/global_step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        if "valid_perplexity" in df.columns:
            ppl = df[[STEP_COL, "valid_perplexity"]].dropna()
            ppl = ppl[ppl["valid_perplexity"] < 1e6]
            axes[1].plot(ppl[STEP_COL], ppl["valid_perplexity"],
                         label="valid_perplexity", linewidth=1.5)
            axes[1].scatter(ppl[STEP_COL], ppl["valid_perplexity"], s=8, zorder=5)
        axes[1].set_xlabel("Global step")
        axes[1].set_ylabel("Perplexity")
        axes[1].set_title("Valid perplexity")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        slug = name.replace(" ", "_").replace("—", "-").replace("/", "-")
        out = OUT_DIR / f"{slug}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved → {out}")

        # ── Accumulate into combined plot ──────────────────────────────────
        plot_group(name, df, ax_loss, ax_ppl)

    # ── Combined plot ──────────────────────────────────────────────────────
    for ax, title, ylabel in [
        (ax_loss, "Valid + Train Loss — all models", "Loss"),
        (ax_ppl,  "Valid Perplexity — all models",   "Perplexity"),
    ]:
        ax.set_xlabel("trainer/global_step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig_loss.tight_layout()
    fig_ppl.tight_layout()
    fig_loss.savefig(OUT_DIR / "combined_loss.png", dpi=150)
    fig_ppl.savefig(OUT_DIR / "combined_perplexity.png", dpi=150)
    plt.close("all")
    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
