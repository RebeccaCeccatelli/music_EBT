"""Thesis figure: corpus distributions of the guided attributes.

Reads the per-window attribute values written by
attribute_control/compute_corpus_histograms.py (2,000 training windows of 256
tokens, one per song, seed 0) for REMI and Anticipation, and plots one panel
per attribute. Windows without pitched notes (pitch register = 0, the
function's fallback value) are dropped from the pitch-register panel. The
bracket in each panel is the scale sigma_a used to set guidance targets
(attribute_control/corpus_stats.json), for comparison with the spread.

Usage:
    python eval/plot_corpus_attributes.py --out eth-mit-master-thesis/figures/corpus_attributes.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "attribute_control"
COLORS = {"REMI": "#2a78d6", "Anticipation": "#eb6834"}  # categorical slots 1, 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

PANELS = [
    ("velocity", "Velocity", (0.0, 1.0)),
    ("duration", "Duration", (0.0, 0.30)),
    ("pitch_register", "Pitch register", (0.15, 0.80)),
]


def load():
    remi = json.load(open(ROOT / "corpus_histograms.json"))["values"]
    ant = json.load(open(ROOT / "corpus_histograms_anticipation.json"))["values"]
    sigma = {a: v["stdev"] for a, v in json.load(open(ROOT / "corpus_stats.json")).items()}
    return {"REMI": remi, "Anticipation": ant}, sigma


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data, sigma = load()

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.9), constrained_layout=True)
    for ax, (attr, title, (lo, hi)) in zip(axes, PANELS):
        bins = np.linspace(lo, hi, 41)
        ymax = 0.0
        for tok, vals in data.items():
            if attr not in vals:
                continue
            v = np.asarray(vals[attr], dtype=float)
            if attr == "pitch_register":
                v = v[v > 0]
            h, _ = np.histogram(np.clip(v, lo, hi), bins=bins)
            h = h / len(v)
            ymax = max(ymax, h.max())
            ax.stairs(h, bins, color=COLORS[tok], linewidth=1.2, label=tok)
        # sigma_a bracket, anchored at the REMI mean
        m = float(np.mean([x for x in data["REMI"][attr] if attr != "pitch_register" or x > 0]))
        y = ymax * 1.12
        ax.plot([m, m + sigma[attr]], [y, y], color=INK, linewidth=1.0)
        for x in (m, m + sigma[attr]):
            ax.plot([x, x], [y - ymax * 0.04, y + ymax * 0.04], color=INK, linewidth=1.0)
        ax.text(m + sigma[attr] / 2, y + ymax * 0.06, r"$\sigma_a$", ha="center", va="bottom", color=INK)
        ax.set_ylim(0, ymax * 1.35)
        ax.set_xlim(lo, hi)
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xlabel("attribute value")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("fraction of windows")
    axes[1].legend(frameon=False, loc="center right", fontsize=7.5)
    fig.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
