"""One-off: sample real windows from the training split (REMI or Anticipation)
and compute each attribute's real value on every window, saving the raw
per-window values (not just mean/stdev) so they can be plotted as real
histograms instead of an assumed-normal curve.

Usage:
    python attribute_control/compute_corpus_histograms.py --tokenizer REMI
    python attribute_control/compute_corpus_histograms.py --tokenizer Anticipation-Arrival-Time
"""
import argparse
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.mus.infer_ebt import load_dataset
from attribute_control.attributes import ATTRIBUTES, _anticipation_triplets

WINDOW = 256
N_SONGS = 2000
SEED = 0

REMI_TOKENIZER_CONFIG = "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json"

ATTRS_BY_TOKENIZER = {
    "REMI": ["velocity", "duration", "pitch_register"],
    # Anticipation's vocabulary has no velocity field at all (see
    # compute_velocity's docstring) — only duration and pitch_register apply.
    "Anticipation-Arrival-Time": ["duration", "pitch_register"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="REMI", choices=list(ATTRS_BY_TOKENIZER.keys()))
    args = p.parse_args()
    tokenizer_type = args.tokenizer
    attrs = ATTRS_BY_TOKENIZER[tokenizer_type]
    is_anticipation = tokenizer_type.startswith("Anticipation")

    random.seed(SEED)
    hparams = Namespace(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=REMI_TOKENIZER_CONFIG if tokenizer_type == "REMI" else None,
        dataset_name="giga_midi",
        context_length=512,
        validation_split_pct=0.05,
    )
    ds = load_dataset(hparams, split="train")
    n_available = len(ds)
    indices = random.sample(range(n_available), min(N_SONGS, n_available))

    values = {a: [] for a in attrs}
    n_skipped_short = 0
    for i, idx in enumerate(indices):
        tokens = ds.get_full_tokens(idx)
        if is_anticipation:
            # Raw stored sequences have a leading AUTOREGRESS/ANTICIPATE mode
            # marker and (for ANTICIPATE-mode sequences) real events
            # interleaved with anticipated-control triplets — a naive
            # contiguous slice isn't reliably triplet-aligned. Same cleanup
            # as today's train_density_regressor.py fix, applied here too so
            # this histogram doesn't reproduce that exact bug.
            tokens = _anticipation_triplets(tokens)
        N = len(tokens)
        if N < WINDOW:
            if N < 6:
                n_skipped_short += 1
                continue
            window = tokens
        else:
            start = random.randint(0, N - WINDOW)
            if is_anticipation:
                start -= start % 3
            window = tokens[start:start + WINDOW]
        for attr in attrs:
            try:
                values[attr].append(float(ATTRIBUTES[attr](window, tokenizer_type)))
            except Exception:
                pass
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(indices)}")

    slug = "remi" if tokenizer_type == "REMI" else "anticipation"
    out_path = Path(__file__).parent / f"corpus_histograms_{slug}.json"
    with open(out_path, "w") as f:
        json.dump({"window": WINDOW, "n_songs_sampled": len(indices),
                    "n_skipped_short": n_skipped_short, "values": values}, f)
    print(f"Wrote {out_path}")
    for attr, vals in values.items():
        print(attr, "n=", len(vals))


if __name__ == "__main__":
    main()
