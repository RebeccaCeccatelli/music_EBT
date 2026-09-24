"""
Compute each tokenizer's own oracle-unigram entropy — the cross-entropy a
model achieves by knowing nothing beyond each token's real, empirical
marginal frequency — as a fair baseline for comparing validation loss ACROSS
tokenizers with different vocabulary sizes.

Why this exists: comparing raw valid_loss between REMI (vocab_size=427) and
Anticipation-Arrival-Time (vocab_size=55028) is misleading on its own — a
~129x larger vocabulary has a much higher ceiling of possible uncertainty to
reduce, so a higher raw loss doesn't necessarily mean a worse model. This
script measures each tokenizer's own real (skewed, not uniform) token
distribution and reports how much of ITS OWN entropy a given valid_loss
eliminates — see docs/thesis_findings/2026-09-22_anticipation_loss_vocab_normalization.md
for the first use of this and its result.

Samples by TOKEN BUDGET, not song/sequence count — REMI's dataset unit is a
full, variable-length song (~15.8k tokens on average); Anticipation's is a
fixed 1024-token window. Equal counts of "songs" silently gave REMI ~15x more
raw tokens than Anticipation in an earlier pass, undersampling Anticipation's
long, sparse tail and biasing its entropy estimate down — sampling to an equal
TOKEN total avoids that.

Usage:
    python attribute_control/compute_unigram_entropy.py \\
        --tokenizer_type REMI --valid_loss 0.7551
    python attribute_control/compute_unigram_entropy.py \\
        --tokenizer_type Anticipation-Arrival-Time --valid_loss 1.0864
"""

import sys
import math
import random
import argparse
from pathlib import Path
from argparse import Namespace
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.mus.infer_ebt import load_dataset

REMI_TOKENIZER_CONFIG = "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json"
VOCAB_SIZES = {"REMI": 427, "Anticipation-Arrival-Time": 55028}


def unigram_entropy(tokenizer_type: str, token_budget: int, seed: int = 0, split: str = "validation") -> dict:
    hparams = Namespace(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=REMI_TOKENIZER_CONFIG if tokenizer_type == "REMI" else None,
        dataset_name="giga_midi",
        context_length=512,
        validation_split_pct=0.05,
    )
    dataset = load_dataset(hparams, split=split)
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)

    counts = Counter()
    total = 0
    n_used = 0
    for i in order:
        toks = (dataset.get_full_tokens(i) if hasattr(dataset, "get_full_tokens")
                else dataset[i]["input_ids"].tolist())
        counts.update(toks)
        total += len(toks)
        n_used += 1
        if total >= token_budget:
            break

    H = -sum((c / total) * math.log(c / total) for c in counts.values())
    vocab_size = VOCAB_SIZES.get(tokenizer_type)

    return {
        "tokenizer_type": tokenizer_type,
        "n_sequences_sampled": n_used,
        "n_tokens": total,
        "n_distinct_tokens_seen": len(counts),
        "vocab_size": vocab_size,
        "unigram_entropy_nats": H,
        "uniform_entropy_nats": math.log(vocab_size) if vocab_size else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer_type", required=True, choices=list(VOCAB_SIZES))
    p.add_argument("--token_budget", type=int, default=30_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="validation")
    p.add_argument("--valid_loss", type=float, default=None,
                    help="If given, also prints the entropy-reduction fraction "
                         "(1 - valid_loss / unigram_entropy) this loss represents.")
    args = p.parse_args()

    res = unigram_entropy(args.tokenizer_type, args.token_budget, args.seed, args.split)
    print(res)

    if args.valid_loss is not None:
        H_uni = res["unigram_entropy_nats"]
        H_max = res["uniform_entropy_nats"]
        frac_uniform = 1 - args.valid_loss / H_max
        frac_unigram = 1 - args.valid_loss / H_uni
        print(f"\nvalid_loss={args.valid_loss:.4f}"
              f"  vs. uniform baseline ({H_max:.4f} nats): {frac_uniform:.4%} reduction"
              f"  vs. oracle-unigram baseline ({H_uni:.4f} nats): {frac_unigram:.4%} reduction")


if __name__ == "__main__":
    main()
