"""
Decode random real GigaMIDI windows to MIDI: the reference distribution that
eval/music_quality.py compares generated samples against.

Windows have the same token length as the generated continuations being
evaluated (default 256, the length the guidance sweeps generate), so that
length-dependent metrics like pitch range and bar self-similarity are
comparable.

Usage:
    python eval/dump_reference_midi.py --tokenizer_type REMI --n 500 --out_dir <dir>
    python eval/dump_reference_midi.py --tokenizer_type Anticipation-Arrival-Time --n 500 --out_dir <dir>
"""

import argparse
import random
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_control.train_density_regressor import load_music_dataset
from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer
from inference.mus.tokens_to_midi import tokens_to_midi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer_type', required=True)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--n_tokens', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hp = Namespace(context_length=512, dataset_name='giga-midi', data_dir=None,
                   tokenizer_type=args.tokenizer_type)
    ds = load_music_dataset(args.tokenizer_type, args.split, hp)
    is_ant = args.tokenizer_type.startswith('Anticipation')
    # REMI must use the dataset's own tokenizer.json (vocab 427). Without it,
    # load_tokenizer() silently builds a default REMI (vocab 284) and every
    # decode fails with KeyError. The model checkpoints store this same path.
    config = None if is_ant else ds.TOKENIZER_CONFIG_PATH
    tokenizer = load_tokenizer(args.tokenizer_type, tokenizer_config_path=config)[0]

    written = tries = 0
    while written < args.n and tries < 5 * args.n:
        tries += 1
        tokens = ds.get_full_tokens(random.randrange(len(ds)))
        if is_ant:
            # Keep the mode token (decode() strips it) and start on a triplet boundary.
            body = tokens[1:]
            if len(body) < args.n_tokens:
                continue
            start = random.randrange(0, len(body) - args.n_tokens + 1, 3)
            window = tokens[:1] + body[start:start + args.n_tokens]
        else:
            if len(tokens) < args.n_tokens:
                continue
            start = random.randrange(len(tokens) - args.n_tokens + 1)
            window = tokens[start:start + args.n_tokens]
        try:
            midi = tokens_to_midi(window, tokenizer)
        except Exception as e:
            print(f"  decode failed ({type(e).__name__}: {e}); drawing another window")
            continue
        (out / f"ref_{written:04d}.mid").write_bytes(midi)
        written += 1
    print(f"Wrote {written} reference windows ({args.n_tokens} tokens each) → {out}")


if __name__ == '__main__':
    main()
