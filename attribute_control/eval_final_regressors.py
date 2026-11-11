"""Held-out accuracy of the regressors used in the thesis (MAE, Pearson r).

For each regressor checkpoint (best.pt), samples validation windows of the
same shape it was trained on (H committed tokens + the true next token; for
Anticipation the control-free triplet stream, and note tokens only when the
regressor was trained with note_only_window), computes the exact attribute
value of the window and the regressor's prediction from the frozen embedding
table of the model it was trained against, and reports MAE and Pearson r.
The next token is given as a clean one-hot embedding (no simulated soft-token
noise), so this measures how well the regressor reads real music.

Generalises eval_remi_regressors.py / eval_pitch_register_regressor.py.

Windows without pitched notes are skipped for pitch_register.

Usage: python attribute_control/eval_final_regressors.py OUT.json best.pt [best.pt ...]
"""
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.mus.infer_ebt import load_dataset
from attribute_control.attributes import ATTRIBUTES, _anticipation_triplets
from attribute_control.note_density import NoteDensityRegressor

N_WINDOWS = 1000
# ALIGN=1: start Anticipation windows on a triplet boundary. Training samples
# raw windows at any offset, so its labels are misaligned for 2 of 3 offsets.
import os
ALIGN = os.environ.get("ALIGN", "0") == "1"
SEED = 0
REMI_TOKENIZER_CONFIG = "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json"

_datasets = {}
_embeddings = {}


def get_dataset(tokenizer_type):
    if tokenizer_type not in _datasets:
        hparams = Namespace(
            tokenizer_type=tokenizer_type,
            tokenizer_config_path=REMI_TOKENIZER_CONFIG if tokenizer_type == "REMI" else None,
            dataset_name="giga_midi",
            context_length=512,
            validation_split_pct=0.05,
        )
        _datasets[tokenizer_type] = load_dataset(hparams, split="validation")
    return _datasets[tokenizer_type]


def get_embedding(ckpt_path):
    if ckpt_path not in _embeddings:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _embeddings[ckpt_path] = ckpt["state_dict"]["model.embeddings.weight"].float()
        del ckpt
    return _embeddings[ckpt_path]


def sample_window(tokens, tokenizer_type, n_needed, note_only_window, rng):
    if tokenizer_type != "REMI":
        tokens = _anticipation_triplets(tokens)
    N = len(tokens)
    if note_only_window:
        raw = n_needed * 3
        if N < raw:
            return None
        max_start = N - raw
        start = rng.randint(0, max_start)
        start -= start % 3
        window = tokens[start:start + raw]
        notes = window[2::3]
        return window, notes[:-1], notes[-1:]
    if N < n_needed:
        return None
    start = rng.randint(0, N - n_needed)
    if ALIGN and tokenizer_type != "REMI":
        start -= start % 3  # start on a triplet boundary (training does not)
    window = tokens[start:start + n_needed]
    return window, window[:-1], window[-1:]


def evaluate(path):
    reg = torch.load(path, map_location="cpu", weights_only=False)
    tok_type = reg["tokenizer_type"]
    attr = reg["attribute"]
    H = reg["density_hard_window"]
    note_only = reg.get("note_only_window", False)
    emb = get_embedding(reg["ebt_checkpoint"])

    model = NoteDensityRegressor(emb_dim=reg["emb_dim"], hidden_dim=reg["hidden_dim"])
    model.load_state_dict(reg["model_state"])
    model.eval()

    ds = get_dataset(tok_type)
    rng = random.Random(SEED)
    indices = rng.sample(range(len(ds)), min(4 * N_WINDOWS, len(ds)))
    trues, preds = [], []
    skipped = 0
    with torch.no_grad():
        for idx in indices:
            if len(trues) >= N_WINDOWS:
                break
            s = sample_window(ds.get_full_tokens(idx), tok_type, H + 1, note_only, rng)
            if s is None:
                continue
            window, hard, fut = s
            label = ATTRIBUTES[attr](window, tok_type)
            # pitch_register returns 0.0 when the window has no pitched notes;
            # such windows have no register and are skipped, as in the corpus stats.
            if attr == "pitch_register" and label == 0.0:
                skipped += 1
                continue
            e = (emb[torch.tensor(hard)].sum(0) + emb[torch.tensor(fut)].sum(0)) / (len(hard) + len(fut))
            trues.append(float(label))
            preds.append(model(e.unsqueeze(0)).item())

    t, p = torch.tensor(trues), torch.tensor(preds)
    res = {
        "regressor": str(path),
        "attribute": attr,
        "tokenizer": tok_type,
        "model_checkpoint": reg["ebt_checkpoint"],
        "hard_window": H,
        "note_only_window": note_only,
        "aligned_windows": ALIGN,
        "n": len(trues),
        "skipped_no_pitch": skipped,
        "mae": (t - p).abs().mean().item(),
        "pearson_r": torch.corrcoef(torch.stack([t, p]))[0, 1].item(),
        "true_mean": t.mean().item(), "true_sd": t.std().item(),
        "pred_mean": p.mean().item(), "pred_sd": p.std().item(),
    }
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    out = sys.argv[1]
    results = [evaluate(p) for p in sys.argv[2:]]
    Path(out).write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")
