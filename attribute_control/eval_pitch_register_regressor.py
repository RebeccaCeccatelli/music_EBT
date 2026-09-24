"""One-off: before/after check of the note_only_window pitch_register
regressor retrain — samples real Anticipation validation windows, computes
the TRUE pitch_register value and the regressor's PREDICTED value for each,
and reports the predicted range/correlation. The original bug this retrain
targeted: true values spanning 0.24-0.76, predicted values compressed into
only 0.15-0.19."""
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.mus.infer_ebt import load_dataset
from attribute_control.attributes import ATTRIBUTES
from attribute_control.note_density import NoteDensityRegressor

N_WINDOWS = 300
SEED = 0

CHECKPOINTS = {
    "OLD (broken, no note_only_window)":
        "/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/attr_control/"
        "pitch_register_regressor_ant-at-full_20260922_125323_23488200/best.pt",
    "NEW (retrained, note_only_window)":
        "/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/attr_control/"
        "pitch_register_regressor_ant-at-full_20260924_061847_23640180/best.pt",
}


def extract_embedding_weight(ckpt_path: str) -> torch.Tensor:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["model.embeddings.weight"].float()


def sample_windows(tokenizer_type: str, n: int, hard_window: int, note_only_window: bool):
    import random
    random.seed(SEED)
    hparams = Namespace(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=None,
        dataset_name="giga_midi",
        context_length=512,
        validation_split_pct=0.05,
    )
    ds = load_dataset(hparams, split="validation")
    n_available = len(ds)
    indices = random.sample(range(n_available), min(n, n_available))
    n_needed = hard_window + 1

    windows = []
    for idx in indices:
        tokens = ds.get_full_tokens(idx)
        N = len(tokens)
        if note_only_window:
            raw_window_size = n_needed * 3
            if N < raw_window_size:
                raw_start, raw_window_size = 0, N - (N % 3)
            else:
                max_start = N - raw_window_size
                raw_start = random.randint(0, max_start - (max_start % 3)) if max_start > 0 else 0
                raw_start -= raw_start % 3
            window = tokens[raw_start:raw_start + raw_window_size]
            note_tokens = window[2::3]
            if len(note_tokens) < 2:
                continue
            hard_tokens, future_tokens = note_tokens[:-1], note_tokens[-1:]
        elif N < n_needed:
            continue
        else:
            start = random.randint(0, N - n_needed)
            window = tokens[start:start + n_needed]
            hard_tokens, future_tokens = window[:-1], window[-1:]
        windows.append((window, hard_tokens, future_tokens))
    return windows


def evaluate(name: str, ckpt_path: str):
    reg_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tokenizer_type = reg_ckpt["tokenizer_type"]
    hard_window = reg_ckpt["density_hard_window"]
    note_only_window = reg_ckpt.get("note_only_window", False)
    emb_weight = extract_embedding_weight(reg_ckpt["ebt_checkpoint"])

    regressor = NoteDensityRegressor(emb_dim=reg_ckpt["emb_dim"], hidden_dim=reg_ckpt["hidden_dim"])
    regressor.load_state_dict(reg_ckpt["model_state"])
    regressor.eval()

    windows = sample_windows(tokenizer_type, N_WINDOWS, hard_window, note_only_window)

    trues, preds = [], []
    with torch.no_grad():
        for window, hard_tokens, future_tokens in windows:
            label = ATTRIBUTES["pitch_register"](window, tokenizer_type)
            hard_idx = torch.tensor(hard_tokens, dtype=torch.long)
            future_idx = torch.tensor(future_tokens, dtype=torch.long)
            hard_emb_sum = emb_weight[hard_idx].sum(0)
            fut_emb_sum = emb_weight[future_idx].sum(0)  # clean (no simulated MCMC noise)
            e_mean = (hard_emb_sum + fut_emb_sum) / (len(hard_tokens) + len(future_tokens))
            pred = regressor(e_mean.unsqueeze(0)).item()
            trues.append(label)
            preds.append(pred)

    trues_t = torch.tensor(trues)
    preds_t = torch.tensor(preds)
    mae = (trues_t - preds_t).abs().mean().item()
    corr = torch.corrcoef(torch.stack([trues_t, preds_t]))[0, 1].item()

    print(f"\n== {name} ==")
    print(f"  n={len(trues)}  note_only_window={note_only_window}  hard_window={hard_window}")
    print(f"  TRUE  range: {min(trues):.3f} - {max(trues):.3f}  (mean {sum(trues)/len(trues):.3f})")
    print(f"  PRED  range: {min(preds):.3f} - {max(preds):.3f}  (mean {sum(preds)/len(preds):.3f})")
    print(f"  MAE: {mae:.4f}   Pearson r: {corr:.3f}")


if __name__ == "__main__":
    for name, path in CHECKPOINTS.items():
        evaluate(name, path)
