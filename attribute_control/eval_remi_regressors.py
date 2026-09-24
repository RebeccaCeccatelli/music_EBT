"""One-off: validate the REMI velocity/duration/pitch_register regressors
currently active in the demo (same checkpoints _find_attribute_regressor
would auto-select) by comparing true vs. predicted attribute values on real
held-out validation windows — same method as
eval_pitch_register_regressor.py, for the REMI side."""
import sys
import random
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.mus.infer_ebt import load_dataset
from attribute_control.attributes import ATTRIBUTES
from attribute_control.note_density import NoteDensityRegressor

N_WINDOWS = 300
SEED = 0
REMI_TOKENIZER_CONFIG = "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json"

CHECKPOINTS = {
    "velocity": "/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/attr_control/velocity_regressor_remi_20260827_131302_21395769/best.pt",
    "duration": "/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/attr_control/duration_regressor_remi_20260827_073950_21383289/best.pt",
    "pitch_register": "/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/attr_control/pitch_register_regressor_remi_20260828_185907_21506161/best.pt",
}


def extract_embedding_weight(ckpt_path: str) -> torch.Tensor:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["model.embeddings.weight"].float()


def evaluate(attr_name: str, ckpt_path: str, ds):
    reg_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hard_window = reg_ckpt["density_hard_window"]
    emb_weight = extract_embedding_weight(reg_ckpt["ebt_checkpoint"])

    regressor = NoteDensityRegressor(emb_dim=reg_ckpt["emb_dim"], hidden_dim=reg_ckpt["hidden_dim"])
    regressor.load_state_dict(reg_ckpt["model_state"])
    regressor.eval()

    random.seed(SEED)
    n_needed = hard_window + 1
    indices = random.sample(range(len(ds)), N_WINDOWS)

    trues, preds = [], []
    with torch.no_grad():
        for idx in indices:
            tokens = ds.get_full_tokens(idx)
            N = len(tokens)
            if N < n_needed:
                continue
            start = random.randint(0, N - n_needed)
            window = tokens[start:start + n_needed]
            hard_tokens, future_tokens = window[:-1], window[-1:]

            label = ATTRIBUTES[attr_name](window, "REMI")
            hard_idx = torch.tensor(hard_tokens, dtype=torch.long)
            future_idx = torch.tensor(future_tokens, dtype=torch.long)
            hard_emb_sum = emb_weight[hard_idx].sum(0)
            fut_emb_sum = emb_weight[future_idx].sum(0)
            e_mean = (hard_emb_sum + fut_emb_sum) / (len(hard_tokens) + len(future_tokens))
            pred = regressor(e_mean.unsqueeze(0)).item()
            trues.append(label)
            preds.append(pred)

    trues_t = torch.tensor(trues)
    preds_t = torch.tensor(preds)
    mae = (trues_t - preds_t).abs().mean().item()
    corr = torch.corrcoef(torch.stack([trues_t, preds_t]))[0, 1].item()

    print(f"\n== {attr_name} ==")
    print(f"  n={len(trues)}  hard_window={hard_window}")
    print(f"  TRUE  range: {min(trues):.3f} - {max(trues):.3f}  (mean {sum(trues)/len(trues):.3f})")
    print(f"  PRED  range: {min(preds):.3f} - {max(preds):.3f}  (mean {sum(preds)/len(preds):.3f})")
    print(f"  MAE: {mae:.4f}   Pearson r: {corr:.3f}")


if __name__ == "__main__":
    hparams = Namespace(
        tokenizer_type="REMI",
        tokenizer_config_path=REMI_TOKENIZER_CONFIG,
        dataset_name="giga_midi",
        context_length=512,
        validation_split_pct=0.05,
    )
    ds = load_dataset(hparams, split="validation")
    for attr_name, ckpt_path in CHECKPOINTS.items():
        evaluate(attr_name, ckpt_path, ds)
