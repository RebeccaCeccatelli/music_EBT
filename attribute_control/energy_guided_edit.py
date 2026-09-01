"""
R3-guided post-hoc note removal: an extension to Du et al. 2023's classifier
guidance for the direction it structurally can't handle at generation time.

R3's gradient guidance only ever nudges the NEXT token choice — it can never
undo a token already committed. That makes "add more of attribute X" a good
fit (nudge upcoming note-placement decisions) but "remove X" a poor one: there
is no future decision that un-places an already-generated note. Empirically
(see attribute_control/listen_density_sweep.py runs), pushing harder on the
negative direction via lambda alone doesn't fix this — it mostly just buys
coherence collapse for little extra control (see e.g. prompt 9938 in the
2026-08-29 drum_density mixed-prompt sweep: raising lambda 0.02->0.04 barely
moved achieved density but roughly doubled the late-segment coherence cost).

This module keeps the same R3 objective — minimize EBT's own energy plus
lambda times the attribute regressor's squared error to target — but applies
it as a discrete, greedy accept/reject search over candidate note removals
instead of continuous gradient descent on embeddings. At each step: enumerate
every removable (Pitch/PitchDrum, Velocity, Duration) triple (these three are
100% deterministically linked in this REMI vocab, so removal is always
splicing out a well-formed unit — no repair needed, an "empty" onset is a
normal, already-handled REMI construct, see attributes.py's compute_polyphony
docstring), score each candidate's post-removal state under the R3 objective,
and take the one that minimizes it. Repeat until within tolerance of target
or out of removable candidates.

This is NOT literbatim R3 (Du et al.'s formulation has no discrete edit move)
-- it's a direct extension motivated by autoregressive generation's specific
inability to delete. See discussion in project chat log around 2026-08-29.
"""

import sys
from typing import List, Tuple, Optional
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from attribute_control.note_density import (
    REMI_PITCH_MIN, REMI_PITCH_MAX, REMI_PITCHDRUM_MIN, REMI_PITCHDRUM_MAX,
    NoteDensityRegressor,
)
from attribute_control.attributes import (
    REMI_VELOCITY_MIN_ID, REMI_VELOCITY_MAX_ID,
    REMI_DURATION_MIN_ID, REMI_DURATION_MAX_ID,
    ATTRIBUTES,
)
from attribute_control.musicality_metrics import (
    MIN_EBT_INPUT, ebt_self_energy, score_sample, build_bigram_logprob_table,
)
from attribute_control.listen_density_sweep import load_corpus_stats, render_wav, zscore


def find_note_triples(tokens: List[int], note_type: str = "drum") -> List[int]:
    """
    Start indices of every well-formed (Pitch/PitchDrum, Velocity, Duration)
    triple in `tokens`, filtered by note_type ('drum', 'pitch', or 'any').
    Only counts triples that actually follow the grammar (Pitch/PitchDrum ->
    Velocity -> Duration, all 100% deterministic in real data) — a triple
    that doesn't match this shape is left alone rather than guessed at.
    """
    starts = []
    i = 0
    n = len(tokens)
    while i < n - 2:
        t = tokens[i]
        is_drum = REMI_PITCHDRUM_MIN <= t <= REMI_PITCHDRUM_MAX
        is_pitch = REMI_PITCH_MIN <= t <= REMI_PITCH_MAX
        wants = (
            (note_type == "drum" and is_drum)
            or (note_type == "pitch" and is_pitch)
            or (note_type == "any" and (is_drum or is_pitch))
        )
        if wants:
            v_ok = REMI_VELOCITY_MIN_ID <= tokens[i + 1] <= REMI_VELOCITY_MAX_ID
            d_ok = REMI_DURATION_MIN_ID <= tokens[i + 2] <= REMI_DURATION_MAX_ID
            if v_ok and d_ok:
                starts.append(i)
                i += 3
                continue
        i += 1
    return starts


def _batched_ebt_energy(model, windows: List[List[int]], device) -> List[float]:
    """EBT's own final-MCMC-step energy for each window, one batched forward pass."""
    if not windows:
        return []
    padded = []
    max_len = max(max(len(w), MIN_EBT_INPUT) for w in windows)
    for w in windows:
        w = list(w)
        if len(w) < max_len:
            w = w + [0] * (max_len - len(w))
        padded.append(w)
    x = torch.tensor(padded, dtype=torch.long, device=device)
    with torch.no_grad():
        _, energies = model.forward(x, start_pos=0, learning=False, return_raw_logits=True)
    # energies[-1]: (B, S) final-landscape per-position energy: mean over S per row.
    return energies[-1].mean(dim=-1).detach().cpu().tolist()


def _batched_regressor_predict(regressor, emb_weight, windows: List[List[int]], device) -> List[float]:
    if not windows:
        return []
    preds = []
    with torch.no_grad():
        for w in windows:
            wt = torch.tensor(w, dtype=torch.long, device=device)
            mean_emb = emb_weight[wt].mean(dim=0, keepdim=True)  # (1, D)
            preds.append(regressor(mean_emb).item())
    return preds


def energy_guided_remove(
    tokens: List[int],
    model,
    regressor,
    emb_weight,
    device,
    target: float,
    attribute_compute_fn,
    note_type: str = "drum",
    lam: float = 1.0,
    max_removals: int = 30,
    tolerance: float = 0.02,
    score_window: int = 64,
    verbose: bool = True,
) -> Tuple[List[int], float, List[Tuple[int, float]]]:
    """
    Greedily remove one note triple at a time — the one whose removal most
    reduces (EBT self-energy + lam * regressor squared-error-to-target),
    evaluated on a local window around the removal site — until `achieved`
    is within `tolerance` of `target` or no candidates remain.

    Returns (edited_tokens, final_achieved_value, history), history being
    [(seq_len, achieved_value), ...] after each removal, starting tokens included.
    """
    tokens = list(tokens)
    achieved = attribute_compute_fn(tokens, "REMI")
    history = [(len(tokens), achieved)]

    for step in range(max_removals):
        if abs(achieved - target) <= tolerance:
            break
        candidates = find_note_triples(tokens, note_type=note_type)
        if not candidates:
            if verbose:
                print(f"  [edit step {step}] no more {note_type} candidates to remove")
            break

        half = score_window // 2
        edited_windows = []
        for start in candidates:
            edited = tokens[:start] + tokens[start + 3:]
            w_start = max(0, start - half)
            w_end = min(len(edited), start + half)
            edited_windows.append(edited[w_start:w_end])

        ebt_energies = _batched_ebt_energy(model, edited_windows, device)
        reg_preds = _batched_regressor_predict(regressor, emb_weight, edited_windows, device)

        best_i, best_score = None, None
        for i, (ebt_e, pred) in enumerate(zip(ebt_energies, reg_preds)):
            reg_e = (pred - target) ** 2
            score = ebt_e + lam * reg_e
            if best_score is None or score < best_score:
                best_score, best_i = score, i

        best_start = candidates[best_i]
        tokens = tokens[:best_start] + tokens[best_start + 3:]
        achieved = attribute_compute_fn(tokens, "REMI")
        history.append((len(tokens), achieved))
        if verbose:
            print(f"  [edit step {step}] removed {note_type} triple at {best_start}  "
                  f"score={best_score:.4f}  achieved={achieved:.4f}  target={target:.4f}  "
                  f"n_remaining_candidates={len(candidates) - 1}")

    return tokens, achieved, history


def main():
    """
    Small standalone test: generate one unguided baseline for a given prompt,
    then run energy_guided_remove() targeting a reduced attribute value, and
    print/log a direct before/after comparison — meant to be run against the
    SAME prompts already probed with real-time R3 guidance in
    listen_density_sweep.py, for a fair before/after read.
    """
    import argparse
    import json
    import random
    import tempfile

    import wandb

    from inference.mus.infer_ebt import load_checkpoint, load_dataset
    from inference.mus.generate_music import generate_music
    from data.mus.symbolic.tokenization.tokenizer_utils import load_tokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regressor_checkpoint", required=True)
    p.add_argument("--attribute", default=None)
    p.add_argument("--ebt_checkpoint", default=None)
    p.add_argument("--tokenizer_type", default=None)
    p.add_argument("--prompt_indices", type=str, required=True,
                   help="Comma-separated dataset indices to test, e.g. '8484,9938'")
    p.add_argument("--target_delta_std", type=float, default=-2.0,
                   help="Target = this prompt's own baseline + delta_std * corpus_stdev")
    p.add_argument("--note_type", default="drum", choices=["drum", "pitch", "any"])
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--max_removals", type=int, default=30)
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--score_window", type=int, default=64)
    p.add_argument("--prompt_len", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb_project", default="mus_symb_attr_control")
    p.add_argument("--wandb_run_name", default=None)
    args = p.parse_args()

    device = args.device
    reg_ckpt = torch.load(args.regressor_checkpoint, map_location="cpu", weights_only=False)
    ebt_checkpoint = args.ebt_checkpoint or reg_ckpt["ebt_checkpoint"]
    tokenizer_type = args.tokenizer_type or reg_ckpt.get("tokenizer_type", "REMI")
    attribute = args.attribute or reg_ckpt.get("attribute", "density")
    compute_fn = ATTRIBUTES[attribute]
    corpus_stats = load_corpus_stats(attribute)
    corpus_mean, corpus_stdev = corpus_stats["mean"], corpus_stats["stdev"]
    print(f"Regressor:  {args.regressor_checkpoint}")
    print(f"Attribute:  {attribute}")
    print(f"EBT ckpt:   {ebt_checkpoint}")

    model, hparams = load_checkpoint(ebt_checkpoint, device)
    hparams.device = device
    hparams.infer_max_gen_len = args.gen_len
    hparams.infer_temp = args.temperature
    hparams.infer_topp = args.top_p
    hparams.infer_logprobs = False
    hparams.infer_echo = False
    hparams.infer_ebt_advanced = False
    hparams.tokenizer_type = tokenizer_type
    hparams.attribute_target = None
    hparams.lambda_attribute = 0.0
    hparams.attribute_regressor_ckpt = None

    emb_dim = reg_ckpt["emb_dim"]
    hidden = reg_ckpt.get("hidden_dim", 256)
    regressor = NoteDensityRegressor(emb_dim=emb_dim, hidden_dim=hidden)
    regressor.load_state_dict(reg_ckpt["model_state"])
    regressor.eval().to(device)
    emb_weight = model.embeddings.weight.detach().to(device)

    tokenizer, vocab_size, pad_token_id = load_tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_config_path=getattr(hparams, "tokenizer_config_path", None),
        dataset_name=getattr(hparams, "dataset_name", "giga_midi"),
    )
    dataset = load_dataset(hparams, split=args.split)
    bigram_table = build_bigram_logprob_table(
        dataset, model.embeddings.weight.shape[0], n_songs=3000, seed=0,
        cache_path=str(Path(args.regressor_checkpoint).parent.parent / f"_bigram_table_{tokenizer_type}.npy"),
    )

    out_dir = Path(tempfile.mkdtemp(prefix="energy_guided_edit_"))
    run_name = args.wandb_run_name or f"edit-{Path(args.regressor_checkpoint).parent.name}"
    wandb.init(project=args.wandb_project, name=run_name, job_type="energy_guided_edit",
               config=vars(args))
    columns = ["prompt_id", "condition", "achieved", "achieved_zscore", "n_tokens",
               "bigram_ll", "repetition_ratio", "grammar_violation_rate", "ebt_energy",
               "n_removals", "audio"]
    table_rows = []

    prompt_indices = [int(x) for x in args.prompt_indices.split(",")]
    for pi in prompt_indices:
        full_tokens = dataset.get_full_tokens(pi)
        prompt = full_tokens[: min(args.prompt_len, len(full_tokens))]
        batch = {"input_ids": torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)}
        with torch.no_grad():
            out = generate_music(model, batch, hparams)
        baseline_gen = out["generation_tokens"][0]
        baseline_val = compute_fn(baseline_gen, tokenizer_type)
        target = baseline_val + args.target_delta_std * corpus_stdev

        print(f"\n[prompt {pi}] baseline {attribute}={baseline_val:.4f}  target={target:.4f}  "
              f"(delta={args.target_delta_std:+.1f} stdev = {args.target_delta_std * corpus_stdev:+.4f})")
        before_music = score_sample(model, baseline_gen, device, bigram_table)
        before_energy = ebt_self_energy(model, baseline_gen, device)
        print(f"  before: {before_music}  ebt_self_energy={before_energy:.4f}")

        edited, achieved, history = energy_guided_remove(
            baseline_gen, model, regressor, emb_weight, device, target, compute_fn,
            note_type=args.note_type, lam=args.lam, max_removals=args.max_removals,
            tolerance=args.tolerance, score_window=args.score_window,
        )
        after_music = score_sample(model, edited, device, bigram_table)
        after_energy = ebt_self_energy(model, edited, device)
        print(f"  after:  {after_music}  ebt_self_energy={after_energy:.4f}  "
              f"n_removals={len(history) - 1}  final_len={len(edited)}")

        before_wav = render_wav(baseline_gen, tokenizer, out_dir, f"p{pi}_before")
        after_wav = render_wav(edited, tokenizer, out_dir, f"p{pi}_after")
        table_rows.append([
            pi, "before (baseline gen)", baseline_val, zscore(baseline_val, corpus_mean, corpus_stdev),
            len(baseline_gen), before_music.get("bigram_ll"), before_music.get("repetition_ratio"),
            before_music.get("grammar_violation_rate"), before_energy, 0,
            wandb.Audio(before_wav, caption=f"p{pi} before {attribute}={baseline_val:.3f}"),
        ])
        table_rows.append([
            pi, "after (energy-guided edit)", achieved, zscore(achieved, corpus_mean, corpus_stdev),
            len(edited), after_music.get("bigram_ll"), after_music.get("repetition_ratio"),
            after_music.get("grammar_violation_rate"), after_energy, len(history) - 1,
            wandb.Audio(after_wav, caption=f"p{pi} after {attribute}={achieved:.3f} target={target:.3f}"),
        ])
        table = wandb.Table(columns=columns, data=table_rows)
        wandb.log({"edit_comparison": table})

    wandb.finish()
    print(f"\nWandB run: {run_name} (project={args.wandb_project})")


if __name__ == "__main__":
    main()
