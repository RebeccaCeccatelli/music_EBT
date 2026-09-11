"""
Find the highest-step checkpoint matching a glob pattern whose SAVED
hyperparameters are actually compatible with the current run's stabilization
config (Langevin noise, MCMC replay buffer, randomized step size) — not just
the highest step number regardless of what hyperparameters produced it.

Why this exists: the shell scripts' checkpoint auto-detect glob matches any
run sharing the same base name (model/tokenizer/size), which can span
multiple, hyperparameter-incompatible training lineages over time (e.g. a
pre-stabilization run and a post-stabilization run both named
"ebt-symb-small-remi-s1-job<id>_..."). Blindly resuming into the
highest-step match regardless of lineage would silently splice incompatible
weights into a differently-configured continuation. This script checks each
candidate's own saved hparams before accepting it, starting from the highest
step and working down, and prints nothing (safe: caller falls back to
whatever it does when no checkpoint is found) if none match.

Usage:
    python find_compatible_ckpt.py --pattern "<glob>" \
        --langevin_dynamics_noise 0.01 --mcmc_replay_buffer 1 \
        --mcmc_replay_buffer_size 32 --randomize_mcmc_step_size_scale 2.0 \
        --randomize_mcmc_num_steps 0
Prints the compatible checkpoint's path to stdout, or nothing.
"""
import argparse
import glob
import re
import sys

import torch

# Absolute tolerance for float hparam comparison (avoids float round-trip
# mismatches like 0.01 vs 0.009999999).
_FLOAT_TOL = 1e-6


def _matches(saved, expected_str, key, is_float=False):
    """None means "current run didn't pass this flag" -> treat any saved
    value as compatible for that key (don't over-constrain on unset flags)."""
    if expected_str is None or expected_str == "":
        return True
    saved_val = saved.get(key)
    if saved_val is None:
        return False
    if is_float:
        try:
            return abs(float(saved_val) - float(expected_str)) < _FLOAT_TOL
        except (TypeError, ValueError):
            return False
    return str(saved_val) == str(expected_str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", required=True,
                    help="Glob pattern for candidate .ckpt files")
    p.add_argument("--langevin_dynamics_noise", default=None)
    p.add_argument("--randomize_mcmc_step_size_scale", default=None)
    p.add_argument("--randomize_mcmc_num_steps", default=None)
    p.add_argument("--mcmc_replay_buffer", default=None,
                    help="'1' if the current run passes --mcmc_replay_buffer, else unset")
    p.add_argument("--mcmc_replay_buffer_size", default=None)
    args = p.parse_args()

    candidates = []
    for f in glob.glob(args.pattern):
        m = re.search(r"step=step=(\d+)", f)
        if m:
            candidates.append((int(m.group(1)), f))
    candidates.sort(key=lambda x: x[0], reverse=True)

    expected_replay_buffer_bool = (
        None if args.mcmc_replay_buffer in (None, "") else (args.mcmc_replay_buffer == "1")
    )

    for step, path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [find_compatible_ckpt] skipping unreadable {path}: {e}", file=sys.stderr)
            continue
        hp = ckpt.get("hyper_parameters", {})

        ok = (
            _matches(hp, args.langevin_dynamics_noise, "langevin_dynamics_noise", is_float=True)
            and _matches(hp, args.randomize_mcmc_step_size_scale, "randomize_mcmc_step_size_scale", is_float=True)
            and _matches(hp, args.randomize_mcmc_num_steps, "randomize_mcmc_num_steps", is_float=True)
            and _matches(hp, args.mcmc_replay_buffer_size, "mcmc_replay_buffer_size", is_float=True)
        )
        if expected_replay_buffer_bool is not None:
            saved_rb = hp.get("mcmc_replay_buffer")
            ok = ok and (saved_rb is not None) and (bool(saved_rb) == expected_replay_buffer_bool)

        if ok:
            print(path)
            return
        print(f"  [find_compatible_ckpt] incompatible hparams, skipping step={step}: {path}", file=sys.stderr)

    # No compatible checkpoint found among the candidates — print nothing;
    # the caller must treat this the same as "no checkpoint found at all",
    # NOT fall back to an incompatible one.


if __name__ == "__main__":
    main()
