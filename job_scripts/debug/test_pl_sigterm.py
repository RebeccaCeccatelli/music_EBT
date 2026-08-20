#!/usr/bin/env python3
"""
Simulates PyTorch Lightning's SIGTERM behaviour:
  - Catches SIGTERM, saves a checkpoint, exits 0.
  - Reads prior checkpoint on startup to continue from saved step.

Usage (via the bash wrapper):
  CKPT_FILE=/path/to/ckpt.txt MAX_STEPS=25 python test_pl_sigterm.py
"""

import os
import sys
import signal
import time

CKPT_FILE = os.environ.get("CKPT_FILE", "/tmp/test_pl_ckpt.txt")
MAX_STEPS  = int(os.environ.get("MAX_STEPS", "25"))
STEP_SECS  = float(os.environ.get("STEP_SECS", "8"))

# ── Resume from checkpoint ────────────────────────────────────────────────────
current_step = 0
if os.path.exists(CKPT_FILE):
    with open(CKPT_FILE) as f:
        current_step = int(f.read().strip())
    print(f"[Python] Resumed from checkpoint: step {current_step}/{MAX_STEPS}", flush=True)
else:
    print(f"[Python] No checkpoint found, starting fresh.", flush=True)

# ── SIGTERM handler — mirrors PL's GracefulKill ───────────────────────────────
_sigterm = False

def _handle_sigterm(signum, frame):
    global _sigterm
    _sigterm = True
    print(f"[Python] SIGTERM received at step {current_step}. Saving checkpoint...", flush=True)
    with open(CKPT_FILE, "w") as f:
        f.write(str(current_step))
    print(f"[Python] Checkpoint saved (step={current_step}). Exiting cleanly.", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"[Python] Training: steps {current_step}→{MAX_STEPS}, {STEP_SECS}s per step", flush=True)

while current_step < MAX_STEPS:
    time.sleep(STEP_SECS)
    current_step += 1
    # Save last.ckpt every step (like PL does)
    with open(CKPT_FILE, "w") as f:
        f.write(str(current_step))
    print(f"[Python] step={current_step}/{MAX_STEPS}", flush=True)

print(f"[Python] Training complete at step {current_step}.", flush=True)
sys.exit(0)
