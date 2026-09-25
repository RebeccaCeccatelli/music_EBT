# Training runs: hyperparameters and settings

Reference doc for every model actually used in the demo/thesis — EBT (REMI,
Anticipation x2 lineages) and the Llama/GPT-2 baselines (REMI, Anticipation).
Pulled directly from each run's wandb metadata (`run.metadata['args']`, the
exact CLI argv `train_model.py` was launched with), not from memory — see
"How to reproduce" below to re-pull or verify.

All runs share the same underlying architecture size (`model_size=small`),
dataset (`giga_midi`, the crowd-sourced GigaMIDI corpus, 5% held out for
validation), optimizer settings (AdamW-style: `peak_learning_rate=0.0008`,
`weight_decay=0.05`, `gradient_clip_val=1.0`, `min_lr_scale=10`,
`warm_up_steps=10000`, `max_steps`/`max_scheduling_steps=100000`), and
hardware (1x NVIDIA L40S per job).

## Overview

| Run | Model | Tokenizer | Context len | Batch/device × accum = effective | Final step | Final valid_loss | wandb state |
|---|---|---|---|---|---|---|---|
| EBT REMI (s1) | `ebt` | REMI | 512 | 4 × 64 = 256 | 19,006 | 0.7519 | crashed* |
| EBT Anticipation, original (s1) | `ebt` | Anticipation-Arrival-Time | 512 | 4 × 16 = 64 | 99,014 | 1.0215 | finished |
| EBT Anticipation, stabilized | `ebt` | Anticipation-Arrival-Time | 512 | 2 × 32 = 64 | 33,006 | 1.0593 | crashed* |
| Llama baseline, REMI | `baseline_llama_transformer` | REMI | 1024 | 16 × 16 = 256 | 99,373 | 0.5239 | crashed* |
| GPT-2 baseline, REMI | `baseline_hf_gpt2_transformer` | REMI | 1024 | 16 × 16 = 256 | 99,144 | 0.4756 | crashed* |
| Llama baseline, Anticipation | `baseline_llama_transformer` | Anticipation-Arrival-Time | 1024 | 8 × 16 = 128 | 99,019 | 0.7735 | finished |
| GPT-2 baseline, Anticipation | `baseline_hf_gpt2_transformer` | Anticipation-Arrival-Time | 1024 | 8 × 16 = 128 | 99,019 | 0.7577 | finished |

\* "crashed" here means the wandb run object for that specific resumed segment
ended non-gracefully (e.g. hit its SLURM wallclock limit, mid-resubmit) — see
`job_scripts/mus/pretrain/ebt_s1.sh`'s self-resubmit mechanism (diary,
2026-09-2x entries). It does **not** mean the training itself failed;
REMI and Llama/GPT-2-REMI are all still actively continuing or already usable
lineages. Only the *last wandb segment queried* is reflected here — total
step counts for REMI in particular have moved past 19,006 since (see
`docs/diary/` for day-by-day progress).

**"finished" does not mean "converged" or "saw the full dataset" — it means
the run hit its configured `max_steps=100000` budget and stopped there by
design.** Confirmed directly from the EBT Anticipation (original) log:
`Trainer.fit stopped: max_steps=100000 reached` — a clean, intentional stop,
immediately followed by `Epoch 0: 44%|...| 1,600,000/3,677,762` in the same
log line. Because Anticipation's dataset is ~107x larger than REMI's (see
the vocab-normalization finding), the same 100,000-step budget that carries
REMI through 35+ full epochs gets an Anticipation run only **44% through its
very first epoch**. This applies to all three "finished" runs in the table
above (EBT Anticipation original, and both Anticipation baselines) — treat
their final valid_loss as an under-trained snapshot, not a converged result,
when citing or comparing against REMI's numbers.

## EBT-specific settings (all three EBT runs)

EBT's own MCMC-refinement hyperparameters, not applicable to the Llama/GPT-2
baselines:

| Setting | REMI | Anticipation (original) | Anticipation (stabilized) |
|---|---|---|---|
| `ebt_type` | time_embed | time_embed | time_embed |
| `denoising_initial_condition` | random_noise | random_noise | random_noise |
| `normalize_initial_condition` | on | on | on |
| `mcmc_num_steps` | 2 | 2 | 2 |
| `mcmc_step_size` (initial) | 1 | 1 | 1 |
| `mcmc_step_size_learnable` | on | on | on |
| `mcmc_step_size_lr_multiplier` | **20** | **2** | **2** |
| `mcmc_step_size_max` | 2.0 | 2.0 | 2.0 |
| `clamp_futures_grad` | on | on | on |
| `langevin_dynamics_noise` | 0.01 | *(not set)* | 0.01 |
| `randomize_mcmc_step_size_scale` | 2.0 | *(not set)* | 2.0 |
| `mcmc_replay_buffer` | on | *(not set)* | on |
| `mcmc_replay_buffer_size` | 32 | *(not set)* | 32 |

**The "stabilized" Anticipation lineage is exactly what its name says**: a
deliberate rerun of Anticipation pretraining with three additions the
original lineage never had — Langevin dynamics noise, randomized MCMC
step-size scale, and a replay buffer — matching REMI's own configuration.
This was the fix explored in the batch-size/step-size probe investigation
(`docs/thesis_findings/`, "Open threads" in the vocab-normalization finding)
for Anticipation's higher gradient norms and less stable MCMC steps. The
`mcmc_step_size_lr_multiplier` difference (20 for REMI vs. 2 for both
Anticipation lineages) was the other half of that same investigation — raising
Anticipation's multiplier to 20 to match REMI showed no improvement in a
direct probe, so it was left at 2 for both Anticipation lineages.

**What `randomize_mcmc_step_size_scale` actually does**, confirmed directly
in `model/mus/ebt_symbolic.py`: the MCMC step size is a learnable parameter,
`self.alpha` (an `nn.Parameter`, clamped to `[0.0001, mcmc_step_size_max]`).
Without randomization, every refinement step for every training example uses
that exact same single learned value. With `randomize_mcmc_step_size_scale`
set (2.0 for both REMI and the stabilized Anticipation lineage):

```python
alpha = torch.clamp(self.alpha, min=0.0001, max=alpha_max)
low = alpha / scale
high = alpha * scale
alpha = low + torch.rand_like(expanded_alpha) * (high - low)
```

— each step instead resamples alpha uniformly from `[alpha/2, alpha*2]`
around the learned value, rather than always using it exactly. The point:
a single fixed step size is brittle to how much the energy landscape's local
curvature actually varies across different examples and steps — a value
well-tuned for one case can overshoot or undershoot badly for another.
Randomizing it forces the refinement process to work across a *spread* of
step sizes instead of overfitting to one specific value, the same idea
behind stochastic learning-rate schedules elsewhere, applied here to the
inner MCMC loop rather than the outer training loop.

## Effective batch size note

"Effective batch" = `batch_size_per_device × accumulate_grad_batches` (single
GPU for every run here, so no additional multiplication by GPU count).
REMI and the REMI-tokenized baselines all land on effective batch **256**;
Anticipation's is smaller (64 for EBT, 128 for the baselines) — driven by
Anticipation's larger per-token memory footprint (larger vocabulary — 55,028
vs. REMI's 427 — and its context length differs: EBT stays at 512 for both
tokenizers, while the baselines use 1024 for both). This effective-batch gap
is one of the two hypotheses (along with MCMC step-size) investigated for why
Anticipation pretraining looks like it's converging more slowly by step count
— see the vocab-normalization finding for the fuller picture (Anticipation's
dataset is also ~107x more sequences than REMI's, so far more steps are
needed per epoch regardless of batch size).

## Why EBT takes longer to get through the data than Llama/GPT-2

Comparing the three Anticipation runs directly (all share
`accumulate_grad_batches=16`, so they reach the same raw-iteration count —
1,600,000 — at the same `global_step=100000`, from the wandb-logged
`Trainer.fit stopped: max_steps=100000 reached` line in each run's log):

| Run | batch/device | it/s (raw) | Sequences/sec | % through Epoch 0 at step 100,000 |
|---|---|---|---|---|
| EBT (original) | 4 | 47.94 | ~192 | **44%** (1.6M / 3.68M raw iters) |
| Llama baseline | 8 | 24.48 | ~196 | **87%** (1.6M / 1.84M raw iters) |
| GPT-2 baseline | 8 | 20.71 | ~166 | **87%** (1.6M / 1.84M raw iters) |

**EBT's actual per-sample throughput is not meaningfully slower** than the
baselines' — all three land in the same ~165–196 sequences/second
ballpark. The real cause is that **EBT trains at half the per-device batch
size** (4 vs. 8) while every run shares the same optimizer-step budget
(`max_steps=100000`) — so in the same number of training steps, EBT simply
processes half as much raw data as Llama/GPT-2 do. Neither Llama nor GPT-2
actually completed a full epoch either (87%, not 100%) — but they got
almost twice as far through the same dataset in the same step budget purely
from the larger batch.

The batch-size gap itself is architectural, and confirmed directly in
`model/mus/ebt_symbolic.py` rather than inferred from the hyperparameter
names — it's a genuine second-order-gradient ("double backward")
computation:

```python
torch.autograd.grad(..., create_graph=create_graph, retain_graph=True, ...)
```

Inside a single forward pass, EBT runs its `mcmc_num_steps` (2) refinement
loop. At each step it computes an energy value and calls
`torch.autograd.grad(...)` to get a gradient, which is what actually updates
the prediction toward lower energy. During training, `create_graph=True`
(`create_graph = learning`) — meaning PyTorch keeps that gradient's own
computation graph alive, not just its numeric value. This is required
because `mcmc_step_size` is itself a learnable parameter
(`mcmc_step_size_learnable=True`): for the outer training loss to produce a
correct gradient *for the step size*, the outer backward pass has to
differentiate through the act of "taking a gradient and using it to step" —
the same class of computation as MAML-style meta-learning or
backprop-through-optimization. That needs the graph of the inner gradient's
own construction retained, not just its value, which is substantially more
memory per training example than a standard transformer's single
forward-then-backward pass — and the cost multiplies with `mcmc_num_steps`
(here, 2 separate `create_graph=True` steps per example). That's the
concrete reason a smaller `batch_size_per_device` was needed to fit the same
L40S GPU memory budget that comfortably holds Llama/GPT-2's simpler
single-pass architecture at 2x the batch size. (The "stabilized" Anticipation
lineage's replay buffer pushes this further — its batch/device drops to 2,
doubling the memory overhead again for the buffer itself.)

In short: **EBT isn't fundamentally slower per example — it's fundamentally
more memory-hungry per example**, and under this project's fixed-step-count
training regimen (rather than a fixed-wall-clock or fixed-epochs-seen
regimen), that memory overhead directly translates into less total data
coverage per unit of training budget.

## Regressor-checkpoint pairing (as of 2026-09-24)

Attribute regressors (velocity/duration/pitch_register) are trained against
one *specific, frozen* EBT checkpoint's embedding space — using a checkpoint
newer than that (embeddings keep drifting during continued pretraining)
degrades guidance quality even though the regressor itself is valid (see
diary, 2026-09-24). As of this writing:

- **REMI regressors**: retraining against `job22827004`, epoch 36, step
  19,496, valid_loss 0.7519 (jobs 23662484/23662485/23662486, in progress).
  Previously matched to an older checkpoint, `job21290401` step 10,584 —
  note that checkpoint's labeled valid_loss (0.6038) is itself a likely
  spurious post-resume validation artifact (see
  `docs/thesis_findings/2026-09-22_anticipation_loss_vocab_normalization.md`'s
  correction note) and should not be read as a genuine quality signal.
- **Anticipation regressors** (duration, pitch_register only — Anticipation's
  vocabulary has no velocity field): matched to
  `ebt-symb-small-ant-at-full-s1-job21959744`, step 55,500, valid_loss
  0.3335 — from a separate, earlier Anticipation checkpoint lineage not
  listed in the table above (predates both lineages there).

## How to reproduce this doc / re-pull current values

```python
import wandb
api = wandb.Api()
run = api.run("rceccatelli-eth-z-rich/mus_symb_ebt_pretrain/<wandb_run_id>")
run.metadata["args"]      # exact CLI argv
run.summary["_step"]      # last logged step
run.summary["valid_loss"] # last logged validation loss
```

Each checkpoint directory under
`~/orcd/scratch/rebcecca/music_EBT_logs/checkpoints/<run>/` has its own
`wandb_run_id.txt` — read that to get the exact run ID for any checkpoint,
rather than guessing from the run name (a resumed lineage can span several
job IDs and wandb run IDs over time).

wandb projects actually used: `mus_symb_ebt_pretrain` (EBT),
`mus_symb_baseline_pretrain` (Llama/GPT-2), `mus_symb_attr_control`
(regressors, listening sweeps, compose runs) — all under the
`rceccatelli-eth-z-rich` entity.
