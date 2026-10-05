# Validation loss at matched tokens seen: EBT vs GPT-2 / Llama

_2026-10-05. Script: `eval/plot_tokens_matched_curves.py`. Figures:
`docs/figures/tokens_matched/`._

## Why
The runs used different effective batch sizes and context lengths
(`docs/training_runs.md`). When EBT stopped, the baselines had seen far more
data: REMI 26B vs 4.5B tokens, Anticipation 13.1B vs 3.3B. Final losses
mostly reflect that, so this compares the models at equal tokens seen.

## Method
- Each model's wandb runs come from the `wandb_run_id.txt` in its checkpoint
  dirs. Overlapping steps are resolved "latest run wins".
- Post-resume validation artifacts are dropped: dips >1.5% below and
  spikes >10% above the median of ±6 neighbouring readings.
- Tokens = step × effective batch × context: EBT REMI 256×512, EBT Ant
  64×512, baselines REMI 256×1024, baselines Ant 128×1024.
- "Loss at budget" is the median of the readings within ±5% of that budget.

## Result

![REMI](../figures/tokens_matched/val_loss_vs_tokens_remi.png)
![Anticipation](../figures/tokens_matched/val_loss_vs_tokens_anticipation_at.png)

| REMI | 0.5B | 1.0B | 2.0B | 4.5B (EBT stop) | final |
|---|---|---|---|---|---|
| EBT | 0.911 | 0.824 | 0.767 | **0.718** | 0.716 (4.5B) |
| GPT-2 | 0.783 | 0.726 | 0.672 | 0.597 | 0.542 (26B) |
| Llama | 0.866 | 0.794 | 0.680 | 0.610 | 0.597 (26B) |

| Anticipation-AT | 0.5B | 1.0B | 2.0B | 3.3B (EBT stop) | final |
|---|---|---|---|---|---|
| EBT | **1.171** | **1.057** | 1.005 | 1.021 | 1.022 (3.3B) |
| GPT-2 | 1.336 | 1.101 | 0.933 | 0.864 | 0.758 (13.1B) |
| Llama | 1.563 | 1.193 | 0.974 | 0.897 | 0.774 (13.1B) |

- **Anticipation: EBT learns faster early, then plateaus.** It has lower loss
  than both baselines up to ~1.2B tokens (−12% / −25% vs GPT-2 / Llama at
  0.5B). After that it flattens at ~1.0 while the baselines keep improving,
  so it is behind at 2B+.
- **REMI: EBT is behind at every matched budget.** For example, 0.718 vs
  0.597 / 0.610 at 4.5B.
- **Llama REMI overfits.** Its loss rises again after ~15B tokens
  (~185 epochs), so its best checkpoint is earlier than its final one.

## Caveats
- Context differs: the baselines train and validate at 1024 tokens, EBT at
  512. Longer context lowers per-token loss on its own, so these tables
  favour the baselines somewhat.
- Per-token loss isn't comparable across tokenizers (vocab 427 vs 55,028):
  compare within a table only.
- The REMI baseline curves start at ~0.7B tokens. Their earlier wandb runs
  aren't referenced by any surviving checkpoint dir.
- EBT REMI between ~2.5–3B tokens spans the resubmission/branching period
  (see `2026-09-25_validation_loss_curves.md`), hence the noise.
