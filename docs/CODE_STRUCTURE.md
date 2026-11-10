# Code structure

A file-level map of the Tonic repository. For what the project is and how to run it, see the [README](../README.md). The NLP / image / video code inherited from upstream EBT is listed only briefly at the end.

## How a run flows

```
job_scripts/mus/pretrain/*.sh ──► train_model.py ──► base_model_trainer.py ──► model/mus/<model>.py
   (SLURM + hparams)               (args, model size,     (Lightning loop, datasets,     (Tonic / Llama / GPT-2)
                                    ckpt inheritance)       optimizer, logging)
                                                               │
                                                               └─► data/mus/symbolic/dataloaders  (tokenized GigaMIDI)

checkpoint ──► inference/mus/generate_music.py ──► tokens ──► MIDI ──► WAV / piano roll
                      ▲
                      └── attribute_control/ (regressor energy → gradient on logits, gated per token type)
```

## Core training framework

| File | Role |
|---|---|
| `train_model.py` | Entry point. Parses all arguments, sets the model size, sets up distributed training, and inherits hparams from a checkpoint when it runs in test or inference mode |
| `base_model_trainer.py` | PyTorch Lightning module: builds the model and datasets (`setup`), and holds the train/val/test steps, optimizer and logging |
| `optimization.py` | LR schedule and optimizer helpers |
| `model/model_utils.py`, `model/ar_ebt_*.py`, `model/ar_transformer.py` | Shared transformer backbones (Llama-2 style) and EBT variants |
| `model/replay_buffer.py` | Replay buffer for System-2 EBTs |

## Music models: `model/mus/`

| File | Role |
|---|---|
| `ebt_symbolic.py` | **Tonic**, the EBT for symbolic music: energy over (context, candidate distribution) with MCMC refinement steps |
| `baseline_llama_transformer.py` | Autoregressive Llama-style baseline |
| `baseline_hf_gpt2_transformer.py` | Autoregressive GPT-2 baseline (HuggingFace) |
| `ebt_neural.py` | Placeholder for a neural-audio EBT (unused) |

## Data and tokenization: `data/mus/symbolic/`

| Path | Role |
|---|---|
| `tokenization/miditok_tokenizer.py` | MIDI → REMI tokens via MidiTok, with quarantine for corrupt files |
| `tokenization/anticipation_tokenizer.py` | MIDI → Anticipation (arrival-time) tokens |
| `tokenization/anticipation/` | Git submodule: our fork of the anticipation library |
| `tokenization/anticipation_decode_wrapper.py` | `decode()` interface for Anticipation tokens, used at inference |
| `tokenization/tokenizer_utils.py` | Shared tokenizer loading |
| `dataloaders/giga_midi_dataloader.py`, `giga_midi_miditok_dataset.py` | GigaMIDI datasets (Anticipation text files / MidiTok JSON) |
| `dataloaders/custom_music_dataloader.py` | Generic loader for any pre-tokenized dataset ([CUSTOM_DATASET_README](../data/mus/symbolic/dataloaders/CUSTOM_DATASET_README.md)) |
| `dataloaders/constants.py`, `path_utils.py` | Paths and constants |
| `datasets/` | Dataset folders (raw data lives in pool storage) |

## Inference: `inference/mus/`

| File | Role |
|---|---|
| `generate_music.py` | Shared generation for all models and both tokenizations. Handles sampling, EBT refinement, attribute guidance and per-token-type gating (`_remi_step_is_relevant`, `_anticipation_step_is_relevant`) |
| `infer_ebt.py` | Tonic inference script (prompts → samples → MIDI/WAV) |
| `infer_baselines_interactive.py` | Same for the Llama/GPT-2 baselines |
| `tokens_to_midi.py`, `midi_to_wav.py`, `midi_to_piano_roll.py`, `tokens_to_piano_roll.py` | Converters |
| `upload_to_wandb.py` | Push inference outputs (audio, piano rolls) to wandb |

## Attribute control: `attribute_control/`

How it works: [attribute_control_mechanism.md](attribute_control_mechanism.md).

| File | Role |
|---|---|
| `attributes.py`, `note_density.py` | Attribute definitions (velocity, duration, pitch register, density, polyphony, …) for both tokenizers, plus the MLP regressor |
| `train_density_regressor.py` | Train an attribute regressor on a model's frozen token embeddings |
| `eval_remi_regressors.py`, `eval_pitch_register_regressor.py` | Check regressor accuracy on real windows |
| `listen_density_sweep.py` | EBT guidance sweeps over prompts × λ × target |
| `ar_guidance.py`, `ar_guidance_sweep.py` | Plug-and-play guidance for the AR baselines (PPLM, tilt, best-of-N) and their sweeps |
| `compose_attributes.py`, `compose_guidance_sweep.py` | Several attributes at once (summed energies) |
| `aggregate_guidance_sweeps.py` | Merge sweep outputs into comparison tables (`sweep_tables/`) |
| `musicality_metrics.py` | Cheap token-level musicality proxies |
| `compute_corpus_histograms.py`, `compute_unigram_entropy.py`, `corpus_*.json` | Corpus statistics for picking targets and normalizing losses |
| `sample_midi.py`, `benchmark_density_control.py`, `track_musicality_over_training.py` | Helpers and older experiments |

## Evaluation: `eval/`

| File | Role |
|---|---|
| `music_quality.py`, `dump_reference_midi.py` | Music-quality metrics on decoded MIDI against a real-data reference |
| `unguided_quality_ci.py`, `paired_guidance_stats.py` | Bootstrap CIs and paired significance tests |
| `score_sweep_quality.py` | Controllability vs. quality for guidance sweeps |
| `context_matched_loss.py`, `mcmc_steps_loss.py` | Loss on identical windows across models; loss per MCMC step |
| `train_flops.py` | Training FLOPs per token |
| `guidance_mechanism.py` | Analysis of why strong EBT guidance degrades music |
| `plot_*.py` | Thesis figures (training curves, tokens-matched curves, sweeps, corpus attributes) |
| `ar_listening_grid.py`, `fetch_listening_table.py`, `log_compose_to_wandb.py` | Listening grids and wandb logging |
| `tests/` | Data and token sanity checks, training-logic tests |

## Demos: `demo/`

| Path | Role |
|---|---|
| `app.py`, `launch_demo.sh` | Gradio app (*Tonic · Interactive Demo*): Tonic / Llama / GPT-2 side by side, single- and combined-attribute control |
| `showcase/` | Static listening page on GitHub Pages ([README](../demo/showcase/README.md)) |
| `survey/` | Blind listening survey: build, candidate rendering and analysis ([README](../demo/survey/README.md)) |
| `library/`, `archive/` | Local clip library and listening archive built from rendered sweep clips |
| `selected_prompts.json` | Prompts used by the demo |

## Job scripts: `job_scripts/mus/`

| Folder | Contents |
|---|---|
| `tokenization/` | Tokenize GigaMIDI or a custom dataset |
| `pretrain/` | EBT (S1/S2) and baseline pretraining. Scripts resume and resubmit themselves. `find_compatible_ckpt.py` picks the checkpoint to resume from |
| `inference/` | Generation for each model |
| `attr_control/` | Regressor training, guidance and composition sweeps |
| `eval/` | Quality scoring, loss comparisons, CIs |
| `demo/` | Demo app and showcase/survey builds |

## Docs: `docs/`

`STATUS.md` (current state), `diary/` (daily log), `thesis_findings/` (citable results), `training_runs.md` (run settings), `attribute_control_mechanism.md`, `thesis_plan.md`, `abbreviations.md`, `CODE_INFO.md` (upstream framework notes).

## Other

- `setup_env.sh`, `requirements.txt`, `environment.yml`, `utils/verify_env.py`: environment setup
- `utils/`: metrics, logging, data debugging
- Local only, git-ignored: `eth-mit-master-thesis/` (thesis LaTeX, its own repo), `papers/`, `listening/`, `logs/`, `wandb/`
- Upstream EBT modalities kept for reference: `model/{nlp,img,vid,diffusion}`, `data/{nlp,img,vid}`, `inference/{nlp,img,vid}`, `job_scripts/{nlp,img,vid}`
