# 🎹 Tonic: an Energy-Based Transformer for Symbolic Music

🏠 [Tonic](https://rebeccaceccatelli.github.io/tonic/) | 🎧 [Showcase](https://rebeccaceccatelli.github.io/music_EBT/) | 🗂️ [Code structure](docs/CODE_STRUCTURE.md) | 📊 [Findings](docs/thesis_findings/README.md) | 🧭 [Project status](docs/STATUS.md)

**Tonic** applies **Energy-Based Transformers (EBTs)** to **symbolic music generation**. This repository is its codebase, built for an ETH Zürich / MIT master's thesis.

An EBT does not output the next token in one forward pass. It learns an energy function over (context, candidate) pairs and **refines its prediction by gradient descent on that energy** (MCMC-style "thinking" steps). The project asks two questions:

1. **Can an EBT model music?** Tonic is a small EBT trained on the [GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI) corpus. We compare it with autoregressive Llama and GPT-2 baselines trained on the same data and step budget.
2. **Does the energy view help with control?** At generation time, Tonic steers musical attributes (loudness, note length, pitch register, …) by adding the gradient of a small learned attribute regressor to its prediction ([R³](https://arxiv.org/abs/2302.11552)-style guidance). We compare this with plug-and-play guidance for the AR baselines: PPLM, exponential tilting, and best-of-N.

Everything runs with two tokenizations:
- **REMI** (via [MidiTok](https://github.com/Natooz/MidiTok)): a flat stream of bar, position, pitch, velocity and duration tokens.
- **Anticipation**, arrival-time variant ([Thickstun et al.](https://github.com/jthickstun/anticipation), our fork is a git submodule): (time, duration, note) triplets.

## Highlights

- **Unguided quality.** All models get close to the real-data reference on REMI. Tonic scores slightly lowest. Its typical failure is safe, repetitive music, not cacophony. → [findings](docs/thesis_findings/2026-10-06_unguided_music_quality.md)
- **Attribute control.** On REMI, Tonic's guidance is statistically tied with Llama + PPLM. On Anticipation, PPLM and best-of-N come out ahead. Tonic's guidance only works in a narrow strength (λ) window, and pushing harder flattens the music into cacophony. → [findings](docs/thesis_findings/2026-11-05_heldout_guidance.md)
- **Demos.** The [Tonic front page](https://rebeccaceccatelli.github.io/tonic/), a static [showcase](https://rebeccaceccatelli.github.io/music_EBT/) of curated clips, a blind listening survey, and an interactive Gradio app (*Tonic · Interactive Demo*). The app puts Tonic, Llama and GPT-2 side by side, with live attribute control.

For how guidance works in detail (gating, λ, targets), see [docs/attribute_control_mechanism.md](docs/attribute_control_mechanism.md).

## Repository at a glance

| Path | What's there |
|---|---|
| `train_model.py`, `base_model_trainer.py` | Entry point and PyTorch Lightning trainer, shared by every model and modality |
| `model/mus/` | Music models: `ebt_symbolic.py` (Tonic), `baseline_llama_transformer.py`, `baseline_hf_gpt2_transformer.py` |
| `data/mus/symbolic/` | GigaMIDI / custom dataloaders, REMI + Anticipation tokenizers |
| `inference/mus/` | Generation for all models (incl. guided), tokens → MIDI → WAV / piano roll |
| `attribute_control/` | Attribute definitions, regressor training, guidance sweeps (EBT and AR baselines), composition of attributes |
| `eval/` | Music-quality metrics, loss comparisons, significance tests, plots |
| `demo/` | Gradio app, static showcase site, listening survey, clip library |
| `job_scripts/mus/` | SLURM scripts for tokenization, pretraining, inference, evaluation, demo builds |
| `docs/` | Status, daily diary, findings, training-run reference, mechanism notes |

A file-by-file overview is in **[docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)**.

## Setup

```bash
git clone --recursive https://github.com/RebeccaCeccatelli/music_EBT.git
cd music_EBT
conda create -n music_EBT python=3.12 && conda activate music_EBT
bash setup_env.sh        # syncs the anticipation submodule, installs requirements, checks the environment
```

`setup_env.sh` puts large datasets under `REMOTE_DATA_STORAGE` (edit it at the top of the script, or set it to `""` to keep data in the repo). It also adds `data/mus/symbolic` to `PYTHONPATH`. Log in to wandb with `wandb login`. You can set `HF_HOME` / `HF_TOKEN` if needed.

> The job scripts target the MIT ORCD cluster: SLURM partitions `mit_normal_gpu` / `mit_preemptable`, L40S GPUs, and data paths under `~/orcd/pool`. Adapt the `#SBATCH` headers and paths to your machine.

## Usage

All commands run from the project root. Each script lists its own options at the top.

**1. Tokenize the data**
```bash
sbatch job_scripts/mus/tokenization/gigaMIDI.sh REMI           # or: anticipation
```

**2. Pretrain**
```bash
sbatch job_scripts/mus/pretrain/ebt_s1.sh                       # Tonic, System-1 EBT (REMI by default; --tokenizer_type to change)
sbatch job_scripts/mus/pretrain/baseline_llama_transformer.sh   # AR baselines
sbatch job_scripts/mus/pretrain/baseline_hf_gpt2_transformer.sh
```
The pretraining scripts resume from the latest compatible checkpoint and resubmit themselves when they hit the wall-clock limit. Hyperparameters of every run used in the thesis are in [docs/training_runs.md](docs/training_runs.md).

**3. Generate**
```bash
CHECKPOINT=/path/to/ebt.ckpt sbatch job_scripts/mus/inference/ebt.sh --num_samples 10
sbatch job_scripts/mus/inference/baseline_llama_transformer.sh
```
These write MIDI and WAV. `inference/mus/` also has converters to piano rolls.

**4. Attribute control**
```bash
sbatch job_scripts/mus/attr_control/train_density.sh            # train an attribute regressor on the model's embeddings
sbatch job_scripts/mus/attr_control/ar_guidance_sweep.sh        # PPLM / tilt / best-of-N on the AR baselines
sbatch job_scripts/mus/attr_control/compose_guidance_sweep.sh   # several attributes at once
```

**5. Evaluate**: scripts in `job_scripts/mus/eval/`, for example `music_quality_generate.sh`, `context_matched_loss.sh` and `unguided_quality_ci.sh`.

**6. Demo**
```bash
bash demo/launch_demo.sh            # Gradio app on a GPU node (SSH tunnel); --share for a public link
```
How to build the showcase and the survey is described in [demo/showcase/README.md](demo/showcase/README.md) and [demo/survey/README.md](demo/survey/README.md).

## Documentation

- [docs/STATUS.md](docs/STATUS.md): current state, final checkpoints, open TODOs
- [docs/diary/](docs/diary/README.md): day-by-day log of what was done and why
- [docs/thesis_findings/](docs/thesis_findings/README.md): self-contained, citable results
- [docs/training_runs.md](docs/training_runs.md): exact settings of every trained model
- [docs/attribute_control_mechanism.md](docs/attribute_control_mechanism.md): how guidance works
- [docs/CODE_INFO.md](docs/CODE_INFO.md): notes on the training framework, inherited from upstream EBT

## Acknowledgements

This repository is a fork of the official **Energy-Based Transformers** code by Alexi Gladstone et al. ([paper](https://arxiv.org/abs/2507.02092), [code](https://github.com/alexiglad/EBT)). The general training framework comes from that codebase, as do the NLP / image / video models that are still in the repo (`model/{nlp,img,vid}`, `data/{nlp,img,vid}`). Tonic itself was added here: the music modality, tokenization, attribute control, evaluation and demos. Anticipation tokenization builds on [jthickstun/anticipation](https://github.com/jthickstun/anticipation). Guidance builds on Du et al., *Reduce, Reuse, Recycle* (2023) and Dathathri et al., *PPLM* (2020).

```bibtex
@misc{gladstone2025energybasedtransformersscalablelearners,
  title={Energy-Based Transformers are Scalable Learners and Thinkers},
  author={Alexi Gladstone and Ganesh Nanduru and Md Mofijul Islam and Peixuan Han and Hyeonjeong Ha and Aman Chadha and Yilun Du and Heng Ji and Jundong Li and Tariq Iqbal},
  year={2025},
  eprint={2507.02092},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.02092},
}
```

See [LICENSE](LICENSE).
