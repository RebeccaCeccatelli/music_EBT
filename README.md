# 🎹 Tonic: Energy-Based Transformers for Symbolic Music Generation and Single- and Multi-Attribute Control

**A systematic comparison with autoregressive Transformers and alternative steering methods**

🏠 [Tonic](https://rebeccaceccatelli.github.io/tonic/) | 📖 [Thesis (PDF)](https://rebeccaceccatelli.github.io/tonic/thesis/Tonic_thesis.pdf) | 🎧 [Showcase](https://rebeccaceccatelli.github.io/music_EBT/) | 🗂️ [Code structure](docs/CODE_STRUCTURE.md) | 📊 [Findings](docs/thesis_findings/README.md) | 🧭 [Project status](docs/STATUS.md)

This is the codebase of an ETH Zürich / MIT Media Lab master's thesis (submitted November 2026). The thesis covers three things:
- **Tonic**, the first Energy-Based Transformer (EBT) for symbolic music, to our knowledge.
- **A comparison** between Tonic and autoregressive Llama and GPT-2 models of the same size, trained on the same data with the same recipe.
- **Attribute control**: steering a frozen model towards requested values of musical attributes (loudness, note length, pitch register), one at a time or several combined. Tonic's energy-based control is compared with established plug-and-play methods for the AR models.

## Why energy-based models

Today's music models generate impressive music. The open question is **how precisely a musician can shape what they produce**. A composer or producer usually wants a continuation that is softer, higher or more sustained than the model's own, by a chosen amount, while still sounding like music. Current symbolic models mostly offer coarse, global conditions such as genre or mood. Those controls are fixed at training time, and each new one needs labelled data and a new training run.

Plug-and-play control avoids retraining: a small regressor steers a frozen model at generation time. Autoregressive (AR) Transformers, however, commit to each token in a single forward pass and **never revisit a decision**. Guidance can only be added from outside: by nudging the next-token distribution, reranking samples, or perturbing hidden states.

**Energy-based models** are built differently. An EBT ([Gladstone et al., 2025](https://arxiv.org/abs/2507.02092)) does not output a prediction directly. It assigns an energy to each candidate and **reaches its prediction by minimising that energy step by step**. This structure gives three advantages we hoped to use:
- **Composability.** Energies add up ([Du et al., 2023](https://arxiv.org/abs/2302.11552)). Each musical request becomes its own energy term, with its own target and weight, and enters the model's own refinement loop. Requests can be combined freely at generation time.
- **Finer control.** Requests are continuous, per-attribute targets rather than a fixed set of global labels, and adding a new one only needs a small regressor whose labels come straight from the MIDI.
- **Interpretability.** The output is the result of an explicit optimisation, step by step, so the model's behaviour is easier to inspect than a single forward pass.

The thesis tests whether these advantages hold up for music, in four research questions:
1. Can an EBT be trained on symbolic music and generate coherent, musically plausible continuations?
2. How does it compare with AR Transformers of the same size, trained on the same data with the same recipe, in likelihood and musical quality?
3. Can a frozen EBT be steered towards target attribute values, singly and in combination, by adding attribute energies to its inference loop?
4. How does this compare with plug-and-play steering of AR models (PPLM, exponential tilting, best-of-N), in control accuracy, musical quality and compute?

## What's in this project

- **Tonic**: a small EBT trained on the [GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI) corpus.
- **Baselines**: Llama and GPT-2 trained with the same recipe and step budget.
- **Single-attribute control**: for each attribute, a small regressor's gradient is added to the model's prediction ([R³](https://arxiv.org/abs/2302.11552)-style guidance), gated to the token types that can move it. The AR baselines are steered with PPLM, tilting and best-of-N.
- **Multi-attribute control**: several attributes requested in one generation. Tonic sums their energies; the AR methods are run with the same combined requests.
- **Evaluation**: likelihood on matched windows, music-quality metrics against real data, controllability vs. quality trade-offs, paired significance tests, and a blind listening survey.

Everything runs with two tokenizations:
- **REMI** (via [MidiTok](https://github.com/Natooz/MidiTok)): a flat stream of bar, position, pitch, velocity and duration tokens.
- **Anticipation**, arrival-time variant ([Thickstun et al.](https://github.com/jthickstun/anticipation), our [fork](https://github.com/RebeccaCeccatelli/anticipation-musicEBT) is a git submodule): (time, duration, note) triplets.

## Highlights

- **Unguided quality.** All models get close to the real-data reference on REMI. Tonic scores slightly lowest. Its typical failure is safe, repetitive music, not cacophony. → [findings](docs/thesis_findings/2026-10-06_unguided_music_quality.md)
- **Single-attribute control.** On REMI, Tonic's guidance is statistically tied with Llama + PPLM. On Anticipation, PPLM and best-of-N come out ahead. Tonic's guidance only works in a narrow strength (λ) window, and pushing harder flattens the music into cacophony. → [findings](docs/thesis_findings/2026-11-05_heldout_guidance.md)
- **Multi-attribute control.** Combined requests (e.g. loudness + note length) are evaluated for Tonic and every AR method. See the [thesis](https://rebeccaceccatelli.github.io/tonic/thesis/Tonic_thesis.pdf) for results.
- **Demos.** The [Tonic front page](https://rebeccaceccatelli.github.io/tonic/), a static [showcase](https://rebeccaceccatelli.github.io/music_EBT/) of curated clips, a blind listening survey with its [results](https://rebeccaceccatelli.github.io/tonic/survey-results/), the [Listening Archive](https://rebeccaceccatelli.github.io/listening-archive/) (~20,000 clips from the final runs), and an interactive Gradio app (*Tonic · Interactive Demo*). The app puts Tonic, Llama and GPT-2 side by side, with live attribute control.

For how guidance works in detail (gating, λ, targets), see [docs/attribute_control_mechanism.md](docs/attribute_control_mechanism.md).

## Repository at a glance

| Path | What's there |
|---|---|
| `train_model.py`, `base_model_trainer.py` | Entry point and PyTorch Lightning trainer, shared by every model and modality |
| `model/mus/` | Music models: `ebt_symbolic.py` (Tonic), `baseline_llama_transformer.py`, `baseline_hf_gpt2_transformer.py` |
| `data/mus/symbolic/` | GigaMIDI / custom dataloaders, REMI + Anticipation tokenizers |
| `inference/mus/` | Generation for all models (incl. guided), tokens → MIDI → WAV / piano roll |
| `attribute_control/` | Attribute definitions, regressor training, single-attribute guidance sweeps (Tonic and AR baselines), multi-attribute composition |
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
