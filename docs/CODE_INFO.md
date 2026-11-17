> **Note:** This file was inherited from the original forked codebase and is no longer fully up to date. Directory structure, requirements files, and some referenced scripts have changed. Use it as a general orientation guide only.
>
> **Removed (2026-11-16):** the image, text and video code (`data/`, `inference/`, `model/` and `job_scripts/` subfolders `img/`, `nlp/`, `vid/`), `utils/dataloader_debugger.py` and the `--debug_dataloader` mode; then `model/diffusion/`, `model/diffusion_transformer.py`, `model/bi_ebt_adaln.py` and the image/video/text helpers in `model/model_utils.py`; on 2026-11-17 the unused `MUS_NEUR` (neural-audio) placeholder (see [CODE_STRUCTURE.md](CODE_STRUCTURE.md#other)). The repository is now music-only; the full original code is at the `thesis-submission` tag.

## Tools/Software being used

- The repository uses [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
  - PL is basically a lightweight wrapper over raw pytorch that helps with:
    - Distributed training
    - Provides simple to use callbacks (see `base_model_trainer.py`)
    - Annoying cuda (`.to('cuda')`) calls
    - Speeds some things up
    - Templatizing code base, all code bases will have a similar structure. 
    - Automates the training loop, no need to call `backwards` and `zero_grad`
- Using weights and biases (WandB) as the logger
  - The wandb logger has been configured to log A TON of info including: hparams (see info tab), console logs (see logs tab, which includes model params, hparams, size of model, model layers/arch, etc), checkpoints (super helpful to not manually dig through folders!), the number of gpus, parameters/gradients, etc.
  - You can also store logs to debug in `logs/console.log` if you are not using wandb (`--no_wandb`)
- Bash and slurm scripts. Slurm scripts are basically bash scripts with an extra header on top specifying stuff about which hardware, allocation, and amount of time to use. 


## Debugging and Other Helpful Things

- [abbreviations.md](./abbreviations.md) contains, you guessed it, several abbreviations used in code throughout the repo!
  - Note that MCMC ~= optimization in the codebase, we avoid the term MCMC in the paper to avoid complexity but that's what it's commonly called in the codebase
- Unfortunately, not all hparams (or arguments) are implemented for all cases so I highly recommend using ctrl f to your advantage and potentially checking if a new hparam is implemented. This repo has evolved a lot over time.
- For debugging quickly see the argparse arguments in `train_model.py` under the DEBUGGING header
  - Several of these can be used to increase the speed of debugging as well as debug NANs/dataloader issues/fitting issues/unused parameters (detect_anomaly, debug_dataloader, overfit_batches, debug_unused_parameters)
  - For example: limit_train_batches, limit_val_batches, limit_test_batches can be used to iterate over a smaller version of the dataset
  - Several of these are supported out of the box using the [Pytorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
- By default WandB captures the logs in the logs tab in the UI which can be useful.
- `slurm_executor.sh` is extremely useful for saving the slurm scripts used/built. If you ever find that you are launching a lot of slurm jobs and want to save the actual job script code I recommend using it.
- If CPU ram is leaking over time, chances are its because either code saving the comp graph or because of a dataloader (i.e. an HF dataloader). Check if it occurs with a dummy/synthetic dataloader, if it occurs with non multi-GPU, etc. If it is CPU RAM it is probably data; if it is GPU RAM is probably tensors.

## Known Minor Issues/TODOs

- *Important* due to perplexity being the exponential of the loss (which is a convex function), perplexities logged in wandb will always upper bound the true perplexity and are not accurate (as we log every n steps rather than waiting to average after the whole training/val epoch). Use the exponential of the logged losses in wandb for accurate ppl calculation (you should use final_loss for EBT in particular to be most accurate, as this is the loss after performing optimization).
- We have not yet implemented the KV cache for inference. Also, the EBT inference is not optimized at all (you no longer need the EBT parallelized prediction implementation, etc.) I suspect it can be made significantly faster with these fixes.
- There is a small very minor bug with the learning rate of non model parameters (they will all decay to the same learning rate value of other parameters).
- Let me know of any other issues and feel free to add issues on github :)!

## Repository structure

See [CODE_STRUCTURE.md](CODE_STRUCTURE.md) for the current file map. The upstream tree (with the text / image / video code) is in this file at the `thesis-submission` tag.
