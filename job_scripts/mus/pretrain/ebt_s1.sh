#!/bin/bash
### EBT Symbolic Music (MIDI Tokens) - Stage 1 Pretraining Script
### Trains on tokenized MIDI using iterative refinement (MCMC-style)

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --mem=160GB
#SBATCH --partition=gpu

### ADDITIONAL RUN INFO ###
#SBATCH --array=0

### LOG INFO ###
#SBATCH --job-name=ebt-symb-xxs-bs_256_s1_lr_
#SBATCH --output=logs/slurm/mus/ebt-symb-xxs-bs_256_s1_lr_%A-%a.log
export RUN_NAME="ebt-symb-xxs-bs_256_s1_lr_"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/mus/

module purge

lr=(0.001)
alpha=(500)
alpha_lr=(1500)

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "MUS_SYMB" \
--model_name "ebt" \
--model_size ${MODEL_SIZE} \
\
--tokenizer_type "REMI" \
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_step_size_lr_multiplier ${alpha_lr[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_num_steps 2 \
\
--context_length 512 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 32 \
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 1000000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "giga_midi" \
--num_workers 12 \
--validation_split_pct 0.01 \
--val_check_interval 5000 \
\
--wandb_project 'mus_symbolic_pretrain' \
\
--log_model_archi \
--log_gradients \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}

# NOTES:
# - Change --dataset_name to use custom datasets (e.g., "custom_midi_small")
# - Use --tokenizer_config_path if using custom tokenizer config
# - Adjust --context_length based on your typical sequence length
# - Stage 2 can load this checkpoint with --checkpoint_path and increase --mcmc_num_steps
