#!/bin/bash
### EBT Neural Music (Audio Waveforms) - Pretraining Script
### Trains on raw/encoded audio waveforms using iterative refinement

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
#SBATCH --job-name=ebt-neur-xxs-bs_256_lr_
#SBATCH --output=logs/slurm/mus/ebt-neur-xxs-bs_256_lr_%A-%a.log
export RUN_NAME="ebt-neur-xxs-bs_256_lr_"
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
--modality "MUS_NEUR" \
--model_name "ebt" \
--model_size ${MODEL_SIZE} \
\
--audio_sample_rate 16000 \
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_step_size_lr_multiplier ${alpha_lr[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_num_steps 2 \
\
--context_length 1024 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 16 \
--accumulate_grad_batches 4 \
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
--wandb_project 'mus_neural_pretrain' \
\
--log_model_archi \
--log_gradients \
--infer_generate_music \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}

# NOTES:
# - Longer context (1024 vs 512) for audio sequences
# - Smaller batch size (16 vs 32) due to larger audio representations
# - Higher gradient accumulation (4 vs 2) to maintain effective batch size
# - Audio will be logged to W&B during inference via --infer_generate_music
# - Adjust --audio_sample_rate if using different sample rates
