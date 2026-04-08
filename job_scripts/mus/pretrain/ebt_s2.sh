#!/bin/bash
### EBT Symbolic Music (MIDI Tokens) - Stage 2 Pretraining Script
### Loads stage 1 checkpoint and continues training with more MCMC steps

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --mem=160GB
#SBATCH --partition=gpu

### ADDITIONAL RUN INFO ###
#SBATCH --array=0

### LOG INFO ###
#SBATCH --job-name=ebt-symb-xxs-bs_256_s2_lr_
#SBATCH --output=logs/slurm/mus/ebt-symb-xxs-bs_256_s2_lr_%A-%a.log
export RUN_NAME="ebt-symb-xxs-bs_256_s2_lr_"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export STAGE1_CKPT_PATH="path/to/stage1/checkpoint.ckpt"  # MODIFY THIS PATH
export MODEL_SIZE="xxs"
mkdir -p logs/slurm/mus/

module purge
eval "$(conda shell.bash hook)"
conda activate music_EBT

export PYTHONPATH="/home/rebcecca/music-EBT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd /home/rebcecca/music-EBT || exit 1

lr=(0.0006)
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
--mcmc_num_steps 4 \
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
--warm_up_steps 5000 \
\
--dataset_name "giga_midi" \
--num_workers 12 \
--validation_split_pct 0.01 \
--val_check_interval 5000 \
\
--checkpoint_path "${STAGE1_CKPT_PATH}" \
\
--wandb_project 'mus_symb_ebt_s2_pretrain' \
\
--log_model_archi \
--log_gradients \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}

# NOTES:
# - Set STAGE1_CKPT_PATH to the checkpoint from stage 1
# - Increased --mcmc_num_steps from 2 to 4 for better refinement
# - Reduced learning rate (0.0006 vs 0.001) for fine-tuning
# - Reduced warm_up_steps since we're continuing from a checkpoint
