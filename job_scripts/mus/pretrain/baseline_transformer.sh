#!/bin/bash
### Baseline Transformer Music - Pretraining Script
### Standard transformer baseline for comparison with EBT models
### This is a quick 2-hour test run to validate training pipeline

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --mem=80GB
#SBATCH --partition=mit_normal_gpu

### ADDITIONAL RUN INFO ###
#SBATCH --array=0

### LOG INFO ###
#SBATCH --job-name=baseline-symb-xxs-test-2h
#SBATCH --output=logs/slurm/mus/baseline-symb-xxs-test-2h_%A-%a.log
export RUN_NAME="baseline-symb-xxs-test-2h"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_SIZE="xxs"
mkdir -p logs/slurm/mus/

module purge
eval "$(conda shell.bash hook)"
conda activate music_EBT

export PYTHONPATH="/home/rebcecca/music-EBT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd /home/rebcecca/music-EBT || exit 1

lr=(0.0006)

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "MUS_SYMB" \
--model_name "baseline_transformer" \
--model_size ${MODEL_SIZE} \
\
--tokenizer_type "REMI" \
--normalize_initial_condition \
--context_length 512 \
\
--gpus "1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 32 \
--accumulate_grad_batches 2 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 500 \
--max_scheduling_steps 500 \
--warm_up_steps 50 \
\
--dataset_name "giga-midi" \
--num_workers 12 \
--validation_split_pct 0.01 \
--val_check_interval 200 \
--limit_train_batches 1 \
--limit_val_batches 1 \
--limit_test_batches 1 \
\
--wandb_project 'mus_symb_baseline_pretrain' \
\
--log_model_archi \
--log_gradients \
--log_every_n_steps 10 \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}

# NOTES:
# - Quick 2-hour test run with 500 steps
# - Standard transformer baseline (no MCMC/energy-based training)
# - Validation every 200 steps with 10 batches to ensure training stability
# - Monitor loss/perplexity to ensure training is working
# - After validation, extend max_steps for longer training runs
