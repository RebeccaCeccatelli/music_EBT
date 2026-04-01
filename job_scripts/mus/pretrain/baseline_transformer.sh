#!/bin/bash
### Baseline Transformer Music - Pretraining Script
### Standard transformer baseline for comparison with EBT models

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
#SBATCH --job-name=baseline-symb-xxs-bs_256_lr_
#SBATCH --output=logs/slurm/mus/baseline-symb-xxs-bs_256_lr_%A-%a.log
export RUN_NAME="baseline-symb-xxs-bs_256_lr_"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/mus/

module purge

lr=(0.001)

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]} \
--modality "MUS_SYMB" \
--model_name "baseline_transformer" \
--model_size ${MODEL_SIZE} \
\
--tokenizer_type "REMI" \
--normalize_initial_condition \
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
# - Standard transformer without energy-based training (no MCMC parameters)
# - Useful as a baseline for comparison with EBT models
# - Can be used for both symbolic (MUS_SYMB) and neural (MUS_NEUR) with appropriate modality change
# - Simpler training loop without iterative refinement
