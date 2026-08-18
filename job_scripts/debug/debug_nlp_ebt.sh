### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4 

### LOG INFO ###
#SBATCH --job-name=ebt-xxs-lr_0.0012_bs_128
#SBATCH --output=logs/slurm/nlp/ebt-xxs-lr_0.0012_bs_128%A-%a.log
export RUN_NAME="ebt-xxs-lr_0.0012_bs_128"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/nlp/
module purge

lr=(0.0012)
alpha=(500)
alpha_lr=(1500)

python train_model.py \
--run_name ${RUN_NAME} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--tokenizer "EleutherAI/gpt-neox-20b" \
\
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_step_size_lr_multiplier ${alpha_lr[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_num_steps 2 \
\
--context_length 256 \
\
--gpus "-1" \
\
--peak_learning_rate ${lr[${SLURM_ARRAY_TASK_ID}]} \
--batch_size_per_device 16 \
--accumulate_grad_batches 1 \
--gradient_clip_val 1.0 \
\
--weight_decay 0.01 \
--min_lr_scale 10 \
--max_steps 100 \
--max_scheduling_steps 10000 \
--warm_up_steps 10000 \
\
--dataset_name "pajama" \
--num_workers 12 \
--validation_split_pct 0.000001 \
--val_check_interval 50 \
\
--wandb_project 'debug' \
\
--log_model_archi \
--log_gradients \
\
--set_matmul_precision "medium" \
--wandb_watch \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run}