### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4 

### LOG INFO ###
#SBATCH --job-name=ebt-xxs-time_embed_2_mcmc_lr_alpha_
#SBATCH --output=logs/slurm/vid/ebt-xxs-time_embed_2_mcmc_lr_alpha_%A-%a.log
export RUN_NAME="ebt-xxs-time_embed_2_mcmc_lr_alpha_"
# NOTE ctrl d ALL THREE of above to modify job-name, output, and RUN_NAME (which should all be the same)
export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="${RUN_NAME#*-}"; export MODEL_SIZE="${MODEL_SIZE%%-*}"
mkdir -p logs/slurm/vid/
module purge

lr=(0.0001)
alpha=(30000 30000)
alpha_lr=(90000 90000)

python train_model.py \
--run_name ${RUN_NAME}${lr[${SLURM_ARRAY_TASK_ID}]}_${alpha[${SLURM_ARRAY_TASK_ID}]} \
--modality "VID" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size ${alpha[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_step_size_lr_multiplier ${alpha_lr[${SLURM_ARRAY_TASK_ID}]} \
--mcmc_num_steps 2 \
\
--backbone_type "vae" \
--vae_normalization \
--time_between_frames 0.25 \
\
--context_length 16 \
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
--max_steps 30000 \
--max_scheduling_steps 1000000 \
--warm_up_steps 10000 \
\
--dataset_name "something" \
--dataset_dir ${SSV2_DIR} \
--num_workers 12 \
--image_dims 224 224 \
\
--wandb_project 'debug' \
\
--log_model_archi \
--log_gradients \
--log_every_n_steps 20 \
\
--set_matmul_precision "medium" \
--wandb_watch \
--is_slurm_run