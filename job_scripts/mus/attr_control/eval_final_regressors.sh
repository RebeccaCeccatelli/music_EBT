#!/bin/bash
#SBATCH --job-name=eval-final-regressors
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm/eval_final_regressors_%j.log
# Held-out MAE / Pearson r of every regressor used in the thesis (CPU only).
cd /home/rebcecca/music-EBT
export PYTHONPATH="$PWD:$PWD/data/mus/symbolic:$PYTHONPATH"
D=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/attr_control
OUT=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/attr_control/eval_final_regressors_${SLURM_JOB_ID}.json
~/.conda/envs/music_EBT/bin/python attribute_control/eval_final_regressors.py "$OUT" \
  $D/velocity_regressor_remi_20261005_110339_24934690/best.pt \
  $D/duration_regressor_remi_20261005_111449_24934691/best.pt \
  $D/pitch_register_regressor_remi_20261005_111449_24934692/best.pt \
  $D/duration_regressor_ant-at-full_20261005_112459_24934693/best.pt \
  $D/pitch_register_regressor_ant-at-full_20261005_112459_24934694/best.pt \
  $D/velocity_regressor_llama-ebtrecipe-remi_20261019_062104_26348315/best.pt \
  $D/duration_regressor_llama-ebtrecipe-remi_20261019_062104_26348316/best.pt \
  $D/pitch_register_regressor_llama-ebtrecipe-remi_20261019_063203_26348317/best.pt \
  $D/duration_regressor_llama-ebtrecipe-ant-at-full_20261104_200249_27933069/best.pt \
  $D/pitch_register_regressor_llama-ebtrecipe-ant-at-full_20261104_185130_27917682/best.pt
