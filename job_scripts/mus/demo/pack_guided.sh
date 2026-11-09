#!/bin/bash
#SBATCH --job-name=survey-guided-pack
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm_%j.out
# Pack the guided picker data (demo/survey/guided_candidates.py pack) after the render job.
cd /home/rebcecca/music-EBT
for tok in remi ant; do
  ~/.conda/envs/music_EBT/bin/python demo/survey/guided_candidates.py pack --tok $tok --out /home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/survey/guided_pickers/$tok
done
