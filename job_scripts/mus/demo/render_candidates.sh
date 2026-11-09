#!/bin/bash
# Render every candidate survey clip (demo/survey/candidates.py) to small MP3s for the picker page.
#SBATCH --job-name=survey-candidates
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm_%j.out
cd /home/rebcecca/music-EBT
~/.conda/envs/music_EBT/bin/python demo/survey/candidates.py render --workers 4
