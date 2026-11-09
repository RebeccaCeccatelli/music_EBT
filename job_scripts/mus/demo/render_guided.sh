#!/bin/bash
#SBATCH --job-name=survey-guided
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm_%j.out
# Render every guided clip of the final sweeps (demo/survey/guided_candidates.py) to small MP3s.
# Resumable: clips whose MP3 already exists are skipped.
cd /home/rebcecca/music-EBT
~/.conda/envs/music_EBT/bin/python demo/survey/guided_candidates.py render --workers 16
