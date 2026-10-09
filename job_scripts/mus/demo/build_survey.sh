#!/bin/bash
# Render the blind listening survey (demo/survey/build_survey.py build) off the login node.
#SBATCH --job-name=build-survey
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm_%j.out
cd /home/rebcecca/music-EBT
~/.conda/envs/music_EBT/bin/python demo/survey/build_survey.py build ${FORCE:+--force}
