#!/bin/bash
#SBATCH --job-name=build-survey-v5
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm_%j.out
# Survey v5 from the author's picks (demo/survey/build_survey_v5.py): render clips, write survey.json + key_v5.json.
cd /home/rebcecca/music-EBT
~/.conda/envs/music_EBT/bin/python demo/survey/build_survey_v5.py build --version ${VERSION:-v5}
