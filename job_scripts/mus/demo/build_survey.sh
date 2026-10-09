#!/bin/bash
# Render the showcase and the blind listening survey off the login node.
# FORCE=1 re-renders every clip (needed after renderer changes).
#SBATCH --job-name=build-survey
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm_%j.out
cd /home/rebcecca/music-EBT
PY=~/.conda/envs/music_EBT/bin/python
$PY demo/showcase/build_showcase.py build ${FORCE:+--force}
$PY demo/survey/build_survey.py build ${FORCE:+--force} --version ${VERSION:-v4}
