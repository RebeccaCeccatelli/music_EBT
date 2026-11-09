#!/bin/bash
#SBATCH --job-name=build-showcase
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm_%j.out
# Static showcase: render the clips in demo/showcase/selection.json -> site/audio + site/data.json.
cd /home/rebcecca/music-EBT
~/.conda/envs/music_EBT/bin/python demo/showcase/build_showcase.py build ${FORCE:+--force}
