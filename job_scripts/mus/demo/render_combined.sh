#!/bin/bash
#SBATCH --job-name=combined-render
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm_%j.out
# Combined-attribute outputs for the Listening Archive (demo/survey/combined_candidates.py):
# render every finished compose run to MP3 (resumable), cache harsh-dissonance scores, pack the page data.
# Usage: sbatch render_combined.sh remi
TOK=${1:-remi}
cd /home/rebcecca/music-EBT
PY=~/.conda/envs/music_EBT/bin/python
$PY demo/survey/combined_candidates.py render --tok $TOK --workers 16 && \
$PY demo/survey/combined_candidates.py pack --tok $TOK \
    --out /home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/survey/combined_pickers/$TOK
