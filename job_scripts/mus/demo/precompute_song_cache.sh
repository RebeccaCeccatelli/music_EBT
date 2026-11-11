#!/bin/bash
#SBATCH --job-name=demo-songcache
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rebcecca/orcd/scratch/rebcecca/music_EBT_logs/slurm_%j.out
# Fill the interactive demo's song cache (full-song MP3 + roll, time map, default preview)
# for every named song x tokenizer: demo/precompute_song_cache.py. Resumable.
cd /home/rebcecca/music-EBT
export PYTHONPATH=$PWD:$PWD/data/mus/symbolic
export SONG_CACHE_TMP=${TMPDIR:-/tmp}
~/.conda/envs/music_EBT_demo/bin/python demo/precompute_song_cache.py --workers 16
