#!/bin/bash
### Controllability vs music quality on the 16 TEST prompts, with the EBT-recipe baselines
### (ctx 512, EBT's batch/steps; diary 2026-10-19) next to EBT and the 100k-step baselines.
### Same EBT tables / reference sets as the 2026-10-09 final scoring (job 25380029); the
### retrained baselines are labelled <model>-ebtrecipe so they don't merge with the 100k ones.
### Output: music_quality/sweeps/final_{remi,ant_ar}_ebtrecipe.json
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --output=./logs/slurm_%j.out
set -euo pipefail
cd "${HOME}/music-EBT"
export PYTHONPATH="${HOME}/music-EBT" PYTHONUNBUFFERED=1
PY="${HOME}/.conda/envs/music_EBT/bin/python"
L="${HOME}/orcd/scratch/rebcecca/music_EBT_logs"
MQ="${L}/music_quality"; AR="${L}/attr_control/ar_guidance"; LM="${L}/attr_control/listen_midi"
ar() { echo "${AR}/$1/*.table.json::${AR}/$1/midi${2:+::$2}"; }

"${PY}" eval/score_sweep_quality.py \
    --sweep "${MQ}/sweeps/ebt_remi_tables/velocity_remi.table.json::${LM}/25280946" \
    --sweep "${MQ}/sweeps/ebt_remi_tables/duration_remi.table.json::${LM}/25280947" \
    --sweep "${MQ}/sweeps/ebt_remi_tables/pitch_register_remi.table.json::${LM}/25280948" \
    --sweep "$(ar gpt2_tilt_20261005_153523_24940378)" \
    --sweep "$(ar gpt2_best_of_n_20261005_154305_24940380)" \
    --sweep "$(ar llama_tilt_20261005_150439_24940374)" \
    --sweep "$(ar llama_best_of_n_20261005_153108_24940376)" \
    --sweep "$(ar llama_pplm_20261008_070421_25261493)" \
    --sweep "$(ar gpt2-ebtrecipe_tilt_20261019 gpt2-ebtrecipe)" \
    --sweep "$(ar gpt2-ebtrecipe_best_of_n_20261019 gpt2-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_tilt_20261019 llama-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_best_of_n_20261019 llama-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_pplm_20261104 llama-ebtrecipe)" \
    --ref "${MQ}/reference_remi_256tok/scores.csv" \
    --out "${MQ}/sweeps/final_remi_ebtrecipe.json"
echo ======
"${PY}" eval/score_sweep_quality.py \
    --sweep "${MQ}/sweeps/ebt_antar_tables/duration_ant-ar-coarse.table.json::${LM}/25296307" \
    --sweep "${MQ}/sweeps/ebt_antar_tables/pitch_register_ant-ar-coarse.table.json::${LM}/25296308" \
    --sweep "${MQ}/sweeps/ebt_antar_tables/duration_ant-ar-fine.table.json::${LM}/25296410" \
    --sweep "${MQ}/sweeps/ebt_antar_tables/pitch_register_ant-ar-fine.table.json::${LM}/25296411" \
    --sweep "$(ar gpt2_best_of_n_20261008_154914_25296309)" \
    --sweep "$(ar llama_best_of_n_20261008_155455_25296310)" \
    --sweep "$(ar llama_pplm_20261009_113147_25404826)" \
    --sweep "$(ar gpt2-ebtrecipe_best_of_n_ant_20261019 gpt2-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_best_of_n_ant_20261019 llama-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_pplm_ant_20261105 llama-ebtrecipe)" \
    --ref "${MQ}/reference_ant-at-full-ar_256tok/scores.csv" \
    --out "${MQ}/sweeps/final_ant_ar_ebtrecipe.json"
echo "✅ done"
