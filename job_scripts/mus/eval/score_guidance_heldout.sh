#!/bin/bash
### Held-out operating points (diary 2026-10-19 "do it properly"):
###   1. score the TUNE sweeps (16 tune prompts per tokenizer, disjoint from the test prompts)
###      → per (system, attribute) operating point = best accuracy within the quality budget;
###   2. score the TEST sweeps (score_guidance_ebtrecipe.sh);
###   3. paired tests on the TEST prompts at the TUNED operating points, EBT vs the EBT-recipe
###      baselines (the 100k baselines have no tune sweeps), Holm over REMI + Anticipation.
### Outputs: music_quality/sweeps/tune_{remi,ant_ar}.json, final_*_ebtrecipe.json,
###          heldout_paired_{remi,ant_ar}.json, heldout_paired_holm.json
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=03:00:00
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --output=./logs/slurm_%j.out
set -euo pipefail
cd "${HOME}/music-EBT"
export PYTHONPATH="${HOME}/music-EBT" PYTHONUNBUFFERED=1
PY="${HOME}/.conda/envs/music_EBT/bin/python"
L="${HOME}/orcd/scratch/rebcecca/music_EBT_logs"
MQ="${L}/music_quality"; SW="${MQ}/sweeps"; AR="${L}/attr_control/ar_guidance"; LM="${L}/attr_control/listen_midi"
TT="${SW}/ebt_tune_tables"
ar() { echo "${AR}/$1/*.table.json::${AR}/$1/midi${2:+::$2}"; }

echo "### 1. tune sweeps"
"${PY}" eval/score_sweep_quality.py \
    --sweep "${TT}/velocity_remi.table.json::${LM}/tune-ebt-remi-velocity" \
    --sweep "${TT}/duration_remi.table.json::${LM}/tune-ebt-remi-duration" \
    --sweep "${TT}/pitch_register_remi.table.json::${LM}/tune-ebt-remi-pitch_register" \
    --sweep "$(ar tune_gpt2-ebtrecipe_tilt_remi_20261019 gpt2-ebtrecipe)" \
    --sweep "$(ar tune_gpt2-ebtrecipe_best_of_n_remi_20261019 gpt2-ebtrecipe)" \
    --sweep "$(ar tune_llama-ebtrecipe_tilt_remi_20261019 llama-ebtrecipe)" \
    --sweep "$(ar tune_llama-ebtrecipe_best_of_n_remi_20261019 llama-ebtrecipe)" \
    --sweep "$(ar tune_llama-ebtrecipe_pplm_remi_20261104 llama-ebtrecipe)" \
    --ref "${MQ}/reference_remi_256tok/scores.csv" \
    --out "${SW}/tune_remi.json"
"${PY}" eval/score_sweep_quality.py \
    --sweep "${TT}/duration_ant-ar-coarse.table.json::${LM}/tune-ebt-ant-duration-coarse" \
    --sweep "${TT}/duration_ant-ar-fine.table.json::${LM}/tune-ebt-ant-duration-fine" \
    --sweep "${TT}/pitch_register_ant-ar-coarse.table.json::${LM}/tune-ebt-ant-pitch_register-coarse" \
    --sweep "${TT}/pitch_register_ant-ar-fine.table.json::${LM}/tune-ebt-ant-pitch_register-fine" \
    --sweep "$(ar tune_gpt2-ebtrecipe_best_of_n_ant_20261019 gpt2-ebtrecipe)" \
    --sweep "$(ar tune_llama-ebtrecipe_best_of_n_ant_20261019 llama-ebtrecipe)" \
    --sweep "$(ar tune_llama-ebtrecipe_pplm_ant_20261105 llama-ebtrecipe)" \
    --ref "${MQ}/reference_ant-at-full-ar_256tok/scores.csv" \
    --out "${SW}/tune_ant_ar.json"

echo "### 2. test sweeps"
bash job_scripts/mus/eval/score_guidance_ebtrecipe.sh

echo "### 3. paired tests on the test prompts at the tuned operating points"
"${PY}" eval/paired_guidance_stats.py --results "${SW}/tune_remi.json" \
    --sweep "${SW}/ebt_remi_tables/*_remi.table.json" \
    --sweep "$(ar gpt2-ebtrecipe_tilt_20261019 gpt2-ebtrecipe)" \
    --sweep "$(ar gpt2-ebtrecipe_best_of_n_20261019 gpt2-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_tilt_20261019 llama-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_best_of_n_20261019 llama-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_pplm_20261104 llama-ebtrecipe)" \
    --out "${SW}/heldout_paired_remi.json"
"${PY}" eval/paired_guidance_stats.py --results "${SW}/tune_ant_ar.json" \
    --sweep "${SW}/ebt_antar_tables/*_ant-ar-*.table.json" \
    --sweep "$(ar gpt2-ebtrecipe_best_of_n_ant_20261019 gpt2-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_best_of_n_ant_20261019 llama-ebtrecipe)" \
    --sweep "$(ar llama-ebtrecipe_pplm_ant_20261105 llama-ebtrecipe)" \
    --out "${SW}/heldout_paired_ant_ar.json"

# Holm over every comparison of both tokenizers (operating points + best-of-4), as in 2026-10-15.
"${PY}" - "${SW}" <<'EOF'
import json, sys
from pathlib import Path
sw = Path(sys.argv[1])
rows = [dict(r, tok=t) for t in ('remi', 'ant_ar') for r in json.loads((sw / f'heldout_paired_{t}.json').read_text())]
rows.sort(key=lambda r: r['p_signflip'])
m, running = len(rows), 0.0
for i, r in enumerate(rows):
    running = max(running, min(1.0, (m - i) * r['p_signflip']))
    r['p_holm'] = running
(sw / 'heldout_paired_holm.json').write_text(json.dumps(rows, indent=2))
print(f"Holm over {m} comparisons")
for r in sorted(rows, key=lambda r: (r['tok'], r['attribute'], r['other'], r['kind'])):
    print(f"  {r['tok']:6s} {r['attribute']:15s} {r['other']:26s} {r['kind']:16s} "
          f"EBT λ {r['ref_strength']:>6s} vs {str(r['other_strength']):>5s}  diff {r['diff']:+.3f} "
          f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  p {r['p_signflip']:.3f}  Holm {r['p_holm']:.3f}")
EOF
echo "✅ done"
