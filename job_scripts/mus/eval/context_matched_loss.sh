#!/bin/bash
### Validation loss / perplexity of the final EBT and baseline checkpoints on
### identical validation windows at context 512 (+ baselines at 1024).
### See eval/context_matched_loss.py.
###
###   sbatch job_scripts/mus/eval/context_matched_loss.sh
#SBATCH --nodes=1
#SBATCH --gpus=l40s:1  # L40S only: this torch build (2.4+cu121) has no kernels for the RTX Pro 6000 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=60GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_general
#SBATCH --job-name=context-matched-loss
#SBATCH --output=./logs/slurm_%j.out

set -euo pipefail

find_project_root() {
    local dir="$1"
    for ((i=0; i<10; i++)); do
        if [[ -f "${dir}/train_model.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    echo ""
    return 1
}
PROJECT_ROOT="$(find_project_root "$(pwd)")"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${HOME}/music-EBT/data/mus/symbolic:${PYTHONPATH:-}"
export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1

N_WINDOWS="${N_WINDOWS:-1000}"
TOKS="${TOKS:-remi ant}"   # which tokenizers to run
# EBT's MCMC step keeps gradients over (batch × 512 × vocab) logits: with
# Anticipation's 55k vocab, batch 8 runs out of memory on a 44 GB GPU.
REMI_BATCH="${REMI_BATCH:-8}"
ANT_BATCH="${ANT_BATCH:-2}"
B="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/checkpoints"
OUT="${HOME}/orcd/scratch/rebcecca/music_EBT_logs/music_quality/context_matched_loss"
mkdir -p "${OUT}"

# Final checkpoints, as listed in docs/STATUS.md.
one() { ls $1 | head -1; }
# Best-val checkpoint at or before a target step, over all job dirs of a lineage.
best_upto() {
    ls $1 | sed -nE 's/.*step=step=([0-9]+)-valid_loss=valid_loss=([0-9.]+[0-9])\.ckpt$/\1 \2 &/p' \
        | awk -v t="$2" '$1 <= t' | sort -k2,2g | head -1 | cut -d' ' -f3-
}

# SET=antcurve: every Anticipation checkpoint still on disk, on the same 1000 random windows at ctx 512.
# Logged Ant valid_loss is not comparable across runs (unshuffled loader, limit_val_batches=1072: each
# run scores the first 1072 × batch sequences; diary 2026-10-15), so EBT lineages and the EBT-recipe /
# token-matched baselines are compared here instead. EBT 88,800 first (paired reference).
if [[ "${SET:-main}" == antcurve ]]; then
    ck() { ls ${B}/$1/*step=$2-*.ckpt | head -1; }
    MODELS=(
        --model "EBT-s1-88800=$(ck 'ebt-symb-small-ant-at-full-s1-job22827009*' 88800)"
        --model "EBT-s1-99900=$(ck 'ebt-symb-small-ant-at-full-s1-job22827009*' 99900)"
        --model "EBT-stab+-42900=$(ck 'ebt-symb-small-ant-at-full-s1-stab+-job*' 42900)"
        --model "EBT-stab+-44700=$(ck 'ebt-symb-small-ant-at-full-s1-stab+-job*' 44700)"  # 44,800 is corrupted (testzip: data/2)
        --model "EBT-stab-37400=$(ck 'ebt-symb-small-ant-at-full-s1-stab-job*' 37400)"
    )
    for s in 24300 48400 73000 88500; do MODELS+=(--model "Llama-ebtrecipe-$s=$(ck 'baseline-llama-small-ant-at-full-ebtrecipe-job*' $s)"); done
    for s in 25000 42100 59700 76200 87900; do MODELS+=(--model "GPT-2-ebtrecipe-$s=$(ck 'baseline-hf-gpt2-small-ant-at-full-ebtrecipe-job*' $s)"); done
    for s in 3600 10200 16800 22200; do MODELS+=(--model "Llama-tokmatched-$s=$(ck 'baseline-llama-small-ant-at-full-tokmatched-job*' $s)"); done
    printf '%s\n' "${MODELS[@]}"
    python eval/context_matched_loss.py --tokenizer_type Anticipation-Arrival-Time \
        --n_windows "${N_WINDOWS}" --batch_size "${ANT_BATCH}" "${MODELS[@]}" \
        --baseline_long_context 0 --window_len 1024 --per_window \
        --out "${OUT}/ant-at-full-antcurve.json"
    echo "✅ done"; exit 0
fi

# SET=ebtrecipe: the EBT-recipe baselines (ctx 512, EBT's batch/schedule, stopped at EBT's step).
# EBT and the 100k baselines were already run on these same windows (remi.json / ant-at-full.json).
# WHICH=llama|gpt2|"llama gpt2" picks the families. Windows stay drawn at 1024 so they match; no
# 1024-ctx pass (trained at 512: RoPE table is only 512 long).
# FIRST_N=1 (Anticipation only): score the first N_WINDOWS stored sequences in file order, the slice
# the training-time validation uses (EBT 1072×4 = 4288, baselines 1072×8 = 8576), not a random draw.
if [[ "${SET:-main}" == ebtrecipe ]]; then
    WHICH="${WHICH:-llama}"
    for tok in ${TOKS}; do
        case $tok in
            remi) TT=REMI; SLUG=remi; TARGET=33732; BS="${REMI_BATCH}" ;;
            ant)  TT=Anticipation-Arrival-Time; SLUG=ant-at-full; TARGET=88800; BS="${ANT_BATCH}" ;;
        esac
        # EBT first: --per_window compares every other model against the first one (paired bootstrap).
        EBT_CKPT=$(case $tok in remi) one "${B}/ebt-symb-small-remi-s1-*/*step=33732-*.ckpt" ;;
                               ant) one "${B}/ebt-symb-small-ant-at-full-s1-job*/*step=88800-*.ckpt" ;; esac)
        MODELS=(--model "EBT=${EBT_CKPT}")
        [[ " ${WHICH} " == *" llama "* ]] && MODELS+=(--model "Llama-ebtrecipe=$(best_upto "${B}/baseline-llama-small-${SLUG}-ebtrecipe-job*/*.ckpt" ${TARGET})")
        [[ " ${WHICH} " == *" gpt2 "* ]] && MODELS+=(--model "GPT-2-ebtrecipe=$(best_upto "${B}/baseline-hf-gpt2-small-${SLUG}-ebtrecipe-job*/*.ckpt" ${TARGET})")
        printf '%s\n' "${MODELS[@]}"
        python eval/context_matched_loss.py --tokenizer_type "${TT}" \
            --n_windows "${N_WINDOWS}" --batch_size "${BS}" "${MODELS[@]}" \
            --baseline_long_context 0 --window_len 1024 --per_window ${FIRST_N:+--first_n} \
            --out "${OUT}/${SLUG}-ebtrecipe-${WHICH// /-}${FIRST_N:+-first${N_WINDOWS}}.json"
    done
    echo "✅ done"; exit 0
fi
[[ " ${TOKS} " == *" remi "* ]] && python eval/context_matched_loss.py --tokenizer_type REMI \
    --n_windows "${N_WINDOWS}" --batch_size "${REMI_BATCH}" \
    --model "EBT=$(one "${B}/ebt-symb-small-remi-s1-*/*step=33732-*.ckpt")" \
    --model "GPT-2=$(one "${B}/baseline-hf-gpt2-small-remi-*/*step=99660-valid_loss=valid_loss=0.4756.ckpt")" \
    --model "Llama=$(one "${B}/baseline-llama-small-remi-*/*step=99660-valid_loss=valid_loss=0.5239.ckpt")" \
    --out "${OUT}/remi.json"

[[ " ${TOKS} " == *" ant "* ]] && python eval/context_matched_loss.py --tokenizer_type Anticipation-Arrival-Time \
    --n_windows "${N_WINDOWS}" --batch_size "${ANT_BATCH}" \
    --model "EBT=$(one "${B}/ebt-symb-small-ant-at-full-s1-job*/*step=88800-*.ckpt")" \
    --model "GPT-2=$(one "${B}/baseline-hf-gpt2-small-ant-at-full-*/*step=98900-*.ckpt")" \
    --model "Llama=$(one "${B}/baseline-llama-small-ant-at-full-*/*step=100000-*.ckpt")" \
    --out "${OUT}/ant-at-full.json"
echo "✅ done"
