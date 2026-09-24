#!/bin/bash
### EBT Symbolic Music - Stage 1 Pretraining Script
### Trains EBT on tokenized MIDI using MCMC-style iterative refinement.

### SLURM CONFIGURATION ###
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --signal=TERM@120
#SBATCH --output=./logs/slurm_%j.out
# mit_preemptable's GraceTime=0 (confirmed via `scontrol show partition`) means
# a preempted job gets killed with NO warning — --signal=TERM@120 only governs
# time-limit warnings, not preemption grace, which is this separate,
# partition-level setting. So on preemption, our own SIGTERM trap below never
# runs (no signal ever reaches it), and native SLURM requeue (Requeue=1 by
# partition default) is the ONLY mechanism that reliably restarts the job —
# it operates at the controller level and doesn't need a graceful process
# exit. We used to disable it (--no-requeue) to stop it double-firing
# alongside our own resubmit logic on time-limit events (which DO get the
# full signal grace period); now _do_resubmit() itself checks for an
# already-requeued instance of this exact job ID before submitting a new one,
# so both mechanisms can coexist safely.

### ADDITIONAL RUN INFO ###
#SBATCH --array=0

export MODEL_SIZE="small"
lr=(0.0008)

### Project Root Discovery ###
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
if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "❌ Error: Could not find project root."
    exit 1
fi

SCRATCH_LOGS_DIR="${HOME}/orcd/scratch/rebcecca/music_EBT_logs"

export PYTHONPATH="${PROJECT_ROOT}:${HOME}/music-EBT/data/mus/symbolic:$PYTHONPATH"
export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${PROJECT_ROOT}" || exit 1

# Parse command-line arguments (overridable via sbatch).
# Supported tokenizer types:
#   REMI                      - miditok REMI (default)
#   Anticipation-Arrival-Time - AMT paper format, full vocab 55028
#   Anticipation-Vanilla      - arrival-time, no control block, smaller vocab
DATASET_NAME="giga_midi"
TOKENIZER_TYPE="REMI"
MODEL_SIZE="small"
RESUME_CKPT=""
FRESH_START=""
BATCH_SIZE=""
ACCUM_STEPS=""
VAL_CHECK_INTERVAL=""
LIMIT_VAL_BATCHES=""
MCMC_STEP_SIZE_LR_MULT=""
MCMC_STEP_SIZE_MAX=""
PEAK_LR=""
# EBT-paper-suggested stabilization techniques (Section 3.3): off by default,
# matching both this codebase's and the authors' own reference implementation's
# defaults — must be explicitly opted into via these flags. Threaded through
# both the initial python invocation AND _do_resubmit() below so they survive
# a manual wall-time resubmit, not just a native SLURM requeue (which replays
# the original sbatch command line regardless).
LANGEVIN_NOISE=""
RANDOMIZE_STEP_SCALE=""
RANDOMIZE_NUM_STEPS=""
REPLAY_BUFFER=""
REPLAY_BUFFER_SIZE=""
# Empty by default (zero behavior change for existing runs) — set this to
# keep a new run's base name distinct from an existing lineage sharing the
# same model/tokenizer/size, so this script's own duplicate-active-job check
# in _do_resubmit() can't mistake the two for each other and skip resubmitting
# one of them after a preemption.
RUN_NAME_SUFFIX=""
# Defaults to 1 (zero behavior change for existing single-GPU runs). Set via
# --sbatch_gpus N to train DDP data-parallel across N GPUs on one node —
# train_model.py already supports this natively (--distributed_strategy ddp),
# only this launch script hardcoded --gpus 1. Threaded through both the
# initial submission AND every self-resubmit below (as an explicit `sbatch
# --gpus=/--ntasks-per-node=` override, since #SBATCH directives in this file
# are this job's PERMANENT default and are shared by every lineage that
# launches from it — REMI's and the non-stabilized Anticipation run's own
# resubmits must keep defaulting to 1 GPU unless THEY are also explicitly
# given --sbatch_gpus).
SBATCH_GPUS="1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)           DATASET_NAME="$2";       shift 2 ;;
        --run_name_suffix)        RUN_NAME_SUFFIX="$2";    shift 2 ;;
        --tokenizer_type)         TOKENIZER_TYPE="$2";     shift 2 ;;
        --model_size)             MODEL_SIZE="$2";         shift 2 ;;
        --resume_training_ckpt)   RESUME_CKPT="$2";        shift 2 ;;
        --fresh_start)            FRESH_START="1";          shift ;;
        --batch_size_per_device)  BATCH_SIZE="$2";         shift 2 ;;
        --accumulate_grad_batches) ACCUM_STEPS="$2";       shift 2 ;;
        --val_check_interval)     VAL_CHECK_INTERVAL="$2"; shift 2 ;;
        --limit_val_batches)      LIMIT_VAL_BATCHES="$2";  shift 2 ;;
        --peak_learning_rate)     PEAK_LR="$2";            shift 2 ;;
        --mcmc_step_size_lr_multiplier) MCMC_STEP_SIZE_LR_MULT="$2"; shift 2 ;;
        --mcmc_step_size_max)     MCMC_STEP_SIZE_MAX="$2"; shift 2 ;;
        --langevin_dynamics_noise) LANGEVIN_NOISE="$2";    shift 2 ;;
        --randomize_mcmc_step_size_scale) RANDOMIZE_STEP_SCALE="$2"; shift 2 ;;
        --randomize_mcmc_num_steps) RANDOMIZE_NUM_STEPS="$2"; shift 2 ;;
        --mcmc_replay_buffer)     REPLAY_BUFFER="1";       shift ;;
        --mcmc_replay_buffer_size) REPLAY_BUFFER_SIZE="$2"; shift 2 ;;
        --sbatch_gpus)            SBATCH_GPUS="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Tokenizer-aware defaults.
# Anticipation vocab (55028) is ~120x larger than REMI (~452);
# use smaller batch and more frequent checkpoints.
# Also use lower MCMC step size learning rate for Anticipation to prevent divergence.
# mcmc_step_size_max: upper clamp on the learnable MCMC step size (alpha).
# Added after observing unbounded growth (1.76 -> 2.05 -> 2.16+, still climbing)
# coincide with a sustained valid_loss regression (0.60 -> 0.76) on a REMI run —
# alpha's LR multiplier had no ceiling to check it against. Applies to both
# branches as a general safeguard.
case "${TOKENIZER_TYPE}" in
    Anticipation-*)
        BATCH_SIZE="${BATCH_SIZE:-4}"
        ACCUM_STEPS="${ACCUM_STEPS:-16}"
        VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-100}"
        LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1072}"
        MCMC_STEP_SIZE_LR_MULT="${MCMC_STEP_SIZE_LR_MULT:-2}"
        MCMC_STEP_SIZE_MAX="${MCMC_STEP_SIZE_MAX:-2.0}"
        ;;
    *)
        BATCH_SIZE="${BATCH_SIZE:-4}"
        ACCUM_STEPS="${ACCUM_STEPS:-64}"
        VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-100}"
        LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
        # Was 100 — a 100x-amplified LR with no ceiling let alpha run away
        # once past warmup; lowered alongside adding the ceiling above.
        MCMC_STEP_SIZE_LR_MULT="${MCMC_STEP_SIZE_LR_MULT:-20}"
        MCMC_STEP_SIZE_MAX="${MCMC_STEP_SIZE_MAX:-2.0}"
        ;;
esac

case "${TOKENIZER_TYPE}" in
    REMI)                       TOK_SLUG="remi" ;;
    Anticipation-Arrival-Time)  TOK_SLUG="ant-at-full" ;;
    Anticipation-Vanilla)       TOK_SLUG="ant-at-ar" ;;
    Anticipation-Interarrival)  TOK_SLUG="ant-ia-full" ;;
    *)                          TOK_SLUG=$(echo "${TOKENIZER_TYPE}" | tr '[:upper:]' '[:lower:]') ;;
esac

# Use provided peak_learning_rate or default from array
if [[ -z "${PEAK_LR}" ]]; then
    PEAK_LR="${lr[${SLURM_ARRAY_TASK_ID}]}"
fi
BASE_RUN_NAME="ebt-symb-${MODEL_SIZE}-${TOK_SLUG}-s1${RUN_NAME_SUFFIX}"
FULL_RUN_NAME="${BASE_RUN_NAME}-job${SLURM_JOB_ID:-local}"
scontrol update JobId="${SLURM_JOB_ID}" Name="${FULL_RUN_NAME}" 2>/dev/null || true
MAX_STEPS=100000

# Auto-resume from the highest-step checkpoint of any previous run with the
# same base name — selected by the step number embedded in the checkpoint
# filename itself, NOT by directory modification time. A prior mtime-based
# version (`ls -td ... | head -1`) could pick an older, regressed checkpoint
# whenever a less-progressed rerun happened to touch its directory more
# recently than the true best run — this is exactly what caused this job to
# silently reset to an epoch-19 checkpoint after a preemption+requeue cycle.
#
# The base-name glob matches ANY run sharing this model/tokenizer/size,
# which can span multiple hyperparameter-incompatible lineages over time
# (e.g. a pre-stabilization run with no Langevin noise/replay buffer, and a
# post-stabilization run with both) — discovered when this exact glob
# resolved to a different job's pre-stabilization checkpoint while this run
# was mid-training with the stabilization flags on. Blindly resuming into
# the highest step regardless of which hparams produced it would silently
# splice incompatible weights into the continuation. find_compatible_ckpt.py
# checks each candidate's own saved hparams (starting from the highest step)
# and only returns one that actually matches this run's own flags; finding
# none is treated the same as "no checkpoint found" below, not a fallback to
# an incompatible one.
if [[ -z "${RESUME_CKPT}" && -z "${FRESH_START}" ]]; then
    BEST_CKPT=$(python "${PROJECT_ROOT}/job_scripts/mus/pretrain/find_compatible_ckpt.py" \
        --pattern "${SCRATCH_LOGS_DIR}/checkpoints/${BASE_RUN_NAME}*/epoch=*.ckpt" \
        --langevin_dynamics_noise "${LANGEVIN_NOISE}" \
        --randomize_mcmc_step_size_scale "${RANDOMIZE_STEP_SCALE}" \
        --randomize_mcmc_num_steps "${RANDOMIZE_NUM_STEPS}" \
        --mcmc_replay_buffer "${REPLAY_BUFFER}" \
        --mcmc_replay_buffer_size "${REPLAY_BUFFER_SIZE}")
    if [[ -n "${BEST_CKPT}" ]]; then
        RESUME_CKPT="${BEST_CKPT}"
        echo "Auto-resuming from highest-step COMPATIBLE checkpoint: ${RESUME_CKPT}"
    fi
fi

PYTHON_PID=0
SIGTERM_RECEIVED=0
_handle_sigterm() {
    SIGTERM_RECEIVED=1
    echo "[$(date -Iseconds)] SIGTERM received — forwarding as SIGUSR1 to Python (PID ${PYTHON_PID}) for PL auto-requeue..."
    # PL bypasses SIGTERM in SLURM auto-requeue mode; USR1 triggers its checkpoint-and-requeue handler.
    [[ ${PYTHON_PID} -ne 0 ]] && kill -USR1 "${PYTHON_PID}" 2>/dev/null
}
trap '_handle_sigterm' TERM

python train_model.py \
--run_name "${FULL_RUN_NAME}" \
--modality "MUS_SYMB" \
--model_name "ebt" \
--model_size "${MODEL_SIZE}" \
--tokenizer_type "${TOKENIZER_TYPE}" \
--tokenizer_config_path "/home/rebcecca/orcd/pool/music_datasets/giga-midi/tokens/miditok/tokenizer.json" \
\
--normalize_initial_condition \
--ebt_type "time_embed" \
--denoising_initial_condition "random_noise" \
--mcmc_step_size_learnable \
--mcmc_step_size 1 \
--mcmc_step_size_lr_multiplier "${MCMC_STEP_SIZE_LR_MULT}" \
--mcmc_step_size_max "${MCMC_STEP_SIZE_MAX}" \
--mcmc_num_steps 2 \
--clamp_futures_grad \
\
--context_length 512 \
--gpus "${SBATCH_GPUS}" \
\
--peak_learning_rate "${PEAK_LR}" \
--batch_size_per_device "${BATCH_SIZE}" \
--accumulate_grad_batches "${ACCUM_STEPS}" \
--gradient_clip_val 1.0 \
--weight_decay 0.05 \
--min_lr_scale 10 \
--max_steps "${MAX_STEPS}" \
--max_scheduling_steps "${MAX_STEPS}" \
--warm_up_steps 10000 \
\
--dataset_name "${DATASET_NAME}" \
--num_workers 12 \
--validation_split_pct 0.05 \
--limit_train_batches 1.0 \
--limit_val_batches "${LIMIT_VAL_BATCHES}" \
--val_check_interval "${VAL_CHECK_INTERVAL}" \
\
--wandb_project 'mus_symb_ebt_pretrain' \
--log_model_archi \
--log_gradients \
--log_every_n_steps 200 \
--set_matmul_precision "medium" \
--wandb_watch \
${LANGEVIN_NOISE:+--langevin_dynamics_noise "${LANGEVIN_NOISE}"} \
${RANDOMIZE_STEP_SCALE:+--randomize_mcmc_step_size_scale "${RANDOMIZE_STEP_SCALE}"} \
${RANDOMIZE_NUM_STEPS:+--randomize_mcmc_num_steps "${RANDOMIZE_NUM_STEPS}"} \
${REPLAY_BUFFER:+--mcmc_replay_buffer} \
${REPLAY_BUFFER_SIZE:+--mcmc_replay_buffer_size "${REPLAY_BUFFER_SIZE}"} \
${RESUME_CKPT:+--resume_training_ckpt "${RESUME_CKPT}"} \
${SLURM_ARRAY_TASK_ID:+--is_slurm_run} &
PYTHON_PID=$!
wait "${PYTHON_PID}"
TRAIN_EXIT_CODE=$?
if [[ ${SIGTERM_RECEIVED} -eq 1 ]]; then
    wait "${PYTHON_PID}" 2>/dev/null
    TRAIN_EXIT_CODE=$?
fi

_do_resubmit() {
    # Fail SAFE, not open: if squeue itself fails (transient controller issue —
    # observed happening specifically during a job's own shutdown), don't
    # silently treat that as "no duplicates" and resubmit anyway.
    local squeue_output
    if ! squeue_output=$(squeue -u "$USER" -h -o "%i %j %t" 2>&1); then
        echo "Could not query squeue to check for duplicates — skipping resubmit to be safe."
        return 0
    fi
    # Native requeue (see --array comment above) may have already put this
    # exact job ID back in the queue as PENDING by the time we get here —
    # if so, don't also sbatch a separate new one.
    if echo "${squeue_output}" | grep -qE "^${SLURM_JOB_ID}(_[0-9]+)? .* PD$"; then
        echo "Already natively requeued as ${SLURM_JOB_ID} (state PD) — skipping manual resubmit."
        return 0
    fi
    local n_active
    # Exclude the current job so it doesn't count itself; matches both plain
    # ("21008484") and array-task ("21008484_0") squeue job-id formats.
    n_active=$(echo "${squeue_output}" \
        | grep "${BASE_RUN_NAME}" | grep -v -E "^${SLURM_JOB_ID}(_[0-9]+)? " | wc -l)
    if [[ "${n_active}" -gt 0 ]]; then
        echo "Skipping resubmit: ${n_active} other instance(s) of ${BASE_RUN_NAME} already in queue/running."
        return 0
    fi
    sbatch --gpus="${SBATCH_GPUS}" --ntasks-per-node="${SBATCH_GPUS}" "${BASH_SOURCE[0]}" \
        --tokenizer_type "${TOKENIZER_TYPE}" \
        --sbatch_gpus "${SBATCH_GPUS}" \
        --dataset_name "${DATASET_NAME}" \
        --model_size "${MODEL_SIZE}" \
        --peak_learning_rate "${PEAK_LR}" \
        --batch_size_per_device "${BATCH_SIZE}" \
        --accumulate_grad_batches "${ACCUM_STEPS}" \
        --val_check_interval "${VAL_CHECK_INTERVAL}" \
        --limit_val_batches "${LIMIT_VAL_BATCHES}" \
        --mcmc_step_size_lr_multiplier "${MCMC_STEP_SIZE_LR_MULT}" \
        --mcmc_step_size_max "${MCMC_STEP_SIZE_MAX}" \
        ${LANGEVIN_NOISE:+--langevin_dynamics_noise "${LANGEVIN_NOISE}"} \
        ${RANDOMIZE_STEP_SCALE:+--randomize_mcmc_step_size_scale "${RANDOMIZE_STEP_SCALE}"} \
        ${RANDOMIZE_NUM_STEPS:+--randomize_mcmc_num_steps "${RANDOMIZE_NUM_STEPS}"} \
        ${REPLAY_BUFFER:+--mcmc_replay_buffer} \
        ${REPLAY_BUFFER_SIZE:+--mcmc_replay_buffer_size "${REPLAY_BUFFER_SIZE}"} \
        ${RUN_NAME_SUFFIX:+--run_name_suffix "${RUN_NAME_SUFFIX}"}
}

# Exit 0:   clean exit — either training finished or PL saved a checkpoint on SIGTERM.
#           PL does NOT auto-requeue on this cluster, so we check the highest saved
#           step to decide whether to resubmit.
# Exit 143: SIGTERM reached bash before PL could handle it — resubmit unconditionally.
# Anything else: crash (OOM, Python error, etc.) — do not resubmit.
if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    LAST_STEP=$(ls "${SCRATCH_LOGS_DIR}/checkpoints/${BASE_RUN_NAME}"*/epoch=*.ckpt 2>/dev/null \
        | grep -oE "step=step=[0-9]+" | grep -oE "[0-9]+$" | sort -n | tail -1)
    # Checkpoints only save on periodic validation-interval boundaries, so
    # max_steps often doesn't land exactly on one — the highest SAVED
    # checkpoint can sit permanently just below MAX_STEPS even after training
    # genuinely finished, causing an infinite "incomplete, resubmit" loop that
    # just redoes the same final stretch forever (confirmed happening on the
    # baseline scripts: PL's own "Trainer.fit stopped: max_steps=... reached."
    # message was printing every cycle while this check kept concluding
    # "incomplete"). Treat that message in this job's own log as authoritative
    # completion too.
    if [[ -n "${LAST_STEP}" && "${LAST_STEP}" -ge "${MAX_STEPS}" ]] \
        || grep -qE "max_steps=${MAX_STEPS}.*reached" "./logs/slurm_${SLURM_JOB_ID}.out" 2>/dev/null; then
        echo "Training complete (last checkpoint step: ${LAST_STEP:-n/a}, target: ${MAX_STEPS}). Not resubmitting."
    else
        echo "Clean exit but training incomplete (last step: ${LAST_STEP:-none}). Resubmitting..."
        _do_resubmit
    fi
elif [[ ${TRAIN_EXIT_CODE} -eq 143 ]]; then
    echo "Job killed by SIGTERM before PL could handle it. Resubmitting..."
    _do_resubmit
else
    echo "Training failed with exit code ${TRAIN_EXIT_CODE}. Not resubmitting."
    exit ${TRAIN_EXIT_CODE}
fi
