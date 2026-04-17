#!/bin/bash

################################################################################
# Auto-Submission Script for GigaMIDI Tokenization with Checkpoint Recovery
#
# This script:
# 0. (Optional) Downloads/Extracts GigaMIDI and runs preprocessing
# 1. Submits SLURM jobs for each split sequentially
# 2. Monitors job status and detects timeouts
# 3. On timeout, checks for checkpoint and auto-resubmits with --resume
# 4. Retries up to N times per split (default: 5)
# 5. Processes splits one at a time to avoid race conditions
#
# Usage:
#   bash auto_submit_gigamidi.sh [OPTIONS]
#
# Examples:
#   bash auto_submit_gigamidi.sh                              # Full pipeline (download + tokenize, non-vanilla)
#   bash auto_submit_gigamidi.sh --vanilla                    # Full pipeline with vanilla tokenizer
#   bash auto_submit_gigamidi.sh --vanilla --splits train     # Download + tokenize only train split
#   bash auto_submit_gigamidi.sh --skip-download --vanilla    # Skip download, only tokenize
#   bash auto_submit_gigamidi.sh --download-only              # Only download/extract, skip tokenization
#   bash auto_submit_gigamidi.sh --max-attempts 10
#
################################################################################

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# CONFIGURATION
# ============================================================================

VANILLA=0
SPLITS=("train" "test" "validation")
MAX_ATTEMPTS=5
POLL_INTERVAL=30
DETACH=0
SKIP_DOWNLOAD=0
DOWNLOAD_ONLY=0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load .env for wandb and other environment variables
ENV_PATH="$PROJECT_ROOT/.env"
if [ -f "$ENV_PATH" ]; then
    export $(grep -v '^#' "$ENV_PATH" | xargs)
    echo "✅ Environment variables loaded from .env"
    if [ -n "${WANDB_API_KEY:-}" ]; then
        echo "✅ WandB API Key exported (Starts with: ${WANDB_API_KEY:0:4}...)"
    fi
else
    echo "⚠️  Warning: .env file not found at $ENV_PATH"
fi

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo -e "\n${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Parse command line arguments
parse_args() {
    local custom_splits=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            --vanilla)
                VANILLA=1
                shift
                ;;
            --splits)
                shift
                while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                    custom_splits+=("$1")
                    shift
                done
                ;;
            --max-attempts)
                MAX_ATTEMPTS="$2"
                shift 2
                ;;
            --poll-interval)
                POLL_INTERVAL="$2"
                shift 2
                ;;
            --detach)
                DETACH=1
                shift
                ;;
            --skip-download)
                SKIP_DOWNLOAD=1
                shift
                ;;
            --download-only)
                DOWNLOAD_ONLY=1
                shift
                ;;
            *)
                print_error "Unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    if [ ${#custom_splits[@]} -gt 0 ]; then
        SPLITS=("${custom_splits[@]}")
    fi
}

usage() {
    cat <<EOF
Usage: bash auto_submit_gigamidi.sh [OPTIONS]

Options:
  --vanilla               Use vanilla tokenizer (default: arrival-time)
  --splits SPLIT1 SPLIT2  Process specific splits (default: train test validation)
  --max-attempts N        Max resume attempts per split (default: 5)
  --poll-interval N       Seconds between status checks (default: 30)
  --detach                Submit job and return immediately (no monitoring)
  --skip-download         Skip download/preprocessing step, only tokenize
  --download-only         Only download/preprocess, skip tokenization
  -h, --help              Show this help message

Examples:
  bash auto_submit_gigamidi.sh                              # Full pipeline (download + tokenize)
  bash auto_submit_gigamidi.sh --vanilla                    # Full pipeline, vanilla tokenizer
  bash auto_submit_gigamidi.sh --vanilla --splits train     # Download + tokenize train split
  bash auto_submit_gigamidi.sh --skip-download --vanilla    # Skip download, only tokenize
  bash auto_submit_gigamidi.sh --download-only              # Only download/extract, skip tokenization
  bash auto_submit_gigamidi.sh --vanilla --detach           # Full pipeline, return immediately
  bash auto_submit_gigamidi.sh --max-attempts 10            # Full pipeline, max 10 attempts per split
EOF
}

# Validate dependencies
check_dependencies() {
    local missing=0
    
    for cmd in sbatch sacct squeue jq python3; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "Command not found: $cmd"
            missing=1
        fi
    done
    
    # Check that tokenize_gigaMIDI.py exists
    if [ -f "$PROJECT_ROOT/data/mus/symbolic/tokenization/anticipation/train/tokenize_gigaMIDI.py" ]; then
        print_success "Found tokenization script"
    else
        print_error "Tokenization script not found"
        missing=1
    fi
    
    if [ $missing -eq 1 ]; then
        print_error "Missing dependencies or scripts"
        return 1
    fi
    
    return 0
}

# Run the preprocessing pipeline as a SLURM job (download + extract + preprocess MIDI → .compound.txt ONLY, no tokenization)
run_download_and_preprocessing() {
    print_header "🚀 STEP 0: Submitting Preprocessing Job (Download, Extract, Preprocess MIDI)"
    
    print_info "Submitting preprocessing as SLURM job (GigaMIDI download/extraction + MIDI→.compound.txt conversion)..."
    
    local job_name="preprocess-gigamidi"
    [ $VANILLA -eq 1 ] && job_name="${job_name}-vanilla"
    
    local export_cmd="export PYTHONPATH='$PROJECT_ROOT/data/mus/symbolic:$PROJECT_ROOT/data/mus/symbolic/tokenization/anticipation:'\$PYTHONPATH && "
    export_cmd="$export_cmd export PYTHONUNBUFFERED=1 && "
    export_cmd="$export_cmd export TQDM_ISATTY=1 && "
    export_cmd="$export_cmd export TQDM_MININTERVAL=0.1 && "
    export_cmd="$export_cmd export WANDB_CONSOLE=wrap_raw && "
    export_cmd="$export_cmd cd '$PROJECT_ROOT'"
    
    if [ -n "${REMOTE_DATA_STORAGE:-}" ]; then
        export_cmd="$export_cmd && export REMOTE_DATA_STORAGE='$REMOTE_DATA_STORAGE'"
    fi
    
    if [ -n "${CUSTOM_STORAGE_PATH:-}" ]; then
        export_cmd="$export_cmd && export REMOTE_DATA_STORAGE='$CUSTOM_STORAGE_PATH'"
    fi
    
    if [ -n "${WANDB_API_KEY:-}" ]; then
        export_cmd="$export_cmd && export WANDB_API_KEY='${WANDB_API_KEY}'"
    fi
    
    if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
        export_cmd="$export_cmd && export HUGGING_FACE_HUB_TOKEN='${HUGGING_FACE_HUB_TOKEN}'"
    fi
    
    # Build preprocessing commands (download + extract + convert MIDI to .compound.txt, NO tokenization)
    # Step 1: Download and extract GigaMIDI
    # Step 2: Run midi_preprocess.py to convert all MIDI files to .compound.txt format
    local preprocess_cmd="python3 -m dataloaders.giga_midi_dataloader && "
    preprocess_cmd="$preprocess_cmd python3 data/mus/symbolic/tokenization/anticipation/train/midi_preprocess.py /home/rebcecca/orcd/pool/music_datasets/giga-midi/midi"
    
    # Use shared log directory for accessible logs
    local log_dir="$HOME/slurm-logs"
    mkdir -p "$log_dir"
    local job_log="$log_dir/${job_name}.log"
    
    print_info "Submitting: sbatch --job-name='$job_name' -c 16 --mem=32G --time=12:00:00 ..."
    print_info "Log: $job_log"
    
    local output=$( \
        sbatch --job-name="$job_name" \
               --export=ALL \
               -c 16 \
               --mem=32G \
               --time=12:00:00 \
               --output="$job_log" \
               --error="$job_log" \
               --wrap="$export_cmd && $preprocess_cmd" \
        2>&1 \
    )
    
    local job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")
    
    if [ -z "$job_id" ]; then
        print_error "Failed to submit preprocessing job"
        echo "$output" >&2
        return 1
    fi
    
    print_success "Preprocessing job submitted with ID: $job_id"
    print_info "Logs: $job_log"
    
    # Monitor preprocessing job
    print_info "Monitoring preprocessing job..."
    local check_count=0
    local max_checks=$((12*60*60 / 30))  # Allow up to 12 hours with 30s polls
    
    while [ $check_count -lt $max_checks ]; do
        check_count=$((check_count + 1))
        
        # Check job status
        local job_status=$(sacct -j "$job_id" -o State --noheader 2>/dev/null | head -1 | xargs)
        
        if [ "$job_status" = "COMPLETED" ]; then
            print_success "Preprocessing job COMPLETED"
            return 0
        elif [ "$job_status" = "FAILED" ]; then
            print_error "Preprocessing job FAILED"
            print_error "Check logs: $job_log"
            return 1
        elif [ "$job_status" = "TIMEOUT" ]; then
            print_error "Preprocessing job TIMEOUT"
            return 1
        elif [ "$job_status" = "CANCELLED" ]; then
            print_error "Preprocessing job CANCELLED"
            return 1
        fi
        
        # Still running - print progress
        if [ $((check_count % 10)) -eq 0 ]; then
            echo -ne "\r   ⏳ Check $check_count: status=$job_status"
        fi
        
        sleep 30
    done
    
    print_warning "Monitoring timeout (24 hours) reached"
    return 1
}

# Get job status from SLURM (with retries for race conditions)
get_job_status() {
    local job_id=$1
    local retries=5
    
    for attempt in $(seq 1 $retries); do
        local status=$(squeue -j "$job_id" -o '%T' --noheader 2>/dev/null | xargs)
        
        if [ -z "$status" ]; then
            # Job not in queue - check sacct (might be completed)
            status=$(sacct -j "$job_id" -o State --noheader 2>/dev/null | head -1 | xargs)
        fi
        
        if [ -n "$status" ]; then
            echo "$status"
            return 0
        fi
        
        # If still not found, wait and retry
        if [ $attempt -lt $retries ]; then
            sleep 1
        fi
    done
    
    # If we still can't find it after retries
    echo "UNKNOWN"
}

# Submit a tokenization job
submit_job() {
    local split=$1
    local resume=$2
    local attempt=$3
    
    local job_name="tok-gigamidi-${split}"
    [ "$VANILLA" -eq 1 ] && job_name="${job_name}-vanilla"
    [ "$resume" -eq 1 ] && job_name="${job_name}-resume-${attempt}"
    
    # Build sbatch command that will run tokenize_gigaMIDI.py directly
    # We avoid custom.sh because it runs full pipeline; we want checkpoint-aware tokenization
    local python_cmd="cd '$PROJECT_ROOT' && python3 data/mus/symbolic/tokenization/anticipation/train/tokenize_gigaMIDI.py"
    python_cmd="$python_cmd --split '$split'"
    [ "$VANILLA" -eq 1 ] && python_cmd="$python_cmd --vanilla"
    [ "$resume" -eq 1 ] && python_cmd="$python_cmd --resume"
    
    # Set PYTHONPATH and other environment variables for imports
    local export_cmd="export PYTHONPATH='$PROJECT_ROOT/data/mus/symbolic:$PROJECT_ROOT/data/mus/symbolic/tokenization/anticipation:'\$PYTHONPATH && "
    export_cmd="$export_cmd export PYTHONUNBUFFERED=1 && "
    export_cmd="$export_cmd export TQDM_ISATTY=1 && "
    export_cmd="$export_cmd export TQDM_MININTERVAL=0.1 && "
    export_cmd="$export_cmd export WANDB_CONSOLE=wrap_raw && "
    export_cmd="$export_cmd export REMOTE_DATA_STORAGE='${CUSTOM_STORAGE_PATH:-}'"
    if [ -n "${WANDB_API_KEY:-}" ]; then
        export_cmd="$export_cmd && export WANDB_API_KEY='${WANDB_API_KEY}'"
    fi
    
    # Use shared log directory for accessible logs
    local log_dir="$HOME/slurm-logs"
    mkdir -p "$log_dir"
    local job_log="$log_dir/${job_name}.log"
    
    print_info "Submitting: sbatch --job-name='$job_name' -c 8 --mem=16G --time=12:00:00 ..."
    print_info "Log: $job_log"
    
    local output=$( \
        sbatch --job-name="$job_name" \
               --export=ALL \
               -c 8 \
               --mem=16G \
               --time=12:00:00 \
               --output="$job_log" \
               --error="$job_log" \
               --wrap="$export_cmd && $python_cmd" \
        2>&1 \
    )
    
    local job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' || echo "")
    
    if [ -n "$job_id" ]; then
        print_success "Job submitted with ID: $job_id"
        print_info "Logs: $job_log"
        echo "$job_id"
    else
        print_error "Failed to submit job"
        echo "$output" >&2
        echo ""
    fi
}

# Get the correct token directory based on vanilla flag
get_token_dir() {
    local folder_name
    if [ $VANILLA -eq 1 ]; then
        folder_name="anticipation-vanilla"
    else
        folder_name="anticipation"
    fi
    echo "/home/rebcecca/orcd/pool/music_datasets/GigaMIDI/tokens/$folder_name"
}

# Load checkpoint and get progress info
get_checkpoint_info() {
    local split=$1
    local token_dir=$(get_token_dir)
    local checkpoint_file="$token_dir/.checkpoint-${split}.json"
    
    if [ -f "$checkpoint_file" ]; then
        # Use jq to extract values
        local file_index=$(jq -r '.file_index' "$checkpoint_file" 2>/dev/null || echo "")
        local total_files=$(jq -r '.total_files' "$checkpoint_file" 2>/dev/null || echo "")
        local percent=$(jq -r '.percent_complete' "$checkpoint_file" 2>/dev/null || echo "")
        
        if [ -z "$file_index" ]; then
            echo ""
        else
            echo "file_index=$file_index total_files=$total_files percent=$percent"
        fi
    else
        echo ""
    fi
}

# Monitor a job and handle timeout
monitor_job() {
    local job_id=$1
    local split=$2
    local attempt=$3
    
    print_info "Monitoring job $job_id (split: $split, attempt: $attempt/$MAX_ATTEMPTS)"
    
    local check_count=0
    local max_checks=$((12*60*60 / POLL_INTERVAL))  # Allow up to 12 hours
    local token_dir=$(get_token_dir)
    local output_file="$token_dir/tokenized-events-${split}.txt"
    
    while [ $check_count -lt $max_checks ]; do
        check_count=$((check_count + 1))
        
        # Check 1: Is output file complete?
        if [ -f "$output_file" ] && [ -s "$output_file" ]; then
            print_success "Output file complete! Job succeeded"
            return 0
        fi
        
        # Check 2: What's job status?
        local job_status=$(sacct -j "$job_id" -o State --noheader 2>/dev/null | head -1 | xargs)
        
        if [ "$job_status" = "COMPLETED" ]; then
            # Job marked completed - check for output
            if [ -f "$output_file" ]; then
                print_success "Job COMPLETED with output"
                return 0
            else
                print_error "Job COMPLETED but no output file found"
                return 1
            fi
        elif [ "$job_status" = "FAILED" ]; then
            print_error "Job FAILED"
            return 1
        elif [ "$job_status" = "TIMEOUT" ]; then
            print_warning "Job TIMEOUT"
            return 2
        elif [ "$job_status" = "CANCELLED" ]; then
            print_error "Job CANCELLED"
            return 1
        elif [ -z "$job_status" ]; then
            # Job not in sacct yet - might be in squeue
            local squeue_status=$(squeue -j "$job_id" -o '%T' --noheader 2>/dev/null | xargs)
            if [ -n "$squeue_status" ]; then
                if [ $((check_count % 10)) -eq 0 ]; then
                    echo -ne "\r   ⏳ Check $check_count: status=$squeue_status"
                fi
            fi
        fi
        
        sleep "$POLL_INTERVAL"
    done
    
    print_warning "Monitoring timeout (12 hours) reached"
    return 1
}

# Process a single split with retry logic
process_split() {
    local split=$1
    local attempt=1
    
    print_header "Processing split: $split ($attempt/$MAX_ATTEMPTS)"
    
    while [ $attempt -le "$MAX_ATTEMPTS" ]; do
        # Determine if this is a resume attempt
        local resume=0
        [ $attempt -gt 1 ] && resume=1
        
        # Submit job
        local job_id=$(submit_job "$split" "$resume" "$attempt")
        
        if [ -z "$job_id" ]; then
            print_error "Failed to submit job for split '$split'"
            return 1
        fi
        
        # If detach mode, just return after submission
        if [ $DETACH -eq 1 ]; then
            print_success "Job submitted in detach mode. Use 'squeue -j $job_id' to check status."
            return 0
        fi
        
        # Wait a bit for job to appear in sacct (avoids race condition)
        sleep 3
        
        # Monitor job
        monitor_job "$job_id" "$split" "$attempt"
        local result=$?
        
        if [ $result -eq 0 ]; then
            # Successfully completed
            print_success "Split '$split' completed successfully!"
            return 0
        elif [ $result -eq 2 ]; then
            # Timeout - check for checkpoint and retry
            if [ $attempt -lt "$MAX_ATTEMPTS" ]; then
                local checkpoint_info=$(get_checkpoint_info "$split")
                if [ -n "$checkpoint_info" ]; then
                    print_success "Checkpoint found! $checkpoint_info"
                    attempt=$((attempt + 1))
                    echo ""
                    continue
                else
                    print_error "Timeout but no checkpoint found for '$split'"
                    return 1
                fi
            else
                print_error "Max attempts ($MAX_ATTEMPTS) reached for split '$split'"
                return 1
            fi
        else
            # Job failed or was cancelled
            print_error "Split '$split' failed"
            return 1
        fi
    done
    
    print_error "Exhausted retry attempts for split '$split'"
    return 1
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    parse_args "$@"
    
    print_header "🎵 GigaMIDI Auto-Tokenization with Checkpoint Recovery"
    
    # Print configuration
    local tokenizer_type="Vanilla"
    [ "$VANILLA" -eq 0 ] && tokenizer_type="Non-vanilla (Arrival-Time)"
    
    print_info "Configuration:"
    echo "   Tokenizer: Anticipation $tokenizer_type"
    echo "   Splits: ${SPLITS[*]}"
    echo "   Max attempts: $MAX_ATTEMPTS"
    echo "   Poll interval: ${POLL_INTERVAL}s"
    echo "   Skip Download: $([ $SKIP_DOWNLOAD -eq 1 ] && echo "Yes" || echo "No")"
    echo "   Download Only: $([ $DOWNLOAD_ONLY -eq 1 ] && echo "Yes" || echo "No")"
    echo ""
    
    # Check dependencies
    if ! check_dependencies; then
        print_error "Cannot proceed without dependencies"
        exit 1
    fi
    
    echo ""
    
    # STEP 0: Download and preprocessing (unless skipped)
    if [ $SKIP_DOWNLOAD -eq 0 ]; then
        if ! run_download_and_preprocessing; then
            print_error "Download/preprocessing failed"
            exit 1
        fi
        echo ""
    fi
    
    # If download-only mode, exit after preprocessing
    if [ $DOWNLOAD_ONLY -eq 1 ]; then
        print_header "✅ Download/Preprocess-Only Mode Complete"
        print_success "GigaMIDI data download and preprocessing completed. Tokenization jobs NOT submitted."
        exit 0
    fi
    
    # ========================================================================
    # PHASE 1: PARALLEL TOKENIZATION (via separate SLURM jobs)
    # ========================================================================
    print_header "🎹 PHASE 1: Submitting Tokenization Jobs (Per-Split)"
    
    # Process splits
    local completed=()
    local failed=()
    
    for split in "${SPLITS[@]}"; do
        if process_split "$split"; then
            completed+=("$split")
        else
            failed+=("$split")
        fi
        echo ""
    done
    
    # Summary
    print_header "📊 SUMMARY"
    
    echo "✅ Completed: ${#completed[@]}/${#SPLITS[@]}"
    for split in "${completed[@]}"; do
        print_success "$split"
    done
    
    if [ ${#failed[@]} -gt 0 ]; then
        echo ""
        echo "❌ Failed: ${#failed[@]}/${#SPLITS[@]}"
        for split in "${failed[@]}"; do
            print_error "$split"
        done
        exit 1
    else
        echo ""
        echo -e "${GREEN}🎉 All splits completed successfully!${NC}"
        exit 0
    fi
}

# Run main if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
