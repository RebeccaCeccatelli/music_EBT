#!/bin/bash
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu

# Resource Allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    # Matches your num_workers=12
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G             # Total RAM for the node
#SBATCH --time=06:00:00       # 6-hour walltime

# Disable shared memory for NCCL to avoid /dev/shm limits
export NCCL_SHM_DISABLE=1
# Optional: force NCCL to use a specific interface if the node is confused
export NCCL_P2P_DISABLE=1
# --- Ensure your HF_HOME is set for every job ---
export HF_HOME="/home/rebcecca/orcd/pool/hf_cache"

# Limit the number of parallel download threads to 1
export HF_DATASETS_MAX_THREADS=1

# Tell the library to retry failed downloads up to 20 times (giving the server breathing room)
export HF_DATASETS_DOWNLOAD_RETRY_LIMIT=20

### TODO set the above mem-per-cpu and cpus-per-gpu according to your use case