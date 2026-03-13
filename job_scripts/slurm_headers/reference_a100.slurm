#!/bin/bash
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu

# Resource Allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12    # Matches your num_workers=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G             # Total RAM for the node
#SBATCH --time=06:00:00       # 12-hour walltime

### TODO set the above mem-per-cpu and cpus-per-gpu according to your use case