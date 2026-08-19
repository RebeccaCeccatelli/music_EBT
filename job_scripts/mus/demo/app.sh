#!/bin/bash
### EBT Demo UI
### Launches the Gradio app on a GPU node.
###
### Usage:
###   sbatch job_scripts/mus/demo/app.sh           # internal access via SSH tunnel
###   sbatch job_scripts/mus/demo/app.sh --share   # public gradio.live URL

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
#SBATCH --output=./logs/slurm_demo_%j.out
#SBATCH --job-name=ebt-demo

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
    echo "❌ Could not find project root."
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${HOME}/music-EBT/data/mus/symbolic:$PYTHONPATH"
export PATH="${HOME}/.conda/envs/music_EBT/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}" || exit 1

PORT=7860
SHARE_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --share) SHARE_FLAG="--share"; shift ;;
        --port)  PORT="$2";            shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "EBT Demo UI"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null)"
echo "Port:    ${PORT}"
if [[ -n "${SHARE_FLAG}" ]]; then
    echo "Mode:    public (gradio.live tunnel)"
else
    echo "Mode:    local — SSH tunnel to access:"
    echo "         ssh -L ${PORT}:$(hostname):${PORT} rebcecca@eosloan.mit.edu"
    echo "         then open http://localhost:${PORT}"
fi
echo "=========================================="

python demo/app.py --port "${PORT}" ${SHARE_FLAG}
