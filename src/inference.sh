#!/bin/bash
# =============================================================================
#  inference.sh  —  SLURM job script for running inference on Marvin
#
#  Submit with:
#    sbatch inference.sh \
#        --input  /path/to/data.json \
#        --model  allenai/Olmo-3-7B-Think \
#        --output /path/to/results.json
#
#  All extra CLI args after the script name are forwarded to inference.py.
# =============================================================================

#SBATCH --job-name=inference
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 0.  Paths — adjust to your Marvin layout
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/physics_mamba}"
CONDA_ENV="${CONDA_ENV:-physics_mamba}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# ---------------------------------------------------------------------------
# 1.  Environment setup
# ---------------------------------------------------------------------------
echo "=== Job info ==="
echo "Host       : $(hostname)"
echo "SLURM job  : $SLURM_JOB_ID"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo "================"

# Load conda / mamba
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate "$CONDA_ENV"

# ---------------------------------------------------------------------------
# 2.  Install / verify dependencies (idempotent)
# ---------------------------------------------------------------------------
pip install --quiet --upgrade \
    vllm \
    transformers \
    accelerate \
    json-repair \
    torch

# ---------------------------------------------------------------------------
# 3.  Environment variables
# ---------------------------------------------------------------------------
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1           # no network access on compute nodes
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "$PROJECT_DIR/logs"

# ---------------------------------------------------------------------------
# 4.  Run inference
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"

echo ""
echo "=== Starting inference ==="
python inference.py "$@"
echo "=== Done: $(date) ==="
