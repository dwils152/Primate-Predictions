#!/bin/bash
# GPU job submitted by ./check_install.sh. Use that script instead of this file.
#SBATCH --job-name=check-install
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f scripts/common.sh ]] || { echo "Submit this job from the Primate-Predictions folder." >&2; exit 1; }
source scripts/common.sh
export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started $(date)"
"$PY" src/check_install.py
echo "Finished $(date)"
