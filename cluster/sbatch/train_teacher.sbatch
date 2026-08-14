#!/bin/bash
#SBATCH --job-name=go2-parkour-teacher
#SBATCH --account=REPLACE_ME            # your allocation, e.g. def-<pi-name>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00                 # paper: ~30 h on one RTX 3090
#SBATCH --output=%x-%j.out

set -euo pipefail
ROOT="${ES_GPU_ROOT:-$HOME/scratch/es_parkour_gpu}"
source "$ROOT/venv/bin/activate" 2>/dev/null || {
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate esparkour; }

cd "$ROOT/parkour/legged_gym"
# go2_field = the parkour oracle/teacher task (privileged obs, all obstacle types)
python legged_gym/scripts/train.py --headless --task go2_field "$@"
