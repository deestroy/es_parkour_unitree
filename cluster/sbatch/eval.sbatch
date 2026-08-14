#!/bin/bash
#SBATCH --job-name=go2-snn-eval
#SBATCH --account=REPLACE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Per-terrain success rates (Fig. 5) + FLOPs/SOPs energy (Tables II/III, Eq. 10).
# Usage: sbatch eval.sbatch --load_run <logs/distill_go2/RUN>
set -euo pipefail
ROOT="${ES_GPU_ROOT:-$HOME/scratch/es_parkour_gpu}"
source "$ROOT/venv/bin/activate" 2>/dev/null || {
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate esparkour; }
export ES_SNN_OVERLAY=1
export PYTHONPATH="$ROOT/es_parkour_unitree:$PYTHONPATH"

cd "$ROOT/parkour/legged_gym"
python legged_gym/scripts/play.py --headless --task go2_distill "$@" \
    2>&1 | tee eval_rollout.log
python -m cluster.es_snn.energy_eval "$@"
