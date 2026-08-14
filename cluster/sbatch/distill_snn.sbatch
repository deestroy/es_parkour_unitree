#!/bin/bash
#SBATCH --job-name=go2-snn-distill
#SBATCH --account=REPLACE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# ES-Parkour distillation: teacher (go2_field ckpt) -> spiking event-camera student.
# Pass the teacher run:  sbatch distill_snn.sbatch --load_run <logs/field_go2/RUN>
set -euo pipefail
ROOT="${ES_GPU_ROOT:-$HOME/scratch/es_parkour_gpu}"
source "$ROOT/venv/bin/activate" 2>/dev/null || {
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate esparkour; }

# Overlay: registers the SNN student + event transform into the go2_distill task
# (see es_snn/README.md; ES_SNN_OVERLAY=1 activates the patched modules).
export ES_SNN_OVERLAY=1
export PYTHONPATH="$ROOT/es_parkour_unitree:$PYTHONPATH"

cd "$ROOT/parkour/legged_gym"
python legged_gym/scripts/train.py --headless --task go2_distill "$@"
