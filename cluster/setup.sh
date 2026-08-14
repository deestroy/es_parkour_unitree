#!/usr/bin/env bash
# ES-Parkour GPU setup: IsaacGym Preview 4 + Zhuang parkour repo + SNN deps.
# Usage: bash setup.sh /path/to/IsaacGym_Preview_4_Package.tar.gz [install_dir]
set -euo pipefail

ISAACGYM_TARBALL="${1:?usage: setup.sh <IsaacGym_Preview_4_Package.tar.gz> [install_dir]}"
ROOT="${2:-$PWD/es_parkour_gpu}"
mkdir -p "$ROOT" && cd "$ROOT"

echo "== [1/6] Python 3.8 environment =="
if command -v module >/dev/null 2>&1 && module avail python/3.8 2>&1 | grep -q "python/3.8"; then
    module load python/3.8
    python3.8 -m venv venv
    source venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
    conda create -y -n esparkour python=3.8
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate esparkour
else
    python3.8 -m venv venv && source venv/bin/activate
fi
python -V | grep -q "3\.8" || { echo "ERROR: need Python 3.8 (IsaacGym requirement)"; exit 1; }
pip install --upgrade pip

echo "== [2/6] PyTorch (CUDA) =="
# parkour repo is tested against torch 2.4.1; cu118 wheel covers driver >=470 setups.
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

echo "== [3/6] IsaacGym Preview 4 =="
tar -xzf "$ISAACGYM_TARBALL"
pip install -e isaacgym/python
python - <<'PY'
import isaacgym  # noqa: F401  (import order check: isaacgym BEFORE torch at runtime)
print("isaacgym import OK")
PY

echo "== [4/6] Zhuang et al. parkour repo (legged_gym + rsl_rl) =="
[ -d parkour ] || git clone https://github.com/ZiwenZhuang/parkour.git
pip install -e parkour/rsl_rl
pip install -e parkour/legged_gym

echo "== [5/6] ES-Parkour SNN deps =="
pip install spikingjelly numpy tensorboard tqdm pyyaml

echo "== [6/6] smoke test =="
python - <<'PY'
import isaacgym  # must precede torch
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
from spikingjelly.clock_driven import neuron  # noqa: F401
print("spikingjelly OK")
PY

echo
echo "Setup complete. Activate with:  source $ROOT/venv/bin/activate  (or conda activate esparkour)"
echo "Train teacher:  cd $ROOT/parkour/legged_gym && python legged_gym/scripts/train.py --headless --task go2_field"
