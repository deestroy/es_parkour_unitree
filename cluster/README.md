# ES-Parkour GPU recreation — SLURM cluster runbook

Exact-paper stack: **IsaacGym Preview 4 + Zhuang et al. "Robot Parkour Learning"** (`legged_gym` +
`rsl_rl`, the codebase ES-Parkour's teacher follows) for the Go2 teacher, then **our event-camera +
spikingjelly SNN distillation** on top (the `es_snn/` overlay in this folder).

## 0. One manual step first (NVIDIA license)

IsaacGym Preview 4 must be downloaded by a logged-in NVIDIA account (no direct curl):
https://developer.nvidia.com/isaac-gym  →  `IsaacGym_Preview_4_Package.tar.gz`
Upload it to the cluster, e.g.:
```bash
scp IsaacGym_Preview_4_Package.tar.gz <cluster>:~/scratch/
```

## 1. Setup (login node)

```bash
cd ~/scratch
git clone <this repo> es_parkour_unitree && cd es_parkour_unitree/cluster
bash setup.sh ~/scratch/IsaacGym_Preview_4_Package.tar.gz
```

`setup.sh` creates a Python 3.8 env (venv + `module load` if available, else conda), installs
IsaacGym from the tarball, clones `ZiwenZhuang/parkour`, installs `rsl_rl` and `legged_gym`
(editable), plus `spikingjelly` for the SNN. It ends with a CUDA + IsaacGym smoke test.

Notes for Alliance-style clusters (Graham/Cedar/Narval):
- Run setup on a **login node** (compute nodes may lack internet).
- IsaacGym needs `libpython3.8`; the venv route with `module load python/3.8` handles it. If you
  hit `libpython3.8.so.1.0` errors: `export LD_LIBRARY_PATH=$EBROOTPYTHON/lib:$LD_LIBRARY_PATH`.
- IsaacGym supports NVIDIA GPUs with driver >= 470; A100/V100/RTX all fine **headless**.

## 2. Train the teacher (paper's oracle policy)

```bash
sbatch sbatch/train_teacher.sbatch      # go2_field task, headless
```
Edit the `#SBATCH --account=...` / `--gres=gpu:1` lines for your allocation first.
Logs land in `parkour/legged_gym/logs/field_go2/<run>`; the paper trained ~30 h on one RTX 3090 —
budget similar. Monitor with `tail -f` on the SLURM output file.

## 3. Distill to the SNN student (ES-Parkour's contribution)

```bash
sbatch sbatch/distill_snn.sbatch --load_run <field_go2 run dir>
```
This runs the stock `go2_distill` flow with our overlay substituted in (see `es_snn/README.md`):
depth frames → simulated events (Eq. 1–4) → spiking ResNet-18 + GRU + spiking MLP student,
trained with the paper's action-MSE (Eq. 8) + yaw (Eq. 9) losses.

## 4. Evaluate

```bash
sbatch sbatch/eval.sbatch               # success rates per terrain + FLOPs/SOPs energy (Eq. 10)
python legged_gym/scripts/play.py --task go2_field --load_run <run>   # visualize (X11/offscreen)
```

## What ports over from the CPU baseline (already-verified code)

| Piece | File (this repo) | Status on GPU |
|---|---|---|
| Event simulation (Eq. 1–4) | `es_parkour/sensors/event_sim.py` | torch/numpy — works as-is |
| SNN student (spiking ResNet-18+GRU+MLP) | `es_parkour/models/student_snn.py` | `.to(cuda)`; use `base_channels=64` (full width) |
| Distill losses (Eq. 8–9) | `es_parkour/train/distill.py` | logic reused in `es_snn/` overlay |
| Energy accounting (Eq. 10) | `es_parkour/eval/energy.py` | works as-is |

The MuJoCo env/PPO stay as the CPU fallback (commit-tagged); IsaacGym replaces them here.
