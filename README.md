# ES-Parkour on Unitree Go2 + MuJoCo (CPU)

A CPU-only recreation of **ES-Parkour: Advanced Robot Parkour with Bio-inspired Event Camera and
Spiking Neural Network**, using a **Unitree Go2 in MuJoCo** instead of the paper's Spot + IsaacGym.
Faithfulness is to the **method and experiments** (privileged teacher → event camera → SNN student →
distillation → energy/success eval), not to the paper's exact numbers — a 12-DOF Go2 on single-env
CPU won't match 32 parallel GPU envs.

## Pipeline

```
 privileged PPO teacher            depth camera            spiking student            distillation
 (proprio + scandots + heading) -> (MuJoCo render) -> events -> (spiking ResNet-18 -> ANN teacher actions
        MLP actor/critic              Eq.1/Eq.4        GRU -> spiking MLP)   supervise    (Eq.8 action MSE
                                                                                           + Eq.9 yaw MSE)
                                                                          |
                                                                          v
                                             success rate (Fig.5) + energy FLOPs/SOPs (Eq.10, Tables II/III)
```

## Layout

| Path | Role |
|------|------|
| `third_party/unitree_mujoco/` | Go2 MJCF (`unitree_robots/go2/go2.xml`, patched to add a `head` camera) + `terrain_tool/` |
| `es_parkour/sim/scene.py`, `go2.py` | scene composition; Go2 interface (joint maps, PD control, state) |
| `es_parkour/envs/terrain.py` | gap / step / hurdle / parkour generators + difficulty curriculum |
| `es_parkour/envs/camera.py`, `obs.py` | depth camera; proprio + privileged scandots (ray-cast) + oracle heading |
| `es_parkour/envs/parkour_env.py` | `ParkourEnv` — reset/step, parkour reward, termination, curriculum, model cache |
| `es_parkour/sensors/event_sim.py` | depth → events (Eq.1 temporal-diff, Eq.4 motion-field) |
| `es_parkour/models/teacher.py` | privileged ANN actor-critic (scandot encoder + MLP) |
| `es_parkour/models/student_snn.py` | spiking ResNet-18 + GRU + 3-layer spiking MLP `[512,256,128]` + heading head |
| `es_parkour/train/ppo.py`, `train_teacher.py` | compact custom PPO (asymmetric critic, torch/CPU) |
| `es_parkour/train/distill.py` | ANN→SNN distillation (Eq.8 + Eq.9) |
| `es_parkour/eval/energy.py`, `evaluate.py` | FLOPs/SOPs energy (Eq.10) + per-terrain success & motor energy |

## Setup

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco
# then re-apply the head camera to third_party/unitree_mujoco/unitree_robots/go2/go2.xml
# (a <camera name="head" pos="0.33 0 0.08" xyaxes="0 -1 0 0.5 0 0.866" fovy="70"/> inside base_link)
```
Verified on Python 3.9, macOS x86_64: `mujoco==3.2.7` (newest cp39 wheel), `torch==2.2.2`,
`spikingjelly==0.0.0.0.14`, `numpy==1.23.5`.

## Run

```bash
python3 scripts/smoke_test.py                 # Go2 stands + renders a depth frame  -> outputs/
python3 scripts/preview_terrain.py 0.4         # render all four terrains            -> outputs/terrain_all.png
# vectorized teacher training (N parallel envs + walking->parkour curriculum):
python3 scripts/train_teacher.py --steps 1500000 --n-envs 6 --difficulty 0.3 --curriculum
python3 scripts/distill.py --teacher outputs/teacher.pt --samples 2000 --epochs 3  # -> outputs/student.pt
python3 scripts/evaluate.py --student outputs/student.pt --episodes 5 --lighting   # tables
# visual check — GIFs of a policy on each terrain:
python3 scripts/record_gif.py --policy teacher --ckpt outputs/teacher.pt --difficulty 0.15
python3 scripts/record_gif.py --policy random   # baseline (untrained) for comparison
```

### Performance notes (Phase 2)
- **Vectorized envs** (`es_parkour/envs/vec_env.py`): N workers via `multiprocessing` (`spawn`, safe for
  MuJoCo on macOS), auto-reset. **Critical CPU details**: pin each process to one math thread
  (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`) or N workers thrash the cores; each worker writes a
  **PID-unique** scene file so parallel model compiles don't clobber a shared file. ~**5× throughput**
  (≈400 steps/s at 6 envs vs ~77/s single-env).
- **Curriculum + reward**: `flat` warm-up terrain (walk before obstacles) plus a **feet-air-time reward**
  to encourage a stepping gait; difficulty ramps 0→target over the first 60% of training. The model
  cache buckets difficulty at 0.1 so a continuously-ramped curriculum doesn't trigger constant recompiles.

## Status (vertical slice)

Every stage is implemented and independently verified:
- **Sim/terrains/obs/env** ✅ — Go2 stands stably; four terrains render; scandots capture terrain height;
  `ParkourEnv` runs clean on all terrains.
- **Event camera** ✅ — depth→event frames show red/blue polarity on obstacle edges (Fig. 1 style).
- **SNN student** ✅ — spiking ResNet-18 + GRU + spiking MLP forward/backward runs.
- **Distillation** ✅ — action+yaw MSE drops ~8× over a short run.
- **Energy (Eq.10)** ✅ — at ~17% firing rate, **OPs(SNN)/OPs(ANN) ≈ 0.70:1, ~84% energy saving**
  (paper ~88%). This reproduces the paper's central efficiency claim.

### Teacher training results (Phase 2)
With vectorized envs + tuned reward, the Go2 **learns to walk from scratch** and traverse most terrain.
Final adaptive-fine-tuned teacher (`teacher_ft_best.pt`), deterministic rollout at difficulty 0.1:
**flat 92%, parkour 82%, step 71%, hurdle 61%, gap 0%** progress. Gap (crossing a pit) stays beyond
this robot — as in the paper, where gap was also the hardest terrain (~45%).

Training stability notes (hard-won):
- **`log_std` clamp** (`models/teacher.py`, max −1.0): a first long run *collapsed* late (entropy drifted
  5→8.75, σ 0.37→0.8, policy un-learned to walk). Clamping the action std so it can only shrink fixed it.
- **Curriculum matters a lot**: ramping difficulty to 0.3 fed the policy nothing but failure and
  destabilized it. Cap at ~0.15, or use the adaptive curriculum.
- **Adaptive curriculum** (`scripts/finetune_teacher.py`): each terrain's difficulty rises when its
  success clears a threshold and eases when it drops. Caveat found: with a 0 floor and an eager
  down-threshold it drove all difficulties to 0 (lost challenge) — floor the difficulty and raise more
  readily to push harder. Fine-tuning from a stable base still gave the best policy above.

### GPU recreation (exact paper stack)
See **[cluster/README.md](cluster/README.md)**: IsaacGym Preview 4 + Zhuang et al.'s parkour
codebase (`go2_field` teacher, `go2_distill` student) on a SLURM cluster, with our event-camera +
spiking-network overlay in `cluster/es_snn/`. The MuJoCo implementation in this repo remains the
CPU fallback.

### Scaling status (GPU-parallel simulation)
ROCm JAX works on the AMD MI210 (`jax 0.11`, RocmDevice visible) and **single-env MJX steps run**,
but any `vmap`-batched `mjx.step` crashes in kernel launch (HSA malformed packet) — a ROCm-XLA
plugin bug as of jax-rocm7 wheels, not a design problem. `scripts/mjx_benchmark.py` is the go/no-go
test; rerun it on newer jax-rocm releases or on NVIDIA hardware, where MJX batching works today.
Until then, scale training = the `cluster/` IsaacGym package on an NVIDIA machine.

### Next steps
- **Distill the trained teacher** (`teacher_ft_best.pt`) into the SNN student and run the full eval so the
  bio-inspired pipeline reports real success/energy numbers (not the smoke-teacher ones).
- Refine the adaptive curriculum (floor difficulty > 0, raise more readily) to push obstacle success higher.
- Full-width spiking ResNet-18 (`base_channels=64`) and recurrent BPTT distillation.
- Full Tables II–V with the trained student, plus the lighting-robustness sweep (`--lighting`).
