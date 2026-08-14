# ES-SNN overlay — wiring the event camera + spiking student into `go2_distill`

Zhuang's `go2_distill` task trains an ANN student (depth-camera encoder + recurrent policy) to
imitate the `go2_field` oracle. ES-Parkour's change is threefold, and this overlay provides each
piece as sim-agnostic torch code (already validated in the MuJoCo baseline):

| ES-Parkour change | Overlay module | Wires into |
|---|---|---|
| depth frames → polarity **events** (Eq. 1–4) | `event_wrapper.py` (`DepthToEvents`) | the distill env's camera obs pipeline (where depth images are put into the student's observation) |
| ANN encoder → **spiking ResNet-18 + GRU + spiking MLP [512,256,128]**, LIF, T=4 | `snn_policy.py` (`SNNStudentPolicy`) | `rsl_rl` policy class used by the distill runner (config's `policy_class_name` / module registry) |
| imitation loss → **action MSE (Eq. 8) + yaw MSE (Eq. 9)** | `snn_policy.py` (`distill_losses`) | the distill algorithm's loss computation |
| energy accounting (Eq. 10) | `energy_eval.py` | post-hoc on a trained student |

## Integration steps (on the cluster, ~an afternoon)

1. Locate the Go2 distill config under `parkour/legged_gym/legged_gym/envs/go2/` (the file defining
   the `go2_distill` task: camera resolution, student encoder class, dagger/collect settings).
2. Point the student policy at `cluster.es_snn.snn_policy.SNNStudentPolicy` (subclass of the ANN
   student's interface: same `act/act_inference/reset` signatures, recurrent hidden state = GRU h).
3. Insert `DepthToEvents` where the env writes camera obs: it keeps the previous depth frame per
   env-id and emits a 2-channel (+/−) event tensor — batch-safe, GPU-resident, no host copies.
4. Add the yaw-MSE term: the oracle exposes target yaw in its privileged obs; the student's
   `predicted_heading` head regresses it (Eq. 9) alongside the action MSE (Eq. 8).
5. Keep the paper's hyperparameters: lr 1e-3, event sampling 10 Hz (set camera update interval
   accordingly), T=4, full-width encoder (`base_channels=64`).

Import order caveat: **`import isaacgym` must precede any `import torch`** in every entry point.
The sbatch scripts already run inside legged_gym's scripts, which respect this.
