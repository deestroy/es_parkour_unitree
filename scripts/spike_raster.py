"""Spike raster + firing-rate graphs for the SNN student during a rollout.

Hooks every spiking neuron layer, records binary spike outputs at each of the T inner timesteps of
every control step, and renders:
  1. a raster plot (sampled neurons x global spiking timestep, one dot per spike),
  2. per-layer firing rates over the rollout (the sparsity behind the Eq. 10 energy numbers),
  3. the event-input activity for correlation.

Run: python3 scripts/spike_raster.py [student_ckpt] [terrain] [difficulty]
Saves outputs/spike_raster_<terrain>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.envs import ParkourEnv, ParkourConfig  # noqa: E402
from es_parkour.models.student_snn import StudentSNN  # noqa: E402
from es_parkour.sensors import EventSimulator  # noqa: E402
from spikingjelly.clock_driven import neuron  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "outputs" / "student_dagger.pt")
KIND = sys.argv[2] if len(sys.argv) > 2 else "step"
DIFF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
N_NEURONS = 120          # sampled neurons in the raster
MAX_STEPS = 150          # control steps to record (x T inner steps)

ck = torch.load(CKPT, map_location="cpu")
hw = tuple(ck["event_hw"])
st = StudentSNN(base_channels=ck.get("base_channels", 64), event_hw=hw, T=ck.get("T", 4))
st.load_state_dict(ck["model"]); st.eval()

# hook every spiking layer; name them by position in the network
spiking_layers = [(n, m) for n, m in st.named_modules()
                  if isinstance(m, (neuron.IFNode, neuron.LIFNode))]
print(f"{len(spiking_layers)} spiking layers found")
records = {n: [] for n, _ in spiking_layers}      # each: list of flat spike vectors per inner step

def make_hook(name):
    def hook(m, i, o):
        records[name].append(o.detach().flatten().numpy())
    return hook

handles = [m.register_forward_hook(make_hook(n)) for n, m in spiking_layers]

env = ParkourEnv(ParkourConfig(kinds=[KIND], difficulty=DIFF, render_depth=True,
                               depth_hw=hw, episode_seconds=8.0), seed=0)
ev = EventSimulator()
o = env.reset(kind=KIND)
prev = o["depth"].copy()
event_counts = []
with torch.no_grad():
    for t in range(MAX_STEPS):
        e_ch = ev.channels(ev.diff(o["depth"], prev))
        event_counts.append(float(e_ch.sum()))
        a, _, _ = st(torch.as_tensor(e_ch, dtype=torch.float32).unsqueeze(0),
                     torch.as_tensor(o["proprio"], dtype=torch.float32).unsqueeze(0),
                     st.init_hidden(1))
        prev = o["depth"].copy()
        o, _, done, info = env.step(a.squeeze(0).numpy())
        if done:
            break
env.close()
for h in handles:
    h.remove()
n_ctrl = t + 1
T = st.T
print(f"rollout: {n_ctrl} control steps, progress {info['x']/info['goal_x']*100:.0f}%")

# ---- assemble raster: sample neurons stratified across layers -----------------------------------
rng = np.random.default_rng(0)
raster_rows, row_labels, layer_bounds = [], [], []
per_layer = max(4, N_NEURONS // len(spiking_layers))
for name, _ in spiking_layers:
    R = np.stack(records[name])                     # (n_ctrl*T, n_neurons_layer)
    idx = rng.choice(R.shape[1], size=min(per_layer, R.shape[1]), replace=False)
    layer_bounds.append(len(raster_rows))
    for j in idx:
        raster_rows.append(R[:, j] > 0)
    row_labels.append(name.split(".")[-2] if "." in name else name)
raster = np.stack(raster_rows)                      # (rows, n_ctrl*T)

fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                         gridspec_kw={"height_ratios": [3, 1.2, 0.8]}, sharex=False)
ys, xs = np.nonzero(raster)
axes[0].scatter(xs, ys, s=1.2, c="k", marker="|")
for b in layer_bounds[1:]:
    axes[0].axhline(b - 0.5, color="tab:red", lw=0.4, alpha=0.5)
axes[0].set_ylabel(f"neuron (sampled, {raster.shape[0]} across {len(spiking_layers)} layers)")
axes[0].set_title(f"SNN spike raster — {KIND} rollout ({n_ctrl} control steps x T={T}); "
                  f"red lines separate layers (input-side at bottom)")
axes[0].set_xlim(0, raster.shape[1])

# per-layer firing rate over control steps
for name, _ in spiking_layers[:: max(1, len(spiking_layers)//6)]:
    R = np.stack(records[name])
    rate = R.mean(axis=1).reshape(n_ctrl, T).mean(axis=1)
    axes[1].plot(rate * 100, lw=0.9, label=name.rsplit(".", 1)[0][-18:])
axes[1].set_ylabel("firing rate (%)")
axes[1].legend(fontsize=6, ncol=3, loc="upper right")

axes[2].plot(event_counts, color="tab:purple", lw=0.9)
axes[2].set_ylabel("input events"); axes[2].set_xlabel("control step (50 Hz)")
plt.tight_layout()
out = REPO / "outputs" / f"spike_raster_{KIND}.png"
plt.savefig(out, dpi=130)
overall = np.concatenate([np.stack(records[n]).flatten() for n, _ in spiking_layers])
print(f"saved {out}  | overall firing rate {100*overall.mean():.2f}%")
