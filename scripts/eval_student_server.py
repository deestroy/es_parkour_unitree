"""Server-side SNN student evaluation: per-terrain rollouts (GPU inference) + Eq. 10 energy."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.envs import ParkourEnv, ParkourConfig
from es_parkour.models.student_snn import StudentSNN
from es_parkour.sensors import EventSimulator
from es_parkour.eval.energy import energy_report

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "outputs/student_full.pt", map_location="cpu")
hw = tuple(ck["event_hw"])
student = StudentSNN(base_channels=ck["base_channels"], event_hw=hw, T=ck["T"])
student.load_state_dict(ck["model"]); student.eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
sg = StudentSNN(base_channels=ck["base_channels"], event_hw=hw, T=ck["T"]).to(dev)
sg.load_state_dict(ck["model"]); sg.eval()

ev = EventSimulator()
print(f"=== SNN student rollouts (difficulty 0.15, device={dev}) ===")
for kind in ["flat", "gap", "step", "hurdle", "parkour"]:
    prs, srs = [], []
    for seed in range(3):
        env = ParkourEnv(ParkourConfig(kinds=[kind], difficulty=0.15, render_depth=True,
                                       depth_hw=hw, episode_seconds=9.0), seed=seed)
        o = env.reset(kind=kind); prev = o["depth"].copy()
        h = sg.init_hidden(1, device=dev); done = False; info = {}
        reset_h = len(sys.argv) > 2 and sys.argv[2] == "--reset-hidden"
        while not done:
            if reset_h:
                h = sg.init_hidden(1, device=dev)
            e = torch.as_tensor(ev.channels(ev.diff(o["depth"], prev)),
                                dtype=torch.float32, device=dev).unsqueeze(0)
            p = torch.as_tensor(o["proprio"], dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                a, _, h = sg(e, p, h)
            prev = o["depth"].copy()
            o, _, done, info = env.step(a.squeeze(0).cpu().numpy())
        prs.append(info["x"] / info["goal_x"]); srs.append(1.0 if info["success"] else 0.0)
        env.close()
    print(f"  {kind:8s} success={np.mean(srs)*100:3.0f}%  progress={np.mean(prs)*100:3.0f}%")

env = ParkourEnv(ParkourConfig(kinds=["hurdle"], difficulty=0.15, render_depth=True, depth_hw=hw), seed=0)
o = env.reset(kind="hurdle"); d0 = o["depth"].copy()
o, _, _, _ = env.step(np.zeros(12, dtype=np.float32))
ech = torch.as_tensor(ev.channels(ev.diff(o["depth"], d0)), dtype=torch.float32).unsqueeze(0)
pch = torch.as_tensor(o["proprio"], dtype=torch.float32).unsqueeze(0)
env.close()
rep = energy_report(student, ech, pch)
print("=== ENERGY (Eq. 10, full-width spiking ResNet-18) ===")
print(f"  ANN FLOPs: {rep['ann_flops']:.3e}   SNN ops: {rep['snn_ops']:.3e}   "
      f"firing rate: {rep['mean_firing_rate']*100:.1f}%")
print(f"  OPs ratio SNN:ANN = {rep['ops_ratio_snn_over_ann']:.3f}:1")
print(f"  E_ANN={rep['E_ann_J']*1e6:.2f}uJ  E_SNN={rep['E_snn_J']*1e6:.2f}uJ  "
      f"saving={rep['energy_saving_pct']:.1f}%  (paper: 88.3%)")
