"""Evaluate the spiking student: per-terrain success rate (Fig. 5) and motor energy (Table IV),
optionally across lighting conditions (Table V)."""
from __future__ import annotations

import numpy as np
import torch

from ..envs import ParkourEnv, ParkourConfig, TERRAIN_KINDS
from ..sensors import EventSimulator


@torch.no_grad()
def rollout_student(student, kind: str, episodes=5, difficulty=0.2, depth_hw=(48, 64),
                    brightness=1.0, threshold_c=0.15, seed=0, deterministic=True):
    cfg = ParkourConfig(kinds=[kind], difficulty=difficulty, render_depth=True,
                        depth_hw=depth_hw, brightness=brightness)
    env = ParkourEnv(cfg, seed=seed)
    ev = EventSimulator(threshold_c=threshold_c)
    student.eval()

    succ, energies, progress = [], [], []
    for e in range(episodes):
        o = env.reset(kind=kind)
        prev_depth = o["depth"].copy()
        h = student.init_hidden(1)
        ep_energy, done, info = 0.0, False, {}
        while not done:
            ev_ch = torch.as_tensor(ev.channels(ev.diff(o["depth"], prev_depth)),
                                    dtype=torch.float32).unsqueeze(0)
            pr = torch.as_tensor(o["proprio"], dtype=torch.float32).unsqueeze(0)
            a, _, h = student(ev_ch, pr, h)
            prev_depth = o["depth"].copy()
            o, _, done, info = env.step(a.squeeze(0).numpy())
            ep_energy += info["motor_energy"]
        succ.append(1.0 if info.get("success") else 0.0)
        energies.append(ep_energy)
        progress.append((info["x"]) / max(1e-6, info["goal_x"]))
    env.close()
    return {
        "success_rate": float(np.mean(succ)),
        "motor_energy_mJ": float(np.mean(energies)) * 1e3,
        "progress_frac": float(np.mean(progress)),
    }


def evaluate_all(student, kinds=TERRAIN_KINDS, episodes=5, difficulty=0.2, **kw):
    return {k: rollout_student(student, k, episodes=episodes, difficulty=difficulty, **kw)
            for k in kinds}
