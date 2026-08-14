"""Quantify gait quality of a teacher checkpoint: all-airborne fraction + vertical-velocity RMS.

A hop/bound has a high all-airborne fraction and high vz_rms; a grounded trot has ~0% all-airborne
and low vz_rms. Also reports forward progress so we see the gait fix didn't kill locomotion.

Run:  python3 scripts/gait_check.py outputs/teacher_run.pt [kind] [difficulty]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.envs import ParkourEnv, ParkourConfig  # noqa: E402
from es_parkour.models.teacher import TeacherActorCritic  # noqa: E402


def check(ckpt, kind="flat", difficulty=0.0, seconds=8.0):
    ck = torch.load(ckpt, map_location="cpu")
    m = TeacherActorCritic(); m.load_state_dict(ck["model"]); m.eval()
    mean = ck["norm_mean"].astype(np.float32); std = np.sqrt(ck["norm_var"] + 1e-8).astype(np.float32)
    env = ParkourEnv(ParkourConfig(kinds=[kind], difficulty=difficulty,
                                   render_depth=False, episode_seconds=seconds), seed=0)
    o = env.reset(kind=kind)
    vzs, allfly, contacts_per_step, n = [], 0, [], 0
    front_landings, front_together = 0, 0
    prev_c = env.robot.foot_contacts()
    for _ in range(int(seconds / env.cfg.control_dt)):
        to = (ParkourEnv.teacher_obs(o) - mean) / std
        a = m.action_mean(torch.as_tensor(to, dtype=torch.float32).unsqueeze(0))[0].numpy()
        o, r, d, info = env.step(a)
        vzs.append(abs(env.robot.base_lin_vel_world[2]))
        c = env.robot.foot_contacts()          # (FL, FR, RL, RR)
        # front-leg landing analysis: of steps where a front foot lands, how often do BOTH land at once?
        fl_land = c[0] and not prev_c[0]
        fr_land = c[1] and not prev_c[1]
        if fl_land or fr_land:
            front_landings += 1
            if fl_land and fr_land:
                front_together += 1
        prev_c = c
        allfly += (0 if c.any() else 1); contacts_per_step.append(int(c.sum())); n += 1
        if d:
            break
    env.close()
    return {
        "vz_rms": float(np.sqrt(np.mean(np.square(vzs)))),
        "all_airborne_pct": 100.0 * allfly / n,
        "mean_feet_down": float(np.mean(contacts_per_step)),
        "front_together_pct": 100.0 * front_together / max(1, front_landings),
        "progress_pct": 100.0 * info["x"] / info["goal_x"],
    }


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "outputs" / "teacher_run.pt")
    kind = sys.argv[2] if len(sys.argv) > 2 else "flat"
    diff = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    r = check(ckpt, kind, diff)
    print(f"{ckpt}  [{kind} d={diff}]")
    print(f"  all-airborne: {r['all_airborne_pct']:.0f}%   vz_rms: {r['vz_rms']:.3f} m/s   "
          f"mean feet down: {r['mean_feet_down']:.2f}/4   front-legs-land-together: "
          f"{r['front_together_pct']:.0f}%   progress: {r['progress_pct']:.0f}%")
    print("  (dog-like trot = ~0% airborne, low vz_rms, ~2 feet down, front-together ~0%)")
