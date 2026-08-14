"""Record GIFs of a policy rolling out on each terrain, for visual inspection.

Run:
  python3 scripts/record_gif.py --policy teacher --ckpt outputs/teacher.pt --difficulty 0.15
  python3 scripts/record_gif.py --policy student --ckpt outputs/student.pt
  python3 scripts/record_gif.py --policy random

Saves outputs/gif/<policy>_<terrain>.gif (side view that follows the robot).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.envs import ParkourEnv, ParkourConfig, TERRAIN_KINDS  # noqa: E402
from es_parkour.models.teacher import TeacherActorCritic  # noqa: E402
from es_parkour.models.student_snn import StudentSNN  # noqa: E402
from es_parkour.sensors import EventSimulator  # noqa: E402

OUT = REPO / "outputs" / "gif"
OUT.mkdir(parents=True, exist_ok=True)


def side_cam(x, goal_x):
    c = mujoco.MjvCamera()
    c.lookat[:] = [x, 0.0, 0.25]
    c.distance = 2.2
    c.azimuth = 90
    c.elevation = -8
    return c


def load_policy(kind, ckpt):
    if kind == "random":
        return None, None
    ck = torch.load(ckpt, map_location="cpu")
    if kind == "teacher":
        m = TeacherActorCritic(); m.load_state_dict(ck["model"]); m.eval()
        mean = np.asarray(ck["norm_mean"], dtype=np.float32)
        std = np.sqrt(np.asarray(ck["norm_var"]) + 1e-8).astype(np.float32)
        return m, (mean, std)
    if kind == "student":
        hw = tuple(ck.get("event_hw", (48, 64)))
        m = StudentSNN(base_channels=ck.get("base_channels", 16), event_hw=hw, T=ck.get("T", 4))
        m.load_state_dict(ck["model"]); m.eval()
        return m, hw
    raise ValueError(kind)


@torch.no_grad()
def record(policy_kind, model, aux, terrain_kind, difficulty, seconds, fps, size):
    need_depth = (policy_kind == "student")
    hw = aux if policy_kind == "student" else (48, 64)
    env = ParkourEnv(ParkourConfig(kinds=[terrain_kind], difficulty=difficulty,
                                   render_depth=need_depth, depth_hw=hw,
                                   episode_seconds=seconds), seed=0)
    ev = EventSimulator()
    o = env.reset(kind=terrain_kind)
    renderer = mujoco.Renderer(env.model, height=size[0], width=size[1])
    prev_depth = o["depth"].copy() if need_depth else None
    h = model.init_hidden(1) if policy_kind == "student" else None

    every = max(1, round((1.0 / env.cfg.control_dt) / fps))
    frames, t, done = [], 0, False
    while not done:
        if policy_kind == "random":
            a = np.random.uniform(-1, 1, 12).astype(np.float32)
        elif policy_kind == "teacher":
            mean, std = aux
            to = (ParkourEnv.teacher_obs(o) - mean) / std
            a = model.action_mean(torch.as_tensor(to, dtype=torch.float32).unsqueeze(0))[0].numpy()
        else:  # student
            ev_ch = torch.as_tensor(ev.channels(ev.diff(o["depth"], prev_depth)),
                                    dtype=torch.float32).unsqueeze(0)
            pr = torch.as_tensor(o["proprio"], dtype=torch.float32).unsqueeze(0)
            act, _, h = model(ev_ch, pr, h)
            a = act.squeeze(0).numpy()
            prev_depth = o["depth"].copy()

        if t % every == 0:
            renderer.update_scene(env.data, camera=side_cam(float(env.robot.base_pos[0]), env.terrain.goal_x))
            frames.append(Image.fromarray(renderer.render()))
        o, _, done, info = env.step(a)
        t += 1
    env.close()

    path = OUT / f"{policy_kind}_{terrain_kind}.gif"
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
    return path, info.get("success", False), float(info.get("x", 0.0)), float(info.get("goal_x", 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["teacher", "student", "random"], default="teacher")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--kinds", nargs="*", default=list(TERRAIN_KINDS))
    ap.add_argument("--difficulty", type=float, default=0.15)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=int, nargs=2, default=[240, 360])
    args = ap.parse_args()

    ckpt = args.ckpt or str(REPO / "outputs" / (f"{args.policy}.pt"))
    model, aux = load_policy(args.policy, ckpt)
    print(f"recording {args.policy} on {args.kinds} (difficulty={args.difficulty})")
    for k in args.kinds:
        path, succ, x, gx = record(args.policy, model, aux, k, args.difficulty,
                                   args.seconds, args.fps, tuple(args.size))
        print(f"  {k:8s} -> {path.name}  success={succ}  progress={x/gx*100:.0f}% "
              f"(x={x:.2f}/{gx:.2f})")


if __name__ == "__main__":
    main()
