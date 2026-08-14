"""Record SNN-student rollout GIFs on the server (GPU inference, osmesa rendering).

Composites the event-camera view (what the SNN sees) picture-in-picture on the side view.
Usage: python scripts/record_student_server.py outputs/student_dagger.pt step parkour hurdle
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.envs import ParkourEnv, ParkourConfig
from es_parkour.models.student_snn import StudentSNN
from es_parkour.sensors import EventSimulator


def side_cam(x, goal_x):
    c = mujoco.MjvCamera()
    c.lookat[:] = [x, 0.0, 0.25]; c.distance = 2.2; c.azimuth = 90; c.elevation = -8
    return c


ckpt = sys.argv[1]
kinds = sys.argv[2:] or ["step", "parkour", "hurdle"]
ck = torch.load(ckpt, map_location="cpu")
hw = tuple(ck["event_hw"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
st = StudentSNN(base_channels=ck["base_channels"], event_hw=hw, T=ck["T"]).to(dev)
st.load_state_dict(ck["model"]); st.eval()
ev = EventSimulator()
OUT = REPO / "outputs" / "gif"; OUT.mkdir(parents=True, exist_ok=True)


def record(kind, seed):
    env = ParkourEnv(ParkourConfig(kinds=[kind], difficulty=0.15, render_depth=True,
                                   depth_hw=hw, episode_seconds=9.0), seed=seed)
    o = env.reset(kind=kind); prev = o["depth"].copy()
    r = mujoco.Renderer(env.model, 220, 340); frames = []
    t = 0; done = False; info = {}
    while not done:
        e_ch = ev.channels(ev.diff(o["depth"], prev))
        if t % 2 == 0:
            r.update_scene(env.data, camera=side_cam(float(env.robot.base_pos[0]), env.terrain.goal_x))
            f = Image.fromarray(r.render())
            pip = Image.fromarray(ev.to_rgb(e_ch[0] - e_ch[1])).resize((128, 96), Image.NEAREST)
            f.paste(pip, (f.width - pip.width - 4, 4))
            frames.append(f)
        with torch.no_grad():
            a, _, _ = st(torch.as_tensor(e_ch, dtype=torch.float32, device=dev).unsqueeze(0),
                         torch.as_tensor(o["proprio"], dtype=torch.float32, device=dev).unsqueeze(0),
                         st.init_hidden(1, device=dev))
        prev = o["depth"].copy()
        o, _, done, info = env.step(a.squeeze(0).cpu().numpy()); t += 1
    env.close(); r.close()
    return frames, info


for kind in kinds:
    best = None
    for seed in range(4):
        fr, info = record(kind, seed)
        sc = info["x"] / info["goal_x"] - (2.0 if info.get("out_of_corridor") else 0.0)
        if best is None or sc > best[0]:
            best = (sc, fr, info)
    sc, frames, info = best
    p = OUT / f"snn_{kind}.gif"
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=80, loop=0)
    print(f"{p.name}: progress={info['x']/info['goal_x']*100:.0f}% frames={len(frames)}")
