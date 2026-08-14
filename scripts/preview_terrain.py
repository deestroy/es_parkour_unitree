"""Render each terrain (side view + robot head camera) so the terrains are visually verifiable.

Run:  python3 scripts/preview_terrain.py [difficulty]
Saves outputs/terrain_<kind>.png and a montage outputs/terrain_all.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from es_parkour.sim import load_model, Go2  # noqa: E402
from es_parkour.envs import make_terrain, TERRAIN_KINDS  # noqa: E402

OUT = REPO / "outputs"
OUT.mkdir(exist_ok=True)


def side_camera(goal_x: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [goal_x * 0.5, 0.0, 0.2]
    cam.distance = max(3.5, goal_x * 1.1)
    cam.azimuth = 110
    cam.elevation = -18
    return cam


def render(model, data, camera, H=300, W=480):
    r = mujoco.Renderer(model, height=H, width=W)
    r.update_scene(data, camera=camera)
    img = r.render()
    return img


def main():
    d = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    fig, axes = plt.subplots(len(TERRAIN_KINDS), 2, figsize=(9, 3 * len(TERRAIN_KINDS)))

    for row, kind in enumerate(TERRAIN_KINDS):
        terr = make_terrain(kind, difficulty=d)
        model = load_model(terr.xml)
        data = mujoco.MjData(model)
        robot = Go2(model, data)
        robot.reset_to_home(base_pos=terr.spawn_pos)
        # let it settle briefly so it rests on the surface
        q = robot.joint_pos.copy()
        for _ in range(300):
            robot.pd_control(q)
            mujoco.mj_step(model, data)

        side = render(model, data, side_camera(terr.goal_x))
        head = render(model, data, "head", H=180, W=240)

        axes[row, 0].imshow(side)
        axes[row, 0].set_title(f"{kind}  (d={d:.2f}, goal_x={terr.goal_x:.1f}, "
                               f"obstacles={len(terr.obstacles)})", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(head)
        axes[row, 1].set_title(f"{kind} — head camera", fontsize=9)
        axes[row, 1].axis("off")

        # also save the side view standalone
        plt.imsave(OUT / f"terrain_{kind}.png", side)
        print(f"{kind:8s} goal_x={terr.goal_x:5.2f}  spawn_z={terr.spawn_pos[2]:.2f}  "
              f"obstacles={len(terr.obstacles)}  base_z_after_settle={robot.base_pos[2]:.3f}")

    plt.tight_layout()
    plt.savefig(OUT / "terrain_all.png", dpi=110)
    print(f"saved montage: {OUT/'terrain_all.png'}")


if __name__ == "__main__":
    main()
