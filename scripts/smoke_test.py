"""Step 1 smoke test: load Go2, hold a stable stance under PD, render one depth frame.

Run:  python3 scripts/smoke_test.py
Saves RGB + depth PNGs to outputs/ and prints whether the robot stayed standing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco  # noqa: E402
from es_parkour.sim import load_model, Go2  # noqa: E402
from es_parkour.sim.scene import SIM_DT  # noqa: E402

OUT = REPO / "outputs"
OUT.mkdir(exist_ok=True)


def save_gray(arr: np.ndarray, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap="viridis")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_rgb(arr: np.ndarray, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    model = load_model()  # flat ground
    data = mujoco.MjData(model)
    robot = Go2(model, data)
    robot.reset_to_home()

    print(f"model: nq={model.nq} nv={model.nv} nu={model.nu} sim_dt={SIM_DT}")
    print(f"joints mapped: {len(robot.jnt_ids)}  actuators mapped: {len(robot.act_ids)}")

    # Hold the home stance for ~2 s of sim time.
    q_des = robot.joint_pos.copy()
    heights, steps = [], int(2.0 / SIM_DT)
    for _ in range(steps):
        robot.pd_control(q_des)
        mujoco.mj_step(model, data)
        heights.append(float(robot.base_pos[2]))

    heights = np.array(heights)
    z0, zf, zmin = heights[0], heights[-1], heights.min()
    tail = heights[-int(0.5 / SIM_DT):]          # last 0.5 s
    settled = tail.std() < 0.005                  # not oscillating / falling
    upright = robot.projected_gravity[2] < -0.95  # trunk still level
    standing = zmin > 0.20 and settled and upright
    print(f"base height: start={z0:.3f} end={zf:.3f} min={zmin:.3f} "
          f"tail_std={tail.std():.4f}  proj_grav_z={robot.projected_gravity[2]:.3f}  -> "
          f"{'STANDING OK' if standing else 'FELL / UNSTABLE'}")

    # Render one RGB + one depth frame from the head camera.
    H, W = 120, 160
    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.update_scene(data, camera="head")
    rgb = renderer.render()
    save_rgb(rgb, OUT / "smoke_rgb.png")

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="head")
    depth = renderer.render()
    renderer.disable_depth_rendering()
    finite = depth[np.isfinite(depth)]
    print(f"depth frame: shape={depth.shape} min={finite.min():.3f} "
          f"max={finite.max():.3f} mean={finite.mean():.3f} m")
    save_gray(depth, OUT / "smoke_depth.png")

    print(f"saved: {OUT/'smoke_rgb.png'}  {OUT/'smoke_depth.png'}")
    return 0 if standing else 1


if __name__ == "__main__":
    raise SystemExit(main())
