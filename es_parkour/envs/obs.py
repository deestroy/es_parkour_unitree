"""Observation assembly: proprioception, privileged scandots (ray-cast), and oracle heading.

Split by information type so the teacher (privileged) and student (vision) can each take what they need:
  * proprioception  -> both teacher and student
  * scandots        -> teacher only (privileged terrain heightmap around the base)
  * oracle heading  -> teacher only (the student *predicts* heading instead)
"""
from __future__ import annotations

import numpy as np
import mujoco

from ..sim.go2 import DEFAULT_STANCE, Go2

# Scandot grid (in the base yaw frame): more samples ahead of the robot than behind.
SCAN_X = np.linspace(-0.3, 0.9, 11)   # forward (+x)
SCAN_Y = np.linspace(-0.3, 0.3, 7)    # lateral
SCAN_SHAPE = (len(SCAN_X), len(SCAN_Y))
SCAN_DIM = SCAN_X.size * SCAN_Y.size
PROPRIO_DIM = 3 + 3 + 12 + 12 + 12 + 2  # ang_vel, proj_gravity, dof_pos, dof_vel, last_action, gait clock
HEADING_DIM = 2

_ONLY_GROUP0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # rays hit terrain/floor only


def yaw_of(quat: np.ndarray) -> float:
    """Yaw (rad) from a (w, x, y, z) quaternion."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def proprioception(robot: Go2, last_action: np.ndarray, gait_phase: float = 0.0) -> np.ndarray:
    """gait_phase in [0, 1): the gait clock. Exposing (sin, cos) of it lets the policy TIME its
    footfalls to the clock — required for the phase-offset (dog-like trot) gait reward."""
    two_pi_p = 2.0 * np.pi * gait_phase
    return np.concatenate([
        robot.base_ang_vel,
        robot.projected_gravity,
        robot.joint_pos - DEFAULT_STANCE,
        0.05 * robot.joint_vel,
        last_action,
        [np.sin(two_pi_p), np.cos(two_pi_p)],
    ]).astype(np.float32)


def scandots(model: mujoco.MjModel, data: mujoco.MjData, robot: Go2,
             clip: float = 1.0) -> np.ndarray:
    """Privileged terrain heightmap around the base: (SCAN_DIM,) of (base_z - ground_z), clipped.

    Rays are cast straight down (only hitting terrain group 0, so the robot's own geoms are ignored).
    """
    base = robot.base_pos
    yaw = yaw_of(robot.base_quat)
    c, s = np.cos(yaw), np.sin(yaw)
    geomid = np.zeros(1, dtype=np.int32)
    vec = np.array([0.0, 0.0, -1.0])
    out = np.empty(SCAN_DIM, dtype=np.float32)
    k = 0
    for gx in SCAN_X:
        for gy in SCAN_Y:
            wx = base[0] + c * gx - s * gy
            wy = base[1] + s * gx + c * gy
            pnt = np.array([wx, wy, base[2] + 1.0])
            dist = mujoco.mj_ray(model, data, pnt, vec, _ONLY_GROUP0, 1, -1, geomid)
            if dist < 0:                      # no ground found below
                out[k] = clip
            else:
                ground_z = pnt[2] - dist
                out[k] = np.clip(base[2] - ground_z, -clip, clip)
            k += 1
    return out


def oracle_heading(robot: Go2, goal_xy) -> np.ndarray:
    """Unit vector (cos, sin) of the heading error toward the goal, in the base frame."""
    base = robot.base_pos
    desired = np.arctan2(goal_xy[1] - base[1], goal_xy[0] - base[0])
    rel = desired - yaw_of(robot.base_quat)
    rel = np.arctan2(np.sin(rel), np.cos(rel))   # wrap to [-pi, pi]
    return np.array([np.cos(rel), np.sin(rel)], dtype=np.float32)
