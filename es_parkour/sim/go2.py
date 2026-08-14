"""Unitree Go2 robot interface: joint/actuator maps, PD control, and state accessors.

Two ordering conventions exist in ``go2.xml`` and we resolve both by *name* via ``mj_name2id``:
  * qpos/qvel joint order follows the body tree  (FL, FR, RL, RR) x (hip, thigh, calf)
  * actuator/ctrl order is                       (FR, FL, RR, RL) x (hip, thigh, calf)

We expose everything in one canonical order (:data:`JOINT_ORDER`) so callers never juggle indices.
"""
from __future__ import annotations

import numpy as np
import mujoco

# Canonical joint order used for observations and actions.
JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
# Actuator name for each joint (strip the trailing "_joint").
_ACT_OF_JOINT = {j: j.replace("_joint", "") for j in JOINT_ORDER}

# Default standing pose (from the "home" keyframe): hip 0, thigh 0.9, calf -1.8.
DEFAULT_STANCE = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)

FOOT_GEOMS = ["FL", "FR", "RL", "RR"]


class Go2:
    """Thin stateful wrapper around an ``MjModel``/``MjData`` pair for the Go2."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 kp: float = 60.0, kd: float = 1.5):
        self.model = model
        self.data = data
        self.kp = kp
        self.kd = kd

        # Per-joint addressing.
        self.jnt_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINT_ORDER])
        self.qpos_adr = np.array([model.jnt_qposadr[i] for i in self.jnt_ids])
        self.qvel_adr = np.array([model.jnt_dofadr[i] for i in self.jnt_ids])
        self.act_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, _ACT_OF_JOINT[j])
             for j in JOINT_ORDER])
        self.ctrl_range = model.actuator_ctrlrange[self.act_ids].copy()  # (12, 2)

        # Base freejoint sits at the first qpos slots.
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.foot_geom_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS]
        self.foot_body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{g}_foot") for g in FOOT_GEOMS]

    # ----- state -----------------------------------------------------------------
    @property
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    @property
    def base_quat(self) -> np.ndarray:  # (w, x, y, z)
        return self.data.qpos[3:7].copy()

    @property
    def base_lin_vel_world(self) -> np.ndarray:
        return self.data.qvel[0:3].copy()

    @property
    def base_ang_vel(self) -> np.ndarray:
        return self.data.qvel[3:6].copy()

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.qvel_adr].copy()

    def _rot_world_to_base(self) -> np.ndarray:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.base_quat)
        return R.reshape(3, 3)

    @property
    def base_lin_vel_local(self) -> np.ndarray:
        return self._rot_world_to_base().T @ self.base_lin_vel_world

    @property
    def projected_gravity(self) -> np.ndarray:
        """World -z expressed in the base frame; a level robot reads ~(0, 0, -1)."""
        return self._rot_world_to_base().T @ np.array([0.0, 0.0, -1.0])

    def foot_positions(self) -> np.ndarray:
        """(4, 3) world positions of the feet (FL, FR, RL, RR)."""
        return self.data.xpos[self.foot_body_ids].copy()

    def foot_contacts(self) -> np.ndarray:
        """Boolean (4,) — whether each foot geom is currently in contact."""
        out = np.zeros(4, dtype=bool)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            for k, gid in enumerate(self.foot_geom_ids):
                if con.geom1 == gid or con.geom2 == gid:
                    out[k] = True
        return out

    # ----- control ---------------------------------------------------------------
    def pd_control(self, q_des: np.ndarray) -> np.ndarray:
        """PD torque toward ``q_des`` (12,), clamped to actuator limits, written to ``data.ctrl``."""
        tau = self.kp * (q_des - self.joint_pos) - self.kd * self.joint_vel
        tau = np.clip(tau, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        self.data.ctrl[self.act_ids] = tau
        return tau

    def reset_to_home(self, base_pos=(0.0, 0.0, 0.30), base_quat=(1.0, 0.0, 0.0, 0.0)):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = base_pos
        self.data.qpos[3:7] = base_quat
        self.data.qpos[self.qpos_adr] = DEFAULT_STANCE
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.act_ids] = 0.0
        mujoco.mj_forward(self.model, self.data)
