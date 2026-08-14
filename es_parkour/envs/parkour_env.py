"""ParkourEnv: a gym-style Go2 parkour environment over the four terrains with a difficulty curriculum.

Deliberately *not* a ``gym.Env`` subclass — we avoid the gym/gymnasium split and keep a minimal
``reset()``/``step()`` returning a dict observation so both the privileged teacher and the vision
student can consume what they need:

    obs = {
        "proprio":  (42,)   proprioception               (teacher + student)
        "scandots": (77,)   privileged terrain heightmap  (teacher only)
        "heading":  (2,)    oracle heading to goal        (teacher only)
        "depth":    (H, W)  head-camera depth, if enabled (student only)
    }

Actions are 12-dim residuals in [-1, 1] around the default stance, applied with PD control at a
control rate of 50 Hz (decimation 10 over the 500 Hz sim).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import mujoco

from ..sim import load_model, Go2
from ..sim.scene import SIM_DT
from ..sim.go2 import DEFAULT_STANCE
from . import obs as O
from .camera import DepthCamera
from .terrain import make_terrain, TERRAIN_KINDS, Terrain


@dataclass
class ParkourConfig:
    kinds: List[str] = field(default_factory=lambda: list(TERRAIN_KINDS))
    difficulty: float = 0.0
    control_dt: float = 0.02          # 50 Hz policy
    episode_seconds: float = 8.0
    # Per-joint action authority (rad, residual around default stance), pattern (abduction, thigh,
    # calf) per leg. Thigh/calf get large excursions — a running stride swings the hip and folds the
    # knee through big arcs, and stepping onto obstacles needs high leg lifts. The old uniform 0.3 rad
    # (~17 deg) could only produce a shuffle: joints barely moved and obstacle climbs were impossible.
    action_scale: tuple = (0.35, 0.85, 0.85) * 4
    render_depth: bool = False        # enable head-camera depth in the obs (student rollouts)
    depth_hw: tuple = (60, 80)
    brightness: float = 1.0           # lighting scale (Table V robustness study)
    # reward weights (forward velocity dominates; alive/heading are small shaping terms)
    w_progress: float = 2.5           # x-velocity reward, capped at v_target
    v_target: float = 0.9             # m/s target forward speed (a running pace, not a stroll)
    w_alive: float = 0.08
    w_heading: float = 0.05
    w_torque: float = 2e-4
    w_actrate: float = 0.05          # stronger smoothing: less leg jitter
    w_angvel: float = 0.02
    w_upright: float = 0.2            # lighter, so "sit tilted but alive" is not a stable optimum
    w_airtime: float = 0.5           # reward taking steps (feet air time); lowered - it favored hops
    airtime_target: float = 0.3      # s; reward air times around this
    # anti-hop / gait-regularization (discourage bounding, encourage a grounded trot)
    w_vz: float = 0.6                # penalize base vertical velocity (bouncing)
    w_allfly: float = 0.5            # penalize all-four-feet-airborne (the flight phase of a hop)
    # phase-clock gait reward: each foot must match its scheduled stance/swing on a shared clock.
    # Trot offsets put FL/FR (and RL/RR) in ANTI-phase, so the front legs never land together —
    # the timing structure of a dog's running gait.
    w_gait: float = 0.6
    # anti-drag: reward swing-phase feet for lifting toward this height above local ground.
    # Without it, a 1 mm ground-skim counts as a "swing" and the rear legs learn to drag.
    w_clearance: float = 0.5
    clearance_target: float = 0.08   # m
    gait_freq: float = 1.5           # Hz, step cycle frequency (1.8 looked frantic; calmer dog trot)
    gait_offsets: tuple = (0.0, 0.5, 0.5, 0.0)   # phase offset per foot (FL, FR, RL, RR) = trot
    gait_duty: float = 0.6           # fraction of the cycle each foot should be in stance
    w_goal: float = 10.0
    # termination
    tip_grav_z: float = -0.4          # projected gravity z above this => tipped over
    collapse_clearance: float = 0.14  # trunk-to-ground clearance below this => collapsed
    corridor_halfwidth: float = 0.7   # |y| beyond this => out of corridor => episode FAILS.
                                      # Obstacles span |y|<=0.75; without this bound the policy
                                      # walks AROUND the course on open floor instead of over it.
    w_drift: float = 0.3              # penalty per |y| drift from the corridor center line
    pit_margin: float = 0.05          # base_z below (start_support - margin) => fell in a gap
                                      # (gap platforms top at 0.4; a fall drops the base to ~0.32)


class ParkourEnv:
    ACTION_DIM = 12

    def __init__(self, config: Optional[ParkourConfig] = None, seed: Optional[int] = None):
        self.cfg = config or ParkourConfig()
        self.rng = np.random.default_rng(seed)
        self.decimation = max(1, round(self.cfg.control_dt / SIM_DT))
        self.max_steps = int(self.cfg.episode_seconds / self.cfg.control_dt)

        self.model = None
        self.data = None
        self.robot: Optional[Go2] = None
        self.cam: Optional[DepthCamera] = None
        self.terrain: Optional[Terrain] = None
        self._geomid = np.zeros(1, dtype=np.int32)
        # Compiled-scene cache: terrains are deterministic given (kind, difficulty, brightness),
        # so we compile each once and just reset the data on subsequent episodes. This avoids an
        # expensive XML recompile on every reset (which dominates early, reset-heavy training).
        self._cache = {}
        # Coarse bucket so a continuously-ramped curriculum difficulty doesn't force constant
        # model recompiles (each new bucket compiles all terrain kinds once, then caches).
        self._difficulty_bucket = 0.1
        # Optional per-terrain difficulty (adaptive curriculum). Falls back to cfg.difficulty.
        self.kind_difficulty: dict = {}

    # ----- curriculum ------------------------------------------------------------
    def set_difficulty(self, d: float):
        self.cfg.difficulty = float(np.clip(d, 0.0, 1.0))

    def set_kind_difficulties(self, mapping: dict):
        """Set per-terrain difficulty (adaptive curriculum). Missing kinds use cfg.difficulty."""
        for k, v in mapping.items():
            self.kind_difficulty[k] = float(np.clip(v, 0.0, 1.0))

    # ----- ground height queries (terrain group only) -----------------------------
    def _ground_under_point(self, x: float, y: float, from_z: float = None) -> float:
        z0 = (self.robot.base_pos[2] if from_z is None else from_z) + 1.0
        pnt = np.array([x, y, z0])
        dist = mujoco.mj_ray(self.model, self.data, pnt, np.array([0.0, 0.0, -1.0]),
                             O._ONLY_GROUP0, 1, -1, self._geomid)
        return pnt[2] - dist if dist >= 0 else -np.inf

    def _ground_under_base(self) -> float:
        base = self.robot.base_pos
        return self._ground_under_point(base[0], base[1])

    # ----- reset -----------------------------------------------------------------
    def _get_bundle(self, kind: str, difficulty: float):
        """Return a cached (terrain, model, data, robot, cam) bundle, compiling on first use."""
        d = round(difficulty / self._difficulty_bucket) * self._difficulty_bucket
        key = (kind, round(d, 4), round(self.cfg.brightness, 3))
        if key not in self._cache:
            terrain = make_terrain(kind, d, np.random.default_rng(0))
            model = load_model(terrain.xml, brightness=self.cfg.brightness)
            data = mujoco.MjData(model)
            robot = Go2(model, data)
            cam = DepthCamera(model, *self.cfg.depth_hw) if self.cfg.render_depth else None
            self._cache[key] = (terrain, model, data, robot, cam)
        return self._cache[key]

    def reset(self, kind: Optional[str] = None) -> dict:
        kind = kind or self.rng.choice(self.cfg.kinds)
        d = self.kind_difficulty.get(kind, self.cfg.difficulty)
        self.terrain, self.model, self.data, self.robot, self.cam = \
            self._get_bundle(kind, d)
        self.robot.reset_to_home(base_pos=self.terrain.spawn_pos)
        # settle onto the surface so the first observation is physical
        q0 = DEFAULT_STANCE.copy()
        for _ in range(self.decimation * 3):
            self.robot.pd_control(q0)
            mujoco.mj_step(self.model, self.data)

        self.start_support = float(self.terrain.support_z)
        self.last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self.prev_x = float(self.robot.base_pos[0])
        self.feet_air_time = np.zeros(4, dtype=np.float32)
        self.gait_phase = 0.0
        self.step_count = 0
        self.success = False
        return self._obs()

    # ----- observation -----------------------------------------------------------
    def _obs(self) -> dict:
        o = {
            "proprio": O.proprioception(self.robot, self.last_action, self.gait_phase),
            "scandots": O.scandots(self.model, self.data, self.robot),
            "heading": O.oracle_heading(self.robot, (self.terrain.goal_x, 0.0)),
        }
        if self.cfg.render_depth and self.cam is not None:
            o["depth"] = self.cam.render_depth(self.data)
        return o

    # ----- step ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        q_des = DEFAULT_STANCE + np.asarray(self.cfg.action_scale) * action

        torque_sq = 0.0
        motor_energy = 0.0
        for _ in range(self.decimation):
            tau = self.robot.pd_control(q_des)
            mujoco.mj_step(self.model, self.data)
            torque_sq += float(np.mean(tau ** 2))
            motor_energy += float(np.sum(np.abs(tau * self.robot.joint_vel))) * SIM_DT
        torque_sq /= self.decimation

        x = float(self.robot.base_pos[0])
        vel_x = (x - self.prev_x) / self.cfg.control_dt      # forward speed this control step
        self.prev_x = x

        # feet air time: reward a stepping gait (a foot that just landed after being airborne)
        contact = self.robot.foot_contacts()
        first_contact = (self.feet_air_time > 0) & contact
        self.feet_air_time += self.cfg.control_dt
        airtime_reward = float(np.sum((self.feet_air_time - self.cfg.airtime_target) * first_contact))
        self.feet_air_time *= (~contact)

        # anti-hop gait shaping: penalize vertical bounce + the all-airborne flight phase of a hop
        vz = float(self.robot.base_lin_vel_world[2])
        all_fly = 0.0 if contact.any() else 1.0

        # phase-clock gait reward: advance the shared clock; each foot (FL, FR, RL, RR) should be in
        # stance for the first `gait_duty` fraction of its own (offset) cycle and in swing otherwise.
        # Trot offsets (0, .5, .5, 0) force the front pair anti-phase — no more both-fronts-together.
        self.gait_phase = (self.gait_phase + self.cfg.gait_freq * self.cfg.control_dt) % 1.0
        gait_score = 0.0
        clearance_score = 0.0
        foot_pos = self.robot.foot_positions()
        n_swing = 0
        for i in range(4):
            leg_phase = (self.gait_phase + self.cfg.gait_offsets[i]) % 1.0
            should_stance = leg_phase < self.cfg.gait_duty
            gait_score += 1.0 if bool(contact[i]) == should_stance else -1.0
            if not should_stance:
                # swing phase: the foot must actually LIFT, not skim the ground. Reward height above
                # the local ground up to a target — the anti-drag term (rear feet were dragging).
                n_swing += 1
                gz = self._ground_under_point(foot_pos[i, 0], foot_pos[i, 1])
                h = foot_pos[i, 2] - gz
                clearance_score += min(h, self.cfg.clearance_target) / self.cfg.clearance_target
        gait_score /= 4.0                     # in [-1, 1]
        clearance_score = clearance_score / n_swing if n_swing else 0.0   # in [0, 1]

        # termination checks
        pg = self.robot.projected_gravity
        clearance = self.robot.base_pos[2] - self._ground_under_base()
        tipped = pg[2] > self.cfg.tip_grav_z
        collapsed = clearance < self.cfg.collapse_clearance
        fell_pit = self.robot.base_pos[2] < (self.start_support - self.cfg.pit_margin)
        y = float(self.robot.base_pos[1])
        out_of_corridor = abs(y) > self.cfg.corridor_halfwidth
        self.success = x >= self.terrain.goal_x and not out_of_corridor
        self.step_count += 1
        timeout = self.step_count >= self.max_steps
        done = bool(tipped or collapsed or fell_pit or out_of_corridor or self.success or timeout)

        c = self.cfg
        heading = self.robot_heading_cos()
        reward = (
            c.w_progress * min(vel_x, c.v_target)
            + c.w_alive
            + c.w_heading * heading
            - c.w_torque * torque_sq
            - c.w_actrate * float(np.mean((action - self.last_action) ** 2))
            - c.w_angvel * float(np.sum(self.robot.base_ang_vel[:2] ** 2))
            - c.w_upright * float(np.sum(pg[:2] ** 2))
            + c.w_airtime * airtime_reward
            - c.w_vz * vz * vz
            - c.w_allfly * all_fly
            + c.w_gait * gait_score
            + c.w_clearance * clearance_score
            - c.w_drift * abs(y)
        )
        if self.success:
            reward += c.w_goal
        if tipped or collapsed or fell_pit or out_of_corridor:
            reward -= 1.0

        self.last_action = action
        info = {
            "success": self.success, "tipped": tipped, "collapsed": collapsed,
            "fell_pit": fell_pit, "out_of_corridor": out_of_corridor, "timeout": timeout, "x": x,
            "goal_x": self.terrain.goal_x, "kind": self.terrain.kind,
            "terminal": done and not timeout, "motor_energy": motor_energy,
        }
        return self._obs(), float(reward), done, info

    def robot_heading_cos(self) -> float:
        return float(O.oracle_heading(self.robot, (self.terrain.goal_x, 0.0))[0])

    # ----- convenience for the privileged teacher -------------------------------
    @staticmethod
    def teacher_obs(o: dict) -> np.ndarray:
        return np.concatenate([o["proprio"], o["scandots"], o["heading"]]).astype(np.float32)

    @property
    def teacher_obs_dim(self) -> int:
        return O.PROPRIO_DIM + O.SCAN_DIM + O.HEADING_DIM

    def close(self):
        for _, _, _, _, cam in self._cache.values():
            if cam is not None:
                cam.close()
        self._cache.clear()
        self.cam = None
