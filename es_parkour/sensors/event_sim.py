"""Event-camera simulation from MuJoCo depth frames.

An event fires where the log-intensity change crosses a threshold C (paper Eq. 1):
    Delta L(u, t_k) = L(u, t_k) - L(u, t_k - dt) = p_k * C
We derive a log-intensity image from depth (closer surfaces are "brighter": L = -log(depth)), so that
obstacle edges approaching the camera produce events — exactly what an event camera sees during motion.

Two synthesis paths:
  * ``diff``  (default): true temporal difference between two rendered depth frames -> the direct Eq. 1
    definition. Robust, needs no ego-motion estimate.
  * ``flow``  (paper Eq. 4, single frame): Delta L ~= -grad(L) . v * dt, where v is the per-pixel optical
    flow (motion field) induced by the camera's ego-motion at each pixel's depth. Lets one depth frame
    synthesize events, matching the paper's method (Fig. 4).

Output conventions:
  * ``events``   : signed per-pixel event count (H, W), positive = brightening, negative = darkening.
  * ``channels`` : (2, H, W) float — [positive, negative] counts, the SNN's input tensor.
  * ``to_rgb``   : (H, W, 3) uint8 red/blue polarity image (Fig. 1 style) for visualization.
"""
from __future__ import annotations

import numpy as np


class EventSimulator:
    def __init__(self, threshold_c: float = 0.15, fovy_deg: float = 70.0,
                 depth_clip=(0.1, 6.0)):
        self.C = float(threshold_c)
        self.fovy = np.deg2rad(fovy_deg)
        self.dmin, self.dmax = depth_clip

    # ----- intensity -------------------------------------------------------------
    def log_intensity(self, depth: np.ndarray) -> np.ndarray:
        d = np.clip(depth, self.dmin, self.dmax)
        return -np.log(d).astype(np.float32)      # near -> bright

    def _quantize(self, dL: np.ndarray) -> np.ndarray:
        """Signed integer number of threshold crossings per pixel."""
        return np.fix(dL / self.C).astype(np.float32)

    # ----- Eq. 1: temporal difference -------------------------------------------
    def diff(self, depth_t: np.ndarray, depth_prev: np.ndarray) -> np.ndarray:
        dL = self.log_intensity(depth_t) - self.log_intensity(depth_prev)
        return self._quantize(dL)

    # ----- Eq. 4: single-frame motion field -------------------------------------
    def flow(self, depth: np.ndarray, cam_lin_vel, cam_ang_vel, dt: float) -> np.ndarray:
        """Synthesize events from one depth frame + camera-frame velocities (Longuet-Higgins/Prazdny).

        ``cam_lin_vel`` (Tx,Ty,Tz), ``cam_ang_vel`` (wx,wy,wz) are in the CAMERA frame (z = optical axis).
        """
        L = self.log_intensity(depth)
        H, W = depth.shape
        f = 0.5 * H / np.tan(0.5 * self.fovy)              # focal length in pixels
        cx, cy = W / 2.0, H / 2.0
        xs = np.arange(W) - cx
        ys = np.arange(H) - cy
        x, y = np.meshgrid(xs, ys)                          # centered pixel coords
        Z = np.clip(depth, self.dmin, self.dmax)

        Tx, Ty, Tz = cam_lin_vel
        wx, wy, wz = cam_ang_vel
        u_dot = (-f * Tx + x * Tz) / Z + (x * y / f) * wx - (f + x * x / f) * wy + y * wz
        v_dot = (-f * Ty + y * Tz) / Z + (f + y * y / f) * wx - (x * y / f) * wy - x * wz

        gy, gx = np.gradient(L)                             # image gradient (per pixel)
        dL = -(gx * u_dot + gy * v_dot) * dt
        return self._quantize(dL)

    # ----- representations -------------------------------------------------------
    @staticmethod
    def channels(events: np.ndarray) -> np.ndarray:
        pos = np.clip(events, 0, None)
        neg = np.clip(-events, 0, None)
        return np.stack([pos, neg], axis=0).astype(np.float32)

    @staticmethod
    def to_rgb(events: np.ndarray) -> np.ndarray:
        """Red = positive, blue = negative, black = no event (matches paper Fig. 1)."""
        H, W = events.shape
        img = np.zeros((H, W, 3), dtype=np.uint8)
        mag = np.clip(np.abs(events), 0, 3) / 3.0
        img[..., 0] = (255 * mag * (events > 0)).astype(np.uint8)   # red for positive
        img[..., 2] = (255 * mag * (events < 0)).astype(np.uint8)   # blue for negative
        return img
