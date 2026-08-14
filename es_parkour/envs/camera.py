"""Head-mounted depth camera for the Go2, backed by a persistent ``mujoco.Renderer``.

Returns metric depth (metres) from the "head" camera added to ``go2.xml``. The depth frame is the
input to the event-camera simulation (:mod:`es_parkour.sensors.event_sim`) for the SNN student.
"""
from __future__ import annotations

import numpy as np
import mujoco


class DepthCamera:
    def __init__(self, model: mujoco.MjModel, height: int = 60, width: int = 80,
                 camera: str = "head", far_clip: float = 6.0):
        self.model = model
        self.height = height
        self.width = width
        self.camera = camera
        self.far_clip = far_clip
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render_depth(self, data: mujoco.MjData) -> np.ndarray:
        """(H, W) float32 depth in metres, clipped to ``far_clip`` (far/sky -> far_clip)."""
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self.camera)
        depth = self._renderer.render().astype(np.float32)
        self._renderer.disable_depth_rendering()
        depth = np.nan_to_num(depth, nan=self.far_clip, posinf=self.far_clip)
        return np.clip(depth, 0.0, self.far_clip)

    def render_rgb(self, data: mujoco.MjData) -> np.ndarray:
        self._renderer.update_scene(data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        try:
            self._renderer.close()
        except Exception:
            pass
