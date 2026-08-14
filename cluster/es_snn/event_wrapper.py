"""Batched, GPU-resident depth->event conversion for IsaacGym vectorized envs.

Same physics as the MuJoCo baseline's EventSimulator (paper Eq. 1: threshold crossings of the
log-intensity change, L = -log(depth)), but operating on a (N_envs, H, W) depth tensor per step
with per-env previous-frame state kept on the GPU.
"""
from __future__ import annotations

import torch


class DepthToEvents:
    def __init__(self, n_envs: int, height: int, width: int, threshold_c: float = 0.15,
                 depth_clip=(0.1, 6.0), device="cuda"):
        self.C = threshold_c
        self.dmin, self.dmax = depth_clip
        self.prev_logL = torch.zeros(n_envs, height, width, device=device)
        self.initialized = torch.zeros(n_envs, dtype=torch.bool, device=device)

    def _logL(self, depth: torch.Tensor) -> torch.Tensor:
        return -torch.log(depth.clamp(self.dmin, self.dmax))

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor, env_ids_reset: torch.Tensor = None) -> torch.Tensor:
        """depth: (N, H, W) metres -> events: (N, 2, H, W) [positive, negative] counts.

        Call with ``env_ids_reset`` for envs that were reset this step (their previous frame is
        invalid; they emit zero events for one frame).
        """
        logL = self._logL(depth)
        if env_ids_reset is not None and len(env_ids_reset) > 0:
            self.initialized[env_ids_reset] = False
        dL = logL - self.prev_logL
        events = torch.fix(dL / self.C)
        events[~self.initialized] = 0.0
        self.prev_logL = logL
        self.initialized[:] = True
        return torch.stack([events.clamp(min=0), (-events).clamp(min=0)], dim=1)
