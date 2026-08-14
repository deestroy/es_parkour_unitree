"""Privileged teacher actor-critic (ANN).

Matches the paper's teacher structure: a scandot (terrain) encoder produces a latent that is
concatenated with proprioception and the oracle heading, then fed to an MLP actor and critic.
The critic is privileged (same inputs as the actor) — a standard asymmetric-safe choice here since
the teacher always has privileged access.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..envs import obs as O


def mlp(sizes, act=nn.ELU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class TeacherActorCritic(nn.Module):
    def __init__(self, proprio_dim=O.PROPRIO_DIM, scan_dim=O.SCAN_DIM,
                 heading_dim=O.HEADING_DIM, action_dim=12,
                 scan_latent=32, hidden=(256, 256), init_log_std=-1.0):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.scan_dim = scan_dim
        self.heading_dim = heading_dim
        self.action_dim = action_dim

        self.scan_encoder = mlp([scan_dim, 128, scan_latent])
        feat = scan_latent + proprio_dim + heading_dim
        self.actor = mlp([feat, *hidden, action_dim])
        self.critic = mlp([feat, *hidden, 1])
        self.log_std = nn.Parameter(init_log_std * torch.ones(action_dim))
        # Clamp the action std so it cannot drift upward across training and blow up the policy
        # (unbounded log_std growth caused a late-training collapse). Also floors exploration.
        self.log_std_min, self.log_std_max = -2.5, -1.0

    def _std(self) -> torch.Tensor:
        return self.log_std.clamp(self.log_std_min, self.log_std_max).exp()

    # obs is the concatenated teacher_obs vector: [proprio | scandots | heading]
    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        p = obs[..., :self.proprio_dim]
        s = obs[..., self.proprio_dim:self.proprio_dim + self.scan_dim]
        h = obs[..., self.proprio_dim + self.scan_dim:]
        return torch.cat([self.scan_encoder(s), p, h], dim=-1)

    def forward(self, obs: torch.Tensor):
        feat = self._features(obs)
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> Normal:
        mean, _ = self.forward(obs)
        return Normal(mean, self._std())

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic=False):
        feat = self._features(obs)
        mean = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        if deterministic:
            return mean, None, value
        d = Normal(mean, self._std())
        a = d.sample()
        return a, d.log_prob(a).sum(-1), value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        feat = self._features(obs)
        mean = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        d = Normal(mean, self._std())
        return d.log_prob(action).sum(-1), d.entropy().sum(-1), value

    @torch.no_grad()
    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (used as the distillation target for the student)."""
        return self.actor(self._features(obs))
