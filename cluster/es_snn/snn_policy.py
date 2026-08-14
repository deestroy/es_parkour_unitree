"""SNN student policy for the go2_distill pipeline (rsl_rl-style interface).

Wraps the validated StudentSNN (spiking ResNet-18 + GRU + spiking MLP, from the MuJoCo baseline)
behind the recurrent-policy surface the distill runner expects, and provides the paper's
distillation losses (Eq. 8 action MSE + Eq. 9 yaw MSE).

Integration: set this class as the student policy in the go2_distill config (see README.md).
The exact constructor signature may need one-line adaptation to the repo's policy registry.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from es_parkour.models.student_snn import StudentSNN


class SNNStudentPolicy(nn.Module):
    is_recurrent = True

    def __init__(self, proprio_dim: int, action_dim: int = 12, event_hw=(48, 64),
                 base_channels: int = 64, T: int = 4, device: str = "cuda"):
        super().__init__()
        self.net = StudentSNN(proprio_dim=proprio_dim, action_dim=action_dim,
                              event_hw=event_hw, base_channels=base_channels, T=T).to(device)
        self.device = device
        self.hidden = None

    def reset(self, dones: torch.Tensor = None):
        if self.hidden is not None and dones is not None:
            self.hidden[dones.bool()] = 0.0

    def act_inference(self, events: torch.Tensor, proprio: torch.Tensor):
        if self.hidden is None or self.hidden.shape[0] != proprio.shape[0]:
            self.hidden = self.net.init_hidden(proprio.shape[0], device=self.device)
        action, heading, self.hidden = self.net(events, proprio, self.hidden)
        return action, heading

    act = act_inference


def distill_losses(student_actions, teacher_actions, predicted_heading, oracle_heading,
                   w_yaw: float = 1.0):
    """Paper Eq. 8 (per-joint action MSE) + Eq. 9 (yaw MSE)."""
    l_action = nn.functional.mse_loss(student_actions, teacher_actions)
    l_yaw = nn.functional.mse_loss(predicted_heading, oracle_heading)
    return l_action + w_yaw * l_yaw, {"loss_action": float(l_action), "loss_yaw": float(l_yaw)}
