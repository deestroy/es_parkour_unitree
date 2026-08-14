"""Spiking student network (event camera -> actions), matching the paper's Fig. 3 student.

Architecture:
  * Spiking ResNet-18-structured encoder  (event frame -> event latent), [2,2,2,2] basic blocks.
  * Small ANN proprio encoder.
  * GRU fusion of (event latent, proprio latent) -> recurrent hidden state h (temporal memory).
  * Linear "predicted heading" head from h (the student predicts heading; the teacher had it as oracle).
  * 3-layer spiking MLP actor [512, 256, 128] over h + predicted heading -> joint actions.

Spiking dynamics use spikingjelly ``clock_driven`` with T sub-steps (paper T=4). Each spiking sub-net
is reset then run T times on the (static) input; we read the mean spike rate as the analog output.

``base_channels`` controls the ResNet width: 64 = full ResNet-18 (paper), smaller = CPU-friendly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate, functional

from ..envs import obs as O


def make_neuron(kind: str = "LIF"):
    sg = surrogate.ATan()
    if kind.upper() == "IF":
        return neuron.IFNode(surrogate_function=sg, detach_reset=True)
    return neuron.LIFNode(tau=2.0, surrogate_function=sg, detach_reset=True)


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class SpikingBasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, neuron_kind="LIF"):
        super().__init__()
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.sn1 = make_neuron(neuron_kind)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.sn2 = make_neuron(neuron_kind)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.sn2(out + idt)


class SpikingResNetEncoder(nn.Module):
    """ResNet-18 layout ([2,2,2,2]) with spiking neurons; outputs an event latent (spike rate)."""

    def __init__(self, in_ch=2, base=16, latent=128, neuron_kind="LIF"):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(base), make_neuron(neuron_kind))
        self.layer1 = self._make(base, base, 2, 1, neuron_kind)
        self.layer2 = self._make(base, 2 * base, 2, 2, neuron_kind)
        self.layer3 = self._make(2 * base, 4 * base, 2, 2, neuron_kind)
        self.layer4 = self._make(4 * base, 8 * base, 2, 2, neuron_kind)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8 * base, latent)

    def _make(self, cin, cout, n, stride, nk):
        blocks = [SpikingBasicBlock(cin, cout, stride, nk)]
        for _ in range(n - 1):
            blocks.append(SpikingBasicBlock(cout, cout, 1, nk))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SpikingMLPActor(nn.Module):
    """3-layer spiking MLP [512, 256, 128] + linear readout on the last layer's spike rate."""

    def __init__(self, in_dim, action_dim=12, sizes=(512, 256, 128), neuron_kind="LIF"):
        super().__init__()
        layers, d = [], in_dim
        for s in sizes:
            layers += [nn.Linear(d, s), make_neuron(neuron_kind)]
            d = s
        self.body = nn.Sequential(*layers)
        self.readout = nn.Linear(sizes[-1], action_dim)

    def forward(self, x):
        return self.body(x)

    def decode(self, rate):
        return self.readout(rate)


class StudentSNN(nn.Module):
    def __init__(self, proprio_dim=O.PROPRIO_DIM, action_dim=12, event_hw=(60, 80),
                 base_channels=16, event_latent=128, proprio_latent=64, gru_hidden=128,
                 T=4, neuron_kind="LIF"):
        super().__init__()
        self.T = T
        self.gru_hidden = gru_hidden
        self.encoder = SpikingResNetEncoder(2, base_channels, event_latent, neuron_kind)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 128), nn.ELU(), nn.Linear(128, proprio_latent), nn.ELU())
        self.gru = nn.GRUCell(event_latent + proprio_latent, gru_hidden)
        self.heading_head = nn.Linear(gru_hidden, 2)
        self.actor = SpikingMLPActor(gru_hidden + 2, action_dim, neuron_kind=neuron_kind)

    def init_hidden(self, batch, device=None):
        return torch.zeros(batch, self.gru_hidden, device=device)

    def _spiking_rate(self, module, x):
        """Reset ``module`` and run it T times on static input x; return mean spike rate."""
        functional.reset_net(module)
        acc = 0.0
        for _ in range(self.T):
            acc = acc + module(x)
        return acc / self.T

    def forward(self, event_ch, proprio, h):
        """One control step. event_ch:(B,2,H,W) proprio:(B,P) h:(B,gru_hidden).

        Returns action:(B,12), predicted_heading:(B,2), new hidden h:(B,gru_hidden).
        """
        event_latent = self._spiking_rate(self.encoder, event_ch)
        p = self.proprio_encoder(proprio)
        h = self.gru(torch.cat([event_latent, p], dim=-1), h)
        heading = self.heading_head(h)
        rate = self._spiking_rate(self.actor, torch.cat([h, heading], dim=-1))
        action = self.actor.decode(rate)
        return action, heading, h
