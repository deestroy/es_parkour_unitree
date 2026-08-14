"""Energy accounting (Eq. 10, Tables II/III) for a trained SNN student on GPU.

Reuses the validated counter from the MuJoCo baseline: same-architecture ANN-vs-SNN comparison,
E_MAC = 4.6 pJ / E_AC = 0.9 pJ, empirical firing rates via hooks on the spiking neurons.

Usage:  python -m cluster.es_snn.energy_eval --ckpt <student.pt> [--hw 48 64]
"""
from __future__ import annotations

import argparse

import torch

from es_parkour.eval.energy import energy_report
from es_parkour.models.student_snn import StudentSNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=False, help="student checkpoint (state_dict or wrapped)")
    ap.add_argument("--hw", type=int, nargs=2, default=[48, 64])
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--proprio-dim", type=int, default=42)
    args = ap.parse_args()

    net = StudentSNN(proprio_dim=args.proprio_dim, event_hw=tuple(args.hw),
                     base_channels=args.base_channels, T=args.T)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        state = ck.get("model", ck)
        net.load_state_dict(state, strict=False)
    net.eval()

    # representative sparse event input (random polarity map at a realistic event density)
    ev = (torch.rand(1, 2, *args.hw) < 0.05).float()
    pr = torch.randn(1, args.proprio_dim)
    rep = energy_report(net, ev, pr)
    print("=== ANN vs SNN energy (Eq. 10) ===")
    for k, v in rep.items():
        print(f"  {k}: {v:.4g}")


if __name__ == "__main__":
    main()
