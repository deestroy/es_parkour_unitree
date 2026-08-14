"""Distill the privileged ANN teacher into the spiking event-camera student.

Run:  python3 scripts/distill.py --teacher outputs/teacher.pt --samples 2000 --epochs 3
Saves student -> outputs/student.pt and a loss curve -> outputs/distill_loss.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.train.distill import collect_dataset, train_student  # noqa: E402

OUT = REPO / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=str(OUT / "teacher.pt"))
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--difficulty", type=float, default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--out", default=str(OUT / "student.pt"))
    args = ap.parse_args()

    print(f"collecting {args.samples} teacher transitions from {args.teacher} ...")
    ds = collect_dataset(args.teacher, n_samples=args.samples, difficulty=args.difficulty)
    print(f"dataset: events={tuple(ds.events.shape)} proprio={tuple(ds.proprio.shape)}")

    student, hist = train_student(ds, epochs=args.epochs, batch=args.batch,
                                  T=args.T, base_channels=args.base_channels)

    torch.save({"model": student.state_dict(), "T": args.T,
                "base_channels": args.base_channels,
                "event_hw": (ds.events.shape[2], ds.events.shape[3])}, args.out)
    print(f"saved student -> {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot(hist, marker="o")
        plt.xlabel("epoch"); plt.ylabel("distill loss (MSE action + yaw)")
        plt.title("ANN->SNN distillation loss")
        plt.grid(True); plt.savefig(OUT / "distill_loss.png", dpi=110)
        print(f"saved {OUT/'distill_loss.png'}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
