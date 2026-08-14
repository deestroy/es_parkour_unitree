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

from es_parkour.train.distill import collect_dataset, train_student, DistillDataset  # noqa: E402

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
    ap.add_argument("--dagger-rounds", type=int, default=0,
                    help="after the BC warm-up, N rounds of: student drives, teacher labels, retrain")
    ap.add_argument("--out", default=str(OUT / "student.pt"))
    args = ap.parse_args()

    import torch as _t
    device = args.device or ("cuda" if _t.cuda.is_available() else "cpu")

    print(f"[warm-up] collecting {args.samples} teacher-driven transitions from {args.teacher} ...")
    ds = collect_dataset(args.teacher, n_samples=args.samples, difficulty=args.difficulty)
    print(f"dataset: events={tuple(ds.events.shape)} proprio={tuple(ds.proprio.shape)}")

    student, hist = train_student(ds, epochs=args.epochs, batch=args.batch,
                                  T=args.T, base_channels=args.base_channels, device=device)

    # DAGGER rounds: the student's own rollouts (with teacher labels) are aggregated into the
    # dataset so training covers the states the student actually reaches — the paper's
    # "further interaction and optimization of the student" phase.
    for r in range(args.dagger_rounds):
        n_r = max(1000, args.samples // 2)
        print(f"[dagger {r+1}/{args.dagger_rounds}] collecting {n_r} student-driven transitions ...")
        student.eval()
        ds_r = collect_dataset(args.teacher, n_samples=n_r, difficulty=args.difficulty,
                               driver_student=student, device=device)
        ds = DistillDataset(
            torch.cat([ds.events, ds_r.events]), torch.cat([ds.proprio, ds_r.proprio]),
            torch.cat([ds.action, ds_r.action]), torch.cat([ds.heading, ds_r.heading]))
        student.train()
        student, h2 = train_student(ds, epochs=max(2, args.epochs - 2), batch=args.batch,
                                    T=args.T, base_channels=args.base_channels,
                                    device=device, init_student=student)
        hist += h2

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
