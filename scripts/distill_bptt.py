"""BPTT distillation CLI: warm-up on teacher-driven clips, then DAGGER rounds on student-driven
clips (student carries hidden state during collection — the deployment distribution).

Run: python scripts/distill_bptt.py --teacher outputs/teacher_clear_best.pt --clips 300 --dagger-rounds 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.train.distill_bptt import collect_clips, train_student_bptt  # noqa: E402

OUT = REPO / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=str(OUT / "teacher_clear_best.pt"))
    ap.add_argument("--clips", type=int, default=300)
    ap.add_argument("--clip-len", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--difficulty", type=float, default=0.2)
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--out", default=str(OUT / "student_bptt.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[warm-up] collecting {args.clips} teacher-driven clips (K={args.clip_len}) ...", flush=True)
    ds = collect_clips(args.teacher, n_clips=args.clips, clip_len=args.clip_len,
                       difficulty=args.difficulty)
    print(f"clips: {tuple(ds.events.shape)}", flush=True)
    student, hist = train_student_bptt(ds, epochs=args.epochs, batch=args.batch, T=args.T,
                                       base_channels=args.base_channels, device=device)

    for r in range(args.dagger_rounds):
        n_r = max(80, args.clips // 2)
        print(f"[dagger {r+1}/{args.dagger_rounds}] {n_r} student-driven clips ...", flush=True)
        student.eval()
        ds_r = collect_clips(args.teacher, n_clips=n_r, clip_len=args.clip_len,
                             difficulty=args.difficulty, driver_student=student, device=device)
        ds = ds.cat(ds_r)
        student.train()
        student, h2 = train_student_bptt(ds, epochs=max(2, args.epochs - 1), batch=args.batch,
                                         T=args.T, base_channels=args.base_channels,
                                         device=device, init_student=student)
        hist += h2

    torch.save({"model": student.state_dict(), "T": args.T, "base_channels": args.base_channels,
                "event_hw": (ds.events.shape[3], ds.events.shape[4]),
                "bptt_clip_len": args.clip_len}, args.out)
    print(f"saved BPTT student -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
