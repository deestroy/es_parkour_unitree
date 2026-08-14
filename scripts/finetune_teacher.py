"""Adaptive-curriculum staged fine-tune of a pre-trained teacher.

Loads a checkpoint and continues PPO, but each terrain's difficulty is adjusted *independently* from
its own recent success rate: raise it when success clears an upper threshold, back off when it drops
below a lower one. This keeps every terrain in the "learnable but challenging" band and lets the robot
climb toward harder obstacles only as fast as it can handle — the fix for the too-fast fixed ramp that
collapsed the first run.

Run:  python3 scripts/finetune_teacher.py --ckpt outputs/teacher2_best.pt --steps 800000
Saves outputs/teacher_ft.pt (+ _best.pt) and logs per-terrain difficulty as it advances.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.envs import ParkourConfig  # noqa: E402
from es_parkour.envs.vec_env import VecParkourEnv  # noqa: E402
from es_parkour.models.teacher import TeacherActorCritic  # noqa: E402
from es_parkour.train.ppo import PPO, PPOConfig  # noqa: E402

OUT = REPO / "outputs"
# gap listed twice = oversampled (the user's priority + hardest to learn); flat for gait maintenance
KINDS = ["flat", "gap", "gap", "step", "hurdle", "parkour"]
ADAPT_KINDS = ["gap", "step", "hurdle", "parkour"]        # flat ignores difficulty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "teacher2_best.pt"))
    ap.add_argument("--steps", type=int, default=800_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)          # gentler for fine-tuning
    ap.add_argument("--start-diff", type=float, default=0.0)   # per-terrain starting difficulty
    ap.add_argument("--cap", type=float, default=0.6)          # max difficulty
    ap.add_argument("--up", type=float, default=0.7)           # RAISE difficulty when mean progress > this
    ap.add_argument("--step-size", type=float, default=0.05)
    ap.add_argument("--adapt-every", type=int, default=6)      # iters between adjustments
    ap.add_argument("--min-samples", type=int, default=20)     # episodes needed to trust a rate
    ap.add_argument("--out", default=str(OUT / "teacher_ft.pt"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    model = TeacherActorCritic(); model.load_state_dict(ck["model"])
    print(f"loaded teacher from {args.ckpt}", flush=True)

    # per-terrain difficulty: gap starts continuous (0); others start modest since the seed already
    # handles them. flat has no difficulty. All ratchet UP from here as progress clears the threshold.
    diffs = {"gap": args.start_diff, "step": 0.1, "hurdle": 0.1, "parkour": 0.1, "flat": 0.0}

    env = VecParkourEnv(ParkourConfig(kinds=KINDS, difficulty=args.start_diff, render_depth=False),
                        n_envs=args.n_envs, seed=0)
    env.set_kind_difficulties(diffs)

    ppo = PPO(env, model, PPOConfig(total_steps=args.steps, n_steps=args.n_steps, lr=args.lr))
    ppo.norm.mean = ck["norm_mean"].copy(); ppo.norm.var = ck["norm_var"].copy(); ppo.norm.count = 1e5

    best_path = str(Path(args.out).with_name(Path(args.out).stem + "_best.pt"))
    state = {"it": 0, "best": -1e9}

    def save(path):
        torch.save({"model": model.state_dict(), "norm_mean": ppo.norm.mean, "norm_var": ppo.norm.var,
                    "obs_dim": env.teacher_obs_dim, "action_dim": env.ACTION_DIM,
                    "kinds": KINDS, "kind_difficulty": dict(diffs)}, path)

    def on_log(steps, mret, msuc):
        state["it"] += 1
        if state["it"] % args.adapt_every == 0:
            changed = False
            prog = {}
            for k in ADAPT_KINDS:
                dq = ppo.ep_progress_by_kind.get(k)
                prog[k] = float(np.mean(dq)) if dq else 0.0
                # monotonic ratchet: only raise, and only when the robot clears most of the course
                if dq and len(dq) >= args.min_samples and prog[k] >= args.up and diffs[k] < args.cap:
                    diffs[k] = round(min(args.cap, diffs[k] + args.step_size), 3)
                    changed = True
                    dq.clear()                       # re-measure progress at the new difficulty
            if changed:
                env.set_kind_difficulties(diffs)
            print(f"    [curriculum] diff/prog " +
                  " ".join(f'{k}:{diffs[k]:.2f}/{prog[k]*100:.0f}%' for k in ADAPT_KINDS), flush=True)
        if steps > 0.3 * args.steps and mret == mret and mret > state["best"]:
            state["best"] = mret; save(best_path)
        if state["it"] % 10 == 0:
            save(args.out)

    print(f"adaptive fine-tune (ratchet-up): kinds={KINDS} start={args.start_diff} cap={args.cap} "
          f"raise-when-progress>{args.up} step={args.step_size}", flush=True)
    try:
        ppo.train(on_log=on_log)
    finally:
        save(args.out)
        env.close()
    print(f"saved fine-tuned teacher -> {args.out}  (best -> {best_path})", flush=True)


if __name__ == "__main__":
    main()
