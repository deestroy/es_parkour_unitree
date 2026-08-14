"""Train the privileged PPO teacher with N parallel envs + a walking->parkour curriculum.

Run:  python3 scripts/train_teacher.py --steps 1500000 --n-envs 6 --difficulty 0.3 --curriculum
Saves outputs/teacher.pt (weights + obs-normalizer stats) and appends progress to outputs/teacher_train.log
"""
from __future__ import annotations

import os
# Pin every process (this one + spawned workers, which inherit env) to a single math thread so N
# workers run one-per-core instead of each spawning 8 threads and thrashing an 8-core machine.
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
OUT.mkdir(exist_ok=True)

# "flat" appears twice so ~1/3 of episodes are flat-ground walking practice throughout training.
DEFAULT_KINDS = ["flat", "flat", "gap", "step", "hurdle", "parkour"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--episode-seconds", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kinds", nargs="*", default=DEFAULT_KINDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "teacher.pt"))
    ap.add_argument("--curriculum", action="store_true",
                    help="ramp difficulty 0 -> --difficulty over the first 60%% of training")
    ap.add_argument("--save-every", type=int, default=20, help="save checkpoint every N iters")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    env_cfg = ParkourConfig(kinds=args.kinds, difficulty=(0.0 if args.curriculum else args.difficulty),
                            episode_seconds=args.episode_seconds, render_depth=False)
    env = VecParkourEnv(env_cfg, n_envs=args.n_envs, seed=args.seed)
    model = TeacherActorCritic()
    ppo = PPO(env, model, PPOConfig(total_steps=args.steps, n_steps=args.n_steps, lr=args.lr))

    state = {"it": 0, "best": -1e9}
    best_path = str(Path(args.out).with_name(Path(args.out).stem + "_best.pt"))

    def save(path):
        torch.save({"model": model.state_dict(), "norm_mean": ppo.norm.mean,
                    "norm_var": ppo.norm.var, "obs_dim": env.teacher_obs_dim,
                    "action_dim": env.ACTION_DIM, "kinds": args.kinds,
                    "difficulty": args.difficulty}, path)

    def on_log(steps, mret, msuc):
        state["it"] += 1
        if args.curriculum:
            frac = min(1.0, steps / (0.6 * max(1, args.steps)))
            env.set_difficulty(frac * args.difficulty)
        # keep the best-so-far policy (once past the early curriculum warmup)
        if steps > 0.4 * args.steps and mret == mret and mret > state["best"]:
            state["best"] = mret
            save(best_path)
        if state["it"] % args.save_every == 0:
            save(args.out)

    print(f"training teacher: n_envs={args.n_envs} kinds={args.kinds} "
          f"difficulty={args.difficulty} curriculum={args.curriculum} steps={args.steps}", flush=True)
    try:
        ppo.train(on_log=on_log)
    finally:
        save(args.out)
        env.close()
    print(f"saved teacher -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
