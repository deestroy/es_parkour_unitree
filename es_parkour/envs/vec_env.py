"""Multiprocessing vectorized ParkourEnv (N parallel workers) for faster CPU PPO.

Uses the ``spawn`` start method — required for MuJoCo + native libs on macOS (``fork`` is unsafe there).
Each worker owns its own ParkourEnv and auto-resets on episode end, so the main loop always gets a live
observation for every env.
"""
from __future__ import annotations

from typing import List, Optional
import multiprocessing as mp

import numpy as np

from .parkour_env import ParkourEnv, ParkourConfig
from . import obs as O


def _worker(remote, cfg: ParkourConfig, seed: int):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # one math thread per worker
    env = ParkourEnv(cfg, seed=seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(env.reset(kind=data))
            elif cmd == "step":
                o, r, d, info = env.step(data)
                if d:
                    o = env.reset()            # auto-reset; done flag still reported for GAE
                remote.send((o, r, d, info))
            elif cmd == "set_difficulty":
                env.set_difficulty(data)
                remote.send(None)
            elif cmd == "set_kind_difficulties":
                env.set_kind_difficulties(data)
                remote.send(None)
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError, BrokenPipeError, OSError):
        pass          # main process went away; shut down quietly
    finally:
        env.close()
        try:
            remote.close()
        except Exception:
            pass


class VecParkourEnv:
    ACTION_DIM = ParkourEnv.ACTION_DIM

    def __init__(self, cfg: ParkourConfig, n_envs: int = 6, seed: int = 0):
        self.n_envs = n_envs
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.procs = []
        for i, wr in enumerate(self.work_remotes):
            p = ctx.Process(target=_worker, args=(wr, cfg, seed + 1000 * i), daemon=True)
            p.start()
            self.procs.append(p)
        for wr in self.work_remotes:
            wr.close()
        self.teacher_obs_dim = O.PROPRIO_DIM + O.SCAN_DIM + O.HEADING_DIM

    def reset(self) -> List[dict]:
        for r in self.remotes:
            r.send(("reset", None))
        return [r.recv() for r in self.remotes]

    def step(self, actions: np.ndarray):
        for r, a in zip(self.remotes, actions):
            r.send(("step", np.asarray(a, dtype=np.float32)))
        results = [r.recv() for r in self.remotes]
        obs = [x[0] for x in results]
        rews = np.array([x[1] for x in results], dtype=np.float32)
        dones = np.array([x[2] for x in results], dtype=np.float32)
        infos = [x[3] for x in results]
        return obs, rews, dones, infos

    def set_difficulty(self, d: float):
        for r in self.remotes:
            r.send(("set_difficulty", float(d)))
        for r in self.remotes:
            r.recv()

    def set_kind_difficulties(self, mapping: dict):
        for r in self.remotes:
            r.send(("set_kind_difficulties", dict(mapping)))
        for r in self.remotes:
            r.recv()

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for r in self.remotes:
            try:
                r.close()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
