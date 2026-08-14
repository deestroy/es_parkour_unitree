"""Compact vectorized PPO for the privileged Go2 parkour teacher (N parallel envs, torch/CPU).

Dependency-free (no SB3/gymnasium). GAE, clipped surrogate, value + entropy terms, running obs
normalization. Collects from a :class:`VecParkourEnv` (N workers) which massively improves CPU
throughput and exploration diversity over a single env.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from ..models.teacher import TeacherActorCritic


@dataclass
class PPOConfig:
    total_steps: int = 1_500_000
    n_steps: int = 512           # rollout length PER ENV between updates
    lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatch: int = 1024
    ent_coef: float = 0.0      # let the policy std shrink as it finds good actions (walk, not flail)
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    log_every: int = 1
    # KL-adaptive learning rate (rsl_rl-style). Fixed lr=1e-3 caused a reproducible late-training
    # collapse (~1.2M steps): with 5 reuse epochs, updates go destructively off-policy once the
    # policy is good. Shrink lr when the update KL overshoots, grow it when it undershoots.
    desired_kl: float = 0.01
    lr_min: float = 1e-5
    lr_max: float = 1e-2


class RunningNorm:
    def __init__(self, dim, eps=1e-4):
        self.mean = np.zeros(dim, np.float64); self.var = np.ones(dim, np.float64); self.count = eps

    def update(self, x):
        bm, bv, bn = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean; tot = self.count + bn
        self.mean += delta * bn / tot
        self.var = (self.var * self.count + bv * bn + delta ** 2 * self.count * bn / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


class PPO:
    def __init__(self, vec_env, model: TeacherActorCritic, cfg: PPOConfig = None):
        self.env = vec_env
        self.E = vec_env.n_envs
        self.model = model
        self.cfg = cfg or PPOConfig()
        self.opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        self.lr = self.cfg.lr
        self.last_kl = 0.0
        self.obs_dim = vec_env.teacher_obs_dim
        self.adim = vec_env.ACTION_DIM
        self.norm = RunningNorm(self.obs_dim)
        self.ep_returns = deque(maxlen=100)
        self.ep_success = deque(maxlen=100)
        self.ep_by_kind = {}
        self.ep_progress_by_kind = {}                 # per-terrain mean progress (x/goal_x)
        self.cur_obs = self.env.reset()               # list of E dicts
        self.ep_ret = np.zeros(self.E, np.float32)

    def _flat(self, obs_list):
        from ..envs import ParkourEnv
        return np.stack([ParkourEnv.teacher_obs(o) for o in obs_list]).astype(np.float32)

    def collect(self):
        N, E = self.cfg.n_steps, self.E
        obs_b = np.zeros((N, E, self.obs_dim), np.float32)
        act_b = np.zeros((N, E, self.adim), np.float32)
        logp_b = np.zeros((N, E), np.float32)
        rew_b = np.zeros((N, E), np.float32)
        val_b = np.zeros((N, E), np.float32)
        done_b = np.zeros((N, E), np.float32)

        obs = self._flat(self.cur_obs)
        for t in range(N):
            a, logp, v = self.model.act(torch.as_tensor(self.norm(obs), dtype=torch.float32))
            a = a.numpy(); logp = logp.numpy(); v = v.numpy()
            nxt, r, d, infos = self.env.step(a)
            obs_b[t] = obs; act_b[t] = a; logp_b[t] = logp
            rew_b[t] = r; val_b[t] = v; done_b[t] = d
            self.ep_ret += r
            for i, info in enumerate(infos):
                if d[i]:
                    self.ep_returns.append(float(self.ep_ret[i]))
                    self.ep_success.append(1.0 if info["success"] else 0.0)
                    self.ep_by_kind.setdefault(info["kind"], deque(maxlen=60)).append(
                        1.0 if info["success"] else 0.0)
                    self.ep_progress_by_kind.setdefault(info["kind"], deque(maxlen=60)).append(
                        float(info["x"]) / max(1e-6, float(info["goal_x"])))
                    self.ep_ret[i] = 0.0
            obs = self._flat(nxt)
        self.cur_obs = nxt

        with torch.no_grad():
            last_v = self.model.act(
                torch.as_tensor(self.norm(obs), dtype=torch.float32))[2].numpy()
        self.norm.update(obs_b.reshape(-1, self.obs_dim))
        return obs_b, act_b, logp_b, rew_b, val_b, done_b, last_v

    def gae(self, rew, val, done, last_v):
        N, E = rew.shape
        adv = np.zeros((N, E), np.float32)
        gae = np.zeros(E, np.float32)
        for t in reversed(range(N)):
            next_v = last_v if t == N - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + self.cfg.gamma * next_v * nonterm - val[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * nonterm * gae
            adv[t] = gae
        return adv, adv + val

    def update(self, obs, act, logp_old, adv, ret):
        cfg = self.cfg
        obs = obs.reshape(-1, self.obs_dim); act = act.reshape(-1, self.adim)
        logp_old = logp_old.reshape(-1); adv = adv.reshape(-1); ret = ret.reshape(-1)
        obs_t = torch.as_tensor(self.norm(obs), dtype=torch.float32)
        act_t = torch.as_tensor(act, dtype=torch.float32)
        logp_old_t = torch.as_tensor(logp_old, dtype=torch.float32)
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        n = obs_t.shape[0]; idx = np.arange(n)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, cfg.minibatch):
                b = idx[s:s + cfg.minibatch]
                logp, ent, v = self.model.evaluate(obs_t[b], act_t[b])
                # KL-adaptive lr (approx KL of the update so far on this minibatch)
                with torch.no_grad():
                    log_ratio = logp - logp_old_t[b]
                    kl = float(((log_ratio.exp() - 1.0) - log_ratio).mean())
                    if kl > 2.0 * cfg.desired_kl:
                        self.lr = max(cfg.lr_min, self.lr / 1.5)
                    elif kl < cfg.desired_kl / 2.0 and kl >= 0.0:
                        self.lr = min(cfg.lr_max, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                ratio = (logp - logp_old_t[b]).exp()
                s1 = ratio * adv_t[b]
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t[b]
                loss = (-torch.min(s1, s2).mean()
                        + cfg.vf_coef * 0.5 * (v - ret_t[b]).pow(2).mean()
                        - cfg.ent_coef * ent.mean())
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
        self.last_kl = kl
        return float(-torch.min(s1, s2).mean()), float((v - ret_t[b]).pow(2).mean()), float(ent.mean())

    def train(self, on_log=None):
        cfg = self.cfg
        steps, it = 0, 0
        per_iter = cfg.n_steps * self.E
        while steps < cfg.total_steps:
            obs, act, logp, rew, val, done, last_v = self.collect()
            adv, ret = self.gae(rew, val, done, last_v)
            pi_l, v_l, ent = self.update(obs, act, logp, adv, ret)
            steps += per_iter; it += 1
            mret = np.mean(self.ep_returns) if self.ep_returns else float("nan")
            msuc = np.mean(self.ep_success) if self.ep_success else 0.0
            if it % cfg.log_every == 0:
                by = {k: round(np.mean(v), 2) for k, v in self.ep_by_kind.items()}
                print(f"[{steps:8d}] ret={mret:7.2f} succ={msuc:4.2f} pi={pi_l:+.3f} "
                      f"vf={v_l:.3f} ent={ent:.2f} lr={self.lr:.1e} kl={self.last_kl:.4f} "
                      f"by_kind={by}", flush=True)
            if on_log:
                on_log(steps, mret, msuc)
        return self.model
