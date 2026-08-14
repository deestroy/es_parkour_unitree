"""ANN -> SNN distillation (paper Fig. 3, Eq. 8-9).

The privileged ANN teacher (scandots + oracle heading) supervises the spiking event-camera student:
    L_action = MSE(student_action, teacher_action)      (Eq. 8)
    L_yaw    = MSE(student_heading, oracle_heading)      (Eq. 9)

We collect a dataset of (event_frame, proprio, teacher_action, oracle_heading) by rolling out the
deterministic teacher, converting consecutive depth frames to events, then train the student with
minibatch SGD. (Recurrent BPTT through the GRU across a whole episode is a Phase-2 upgrade; here the
student is trained per-step with zero-initialised hidden state.)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..envs import ParkourEnv, ParkourConfig, TERRAIN_KINDS
from ..models.teacher import TeacherActorCritic
from ..models.student_snn import StudentSNN
from ..sensors import EventSimulator


def load_teacher(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    model = TeacherActorCritic()
    model.load_state_dict(ck["model"])
    model.eval()
    mean = ck["norm_mean"].astype(np.float32)
    std = np.sqrt(ck["norm_var"] + 1e-8).astype(np.float32)
    return model, mean, std, ck.get("difficulty", 0.2), ck.get("kinds", list(TERRAIN_KINDS))


@dataclass
class DistillDataset:
    events: torch.Tensor      # (N, 2, H, W)
    proprio: torch.Tensor     # (N, 42)
    action: torch.Tensor      # (N, 12) teacher target
    heading: torch.Tensor     # (N, 2)  oracle heading target

    def __len__(self):
        return self.events.shape[0]


@torch.no_grad()
def collect_dataset(teacher_ckpt: str, n_samples: int = 2000, difficulty=None,
                    kinds=None, depth_hw=(48, 64), threshold_c=0.15, seed=0,
                    driver_student=None, device="cpu") -> DistillDataset:
    """Collect (events, proprio, teacher_action, oracle_heading) transitions.

    ``driver_student=None``  -> the TEACHER drives (behaviour-cloning warm-up data).
    ``driver_student=StudentSNN`` -> DAGGER: the STUDENT drives (so the data covers the states the
    student actually visits, including its mistakes) while the teacher labels every state with the
    action it would have taken. This is the paper's "further interaction" phase.
    """
    model, mean, std, ck_diff, ck_kinds = load_teacher(teacher_ckpt)
    difficulty = ck_diff if difficulty is None else difficulty
    kinds = kinds or ck_kinds

    cfg = ParkourConfig(kinds=kinds, difficulty=difficulty, render_depth=True, depth_hw=depth_hw)
    env = ParkourEnv(cfg, seed=seed)
    ev = EventSimulator(threshold_c=threshold_c)

    E, P, A, Hd = [], [], [], []
    o = env.reset()
    prev_depth = o["depth"].copy()
    h = driver_student.init_hidden(1, device=device) if driver_student is not None else None
    while len(E) < n_samples:
        t_obs = (ParkourEnv.teacher_obs(o) - mean) / std
        label = model.action_mean(torch.as_tensor(t_obs, dtype=torch.float32).unsqueeze(0))[0].numpy()

        events = ev.channels(ev.diff(o["depth"], prev_depth))   # (2, H, W)
        E.append(events); P.append(o["proprio"]); A.append(label); Hd.append(o["heading"])

        if driver_student is not None:
            e_t = torch.as_tensor(events, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = torch.as_tensor(o["proprio"], dtype=torch.float32, device=device).unsqueeze(0)
            act_drive, _, h = driver_student(e_t, p_t, h)
            act_drive = act_drive.squeeze(0).cpu().numpy()
        else:
            act_drive = label

        prev_depth = o["depth"].copy()
        o, _, done, _ = env.step(act_drive)
        if done:
            o = env.reset(); prev_depth = o["depth"].copy()
            if driver_student is not None:
                h = driver_student.init_hidden(1, device=device)
    env.close()

    return DistillDataset(
        torch.as_tensor(np.stack(E), dtype=torch.float32),
        torch.as_tensor(np.stack(P), dtype=torch.float32),
        torch.as_tensor(np.stack(A), dtype=torch.float32),
        torch.as_tensor(np.stack(Hd), dtype=torch.float32),
    )


def train_student(ds: DistillDataset, epochs=3, batch=8, lr=1e-3, T=4,
                  base_channels=16, w_yaw=1.0, seed=0, log_every=20, device=None,
                  init_student=None):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training student on device={device} (base_channels={base_channels}, T={T})")
    H, W = ds.events.shape[2], ds.events.shape[3]
    if init_student is not None:
        student = init_student.to(device)          # continue training (DAGGER rounds)
    else:
        student = StudentSNN(base_channels=base_channels, event_hw=(H, W), T=T).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    n = len(ds)
    hist = []
    for ep in range(epochs):
        idx = np.random.permutation(n)
        running = 0.0
        nb = 0
        for start in range(0, n - batch + 1, batch):
            b = idx[start:start + batch]
            ev_b = ds.events[b].to(device); pr_b = ds.proprio[b].to(device)
            a_tgt = ds.action[b].to(device); h_tgt = ds.heading[b].to(device)
            h0 = student.init_hidden(len(b), device=device)
            a_pred, head_pred, _ = student(ev_b, pr_b, h0)
            l_action = nn.functional.mse_loss(a_pred, a_tgt)      # Eq. 8
            l_yaw = nn.functional.mse_loss(head_pred, h_tgt)      # Eq. 9
            loss = l_action + w_yaw * l_yaw
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss); nb += 1
            if nb % log_every == 0:
                print(f"  ep{ep} batch{nb}/{n//batch} loss={running/nb:.4f} "
                      f"(action={float(l_action):.4f} yaw={float(l_yaw):.4f})")
        ep_loss = running / max(1, nb)
        hist.append(ep_loss)
        print(f"epoch {ep}: mean loss={ep_loss:.4f}")
    return student, hist
