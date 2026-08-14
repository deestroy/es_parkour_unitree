"""BPTT sequence distillation: train the student's GRU memory on unrolled clips.

The per-frame distillation trained every sample with a zero hidden state — amnesia — so the GRU
never learned to store anything, and deployment had to reset memory every step. Here we train on
CLIPS of K consecutive frames, carrying the hidden state through the unroll and backpropagating
through the h-chain (truncated BPTT). Gradients flowing from late-step losses back through earlier
GRU updates are what teach the network *what is worth remembering* (e.g., an obstacle that has
left the camera's view but is still under the body).

Deployment then carries h across steps — exactly matching training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..envs import ParkourEnv, ParkourConfig, TERRAIN_KINDS
from ..models.student_snn import StudentSNN
from ..sensors import EventSimulator
from .distill import load_teacher


@dataclass
class ClipDataset:
    events: torch.Tensor    # (N, K, 2, H, W)
    proprio: torch.Tensor   # (N, K, P)
    action: torch.Tensor    # (N, K, 12)  teacher labels
    heading: torch.Tensor   # (N, K, 2)

    def __len__(self):
        return self.events.shape[0]

    def cat(self, other: "ClipDataset") -> "ClipDataset":
        return ClipDataset(*[torch.cat([a, b]) for a, b in zip(
            (self.events, self.proprio, self.action, self.heading),
            (other.events, other.proprio, other.action, other.heading))])


@torch.no_grad()
def collect_clips(teacher_ckpt: str, n_clips: int, clip_len: int = 30, difficulty=None,
                  kinds=None, depth_hw=(48, 64), threshold_c=0.15, seed=0,
                  driver_student=None, device="cpu") -> ClipDataset:
    """Collect K-length clips of (events, proprio, teacher_action, heading), episode-ordered.

    ``driver_student`` None -> teacher drives (warm-up). Otherwise the student drives while
    CARRYING its hidden state (the deployment distribution), and the teacher labels every frame.
    """
    model, mean, std, ck_diff, ck_kinds = load_teacher(teacher_ckpt)
    difficulty = ck_diff if difficulty is None else difficulty
    kinds = kinds or ck_kinds

    env = ParkourEnv(ParkourConfig(kinds=kinds, difficulty=difficulty,
                                   render_depth=True, depth_hw=depth_hw), seed=seed)
    ev = EventSimulator(threshold_c=threshold_c)

    clips = {"E": [], "P": [], "A": [], "H": []}
    while len(clips["E"]) < n_clips:
        o = env.reset()
        prev_depth = o["depth"].copy()
        h = driver_student.init_hidden(1, device=device) if driver_student is not None else None
        ep = {"E": [], "P": [], "A": [], "H": []}
        done = False
        while not done:
            t_obs = (ParkourEnv.teacher_obs(o) - mean) / std
            label = model.action_mean(
                torch.as_tensor(t_obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
            events = ev.channels(ev.diff(o["depth"], prev_depth))
            ep["E"].append(events); ep["P"].append(o["proprio"])
            ep["A"].append(label); ep["H"].append(o["heading"])

            if driver_student is not None:
                a_t, _, h = driver_student(
                    torch.as_tensor(events, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(o["proprio"], dtype=torch.float32, device=device).unsqueeze(0),
                    h)
                act = a_t.squeeze(0).cpu().numpy()
            else:
                act = label
            prev_depth = o["depth"].copy()
            o, _, done, _ = env.step(act)

        # chop the episode into non-overlapping clips (each clip starts where h genuinely was 0
        # only for the first clip; later clips approximate — acceptable for truncated BPTT)
        n_full = len(ep["E"]) // clip_len
        for c in range(n_full):
            s = c * clip_len
            for k in clips:
                clips[k].append(np.stack(ep[k][s:s + clip_len]))
            if len(clips["E"]) >= n_clips:
                break
    env.close()
    return ClipDataset(
        torch.as_tensor(np.stack(clips["E"]), dtype=torch.float32),
        torch.as_tensor(np.stack(clips["P"]), dtype=torch.float32),
        torch.as_tensor(np.stack(clips["A"]), dtype=torch.float32),
        torch.as_tensor(np.stack(clips["H"]), dtype=torch.float32),
    )


def train_student_bptt(ds: ClipDataset, epochs=3, batch=8, lr=1e-3, T=4, base_channels=64,
                       w_yaw=1.0, seed=0, device=None, init_student=None, log_every=10):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = ds.events.shape[1]
    H, W = ds.events.shape[3], ds.events.shape[4]
    if init_student is not None:
        student = init_student.to(device)
    else:
        student = StudentSNN(base_channels=base_channels, event_hw=(H, W), T=T).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    print(f"BPTT training on {device}: {len(ds)} clips x K={K} (base={base_channels}, T={T})")

    n = len(ds)
    hist = []
    for ep in range(epochs):
        idx = np.random.permutation(n)
        running, nb = 0.0, 0
        for s in range(0, n - batch + 1, batch):
            b = idx[s:s + batch]
            ev_b = ds.events[b].to(device)     # (B, K, 2, H, W)
            pr_b = ds.proprio[b].to(device)
            a_b = ds.action[b].to(device)
            hd_b = ds.heading[b].to(device)
            h = student.init_hidden(len(b), device=device)
            loss = 0.0
            for k in range(K):                 # unroll: gradients flow back through the h-chain
                a_pred, head_pred, h = student(ev_b[:, k], pr_b[:, k], h)
                loss = loss + nn.functional.mse_loss(a_pred, a_b[:, k]) \
                            + w_yaw * nn.functional.mse_loss(head_pred, hd_b[:, k])
            loss = loss / K
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            running += float(loss); nb += 1
            if nb % log_every == 0:
                print(f"  ep{ep} clip-batch {nb}/{n//batch} loss={running/nb:.4f}", flush=True)
        hist.append(running / max(1, nb))
        print(f"epoch {ep}: mean clip loss={hist[-1]:.4f}", flush=True)
    return student, hist
