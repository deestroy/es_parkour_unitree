"""Evaluate the ES-Parkour student: success rate (Fig. 5), motor energy (Table IV),
ANN-vs-SNN theoretical energy (Tables II/III), and lighting robustness (Table V).

Run:  python3 scripts/evaluate.py --student outputs/student.pt --teacher outputs/teacher.pt --episodes 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.envs import TERRAIN_KINDS  # noqa: E402
from es_parkour.models.student_snn import StudentSNN  # noqa: E402
from es_parkour.models.teacher import TeacherActorCritic  # noqa: E402
from es_parkour.sensors import EventSimulator  # noqa: E402
from es_parkour.eval.evaluate import evaluate_all, rollout_student  # noqa: E402
from es_parkour.eval.energy import energy_report  # noqa: E402

OUT = REPO / "outputs"


def load_student(path):
    ck = torch.load(path, map_location="cpu")
    hw = tuple(ck.get("event_hw", (48, 64)))
    st = StudentSNN(base_channels=ck.get("base_channels", 16), event_hw=hw, T=ck.get("T", 4))
    st.load_state_dict(ck["model"])
    st.eval()
    return st, hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=str(OUT / "student.pt"))
    ap.add_argument("--teacher", default=str(OUT / "teacher.pt"))
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--difficulty", type=float, default=0.2)
    ap.add_argument("--lighting", action="store_true", help="also run Table V lighting conditions")
    args = ap.parse_args()

    student, hw = load_student(args.student)

    # ---- Success rate (Fig. 5) + motor energy (Table IV) ----
    print("\n=== Per-terrain success rate & motor energy (Fig. 5 / Table IV analog) ===")
    res = evaluate_all(student, kinds=TERRAIN_KINDS, episodes=args.episodes, difficulty=args.difficulty,
                       depth_hw=hw)
    print(f"{'terrain':10s} {'success':>8s} {'progress':>9s} {'motor_mJ':>9s}")
    for k, r in res.items():
        print(f"{k:10s} {r['success_rate']*100:7.1f}% {r['progress_frac']*100:8.1f}% "
              f"{r['motor_energy_mJ']:9.2f}")

    # ---- Same-architecture ANN vs SNN energy (Tables II/III, Eq. 10) ----
    print("\n=== Student network as ANN vs SNN — operations & theoretical energy (Tables II/III) ===")
    ev = EventSimulator()
    from es_parkour.envs import ParkourEnv, ParkourConfig
    env = ParkourEnv(ParkourConfig(kinds=["hurdle"], difficulty=args.difficulty,
                                   render_depth=True, depth_hw=hw))
    o = env.reset(kind="hurdle")
    # a REAL event frame: render two views a step apart so neurons actually spike
    d0 = o["depth"].copy()
    o, _, _, _ = env.step(np.zeros(12, dtype=np.float32))
    ev_ch = torch.as_tensor(ev.channels(ev.diff(o["depth"], d0)), dtype=torch.float32).unsqueeze(0)
    pr = torch.as_tensor(o["proprio"], dtype=torch.float32).unsqueeze(0)
    env.close()

    rep = energy_report(student, ev_ch, pr)
    print(f"  ANN FLOPs (1 pass, all MAC):     {rep['ann_flops']:.3e}")
    print(f"  SNN ops (MAC first*T + SOPs):    {rep['snn_ops']:.3e}   (T={rep['T']})")
    print(f"  SNN SOPs (spike-driven ACs):     {rep['snn_sops']:.3e}")
    print(f"  SNN mean firing rate:            {rep['mean_firing_rate']*100:.1f}%")
    print(f"  OPs(SNN)/OPs(ANN):               {rep['ops_ratio_snn_over_ann']:.3f} : 1")
    print(f"  E_ANN = {rep['E_ann_J']*1e6:.4f} uJ   E_SNN = {rep['E_snn_J']*1e6:.4f} uJ")
    print(f"  >>> Energy saving: {rep['energy_saving_pct']:.1f}%   (paper target ~88%)")

    # ---- Lighting robustness (Table V) ----
    if args.lighting:
        print("\n=== Lighting robustness (Table V analog) — success under brightness scaling ===")
        for name, b in [("normal", 1.0), ("overexposed", 1.8), ("underexposed", 0.35)]:
            r = rollout_student(student, "hurdle", episodes=args.episodes,
                                difficulty=args.difficulty, depth_hw=hw, brightness=b)
            print(f"  {name:13s} (brightness={b:>4}): success={r['success_rate']*100:5.1f}% "
                  f"progress={r['progress_frac']*100:5.1f}%")


if __name__ == "__main__":
    main()
