"""Theoretical energy accounting: the student's network as an ANN vs as an SNN (paper Eq. 10, Tables II/III).

The paper compares the *same* architecture run two ways:
  * ANN: one forward pass, every connection is a MAC ->            E_ANN = E_MAC * FLOPs
  * SNN: T sub-steps, but only spikes trigger accumulate (AC) ops, and the first layer takes the
    real-valued input (MAC).  Per Eq. 10:
        E_SNN = E_MAC * FLOP_firstlayer * T  +  E_AC * (sum of spike-driven SOPs over the rest)
    with  SOP ~= mean_firing_rate * FLOPs_layer * T.

E_MAC = 4.6 pJ, E_AC = 0.9 pJ (45 nm, refs [24-26]). Firing rate is measured empirically on a real
event input via forward hooks on the spiking neurons; a low, sparse firing rate is what makes the SNN
cheap. Numbers therefore depend on how the trained student actually spikes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, functional

E_MAC = 4.6e-12   # J per MAC
E_AC = 0.9e-12    # J per AC
_SPIKING = (neuron.LIFNode, neuron.IFNode, neuron.BaseNode)


def _layer_flops(module: nn.Module, inp_shape, out_shape) -> float:
    if isinstance(module, nn.Conv2d):
        cout, oh, ow = out_shape[1], out_shape[2], out_shape[3]
        cin = module.in_channels // module.groups
        kh, kw = module.kernel_size
        return cout * oh * ow * cin * kh * kw
    if isinstance(module, nn.Linear):
        return module.in_features * module.out_features
    return 0.0


def measure(student, event_ch, proprio) -> dict:
    """Run the student once; return single-pass structural FLOPs (deduped per layer),
    the first-layer FLOPs, and the mean spiking firing rate over the T sub-steps."""
    per_layer = {}          # id(module) -> flops (recorded once = single ANN pass)
    order = []
    fr = {"spikes": 0.0, "elems": 0.0}
    handles = []

    def flop_hook(m, i, o):
        if id(m) not in per_layer:
            per_layer[id(m)] = _layer_flops(m, tuple(i[0].shape), tuple(o.shape))
            order.append(id(m))

    def spike_hook(m, i, o):
        fr["spikes"] += float((o != 0).float().sum())
        fr["elems"] += float(o.numel())

    for m in student.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(flop_hook))
        elif isinstance(m, _SPIKING):
            handles.append(m.register_forward_hook(spike_hook))

    functional.reset_net(student)
    with torch.no_grad():
        student(event_ch, proprio, student.init_hidden(event_ch.shape[0]))
    for h in handles:
        h.remove()

    flops = [per_layer[k] for k in order]
    total_flops = float(sum(flops))
    flop_first = float(flops[0]) if flops else 0.0
    mean_fr = fr["spikes"] / max(1.0, fr["elems"])
    return {"total_flops": total_flops, "flop_first": flop_first,
            "rest_flops": total_flops - flop_first, "mean_fr": mean_fr, "T": student.T}


def energy_report(student, event_ch, proprio) -> dict:
    """Same-architecture ANN-vs-SNN operations and theoretical energy (Tables II/III)."""
    m = measure(student, event_ch, proprio)
    T = m["T"]

    ann_flops = m["total_flops"]                                   # 1 pass, all MAC
    snn_sops = m["mean_fr"] * m["rest_flops"] * T                  # spike-driven ACs
    snn_ops = m["flop_first"] * T + snn_sops                       # MAC(first)*T + SOPs

    e_ann = E_MAC * ann_flops
    e_snn = E_MAC * m["flop_first"] * T + E_AC * snn_sops
    return {
        "ann_flops": ann_flops,
        "snn_ops": snn_ops,
        "snn_sops": snn_sops,
        "mean_firing_rate": m["mean_fr"],
        "T": T,
        "ops_ratio_snn_over_ann": snn_ops / max(1.0, ann_flops),
        "E_ann_J": e_ann,
        "E_snn_J": e_snn,
        "energy_saving_pct": 100.0 * (1.0 - e_snn / max(1e-30, e_ann)),
    }
