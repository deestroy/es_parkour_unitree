"""MJX go/no-go benchmark: step N parallel Go2 envs on the GPU and measure throughput.

Loads our composed scene (Go2 + a hurdle terrain), converts to MJX, vmaps physics stepping across
N environments, and times it. This is the scaling decision number: compare against the CPU
pipeline's ~1,700 steps/s (14 workers x ~120 steps/s).

Run (in the JAX venv): python scripts/mjx_benchmark.py [n_envs ...]
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from es_parkour.sim.scene import build_scene  # noqa: E402
from es_parkour.envs.terrain import make_terrain  # noqa: E402

print("jax backend:", jax.default_backend(), jax.devices())

terr = make_terrain("hurdle", 0.3)
path = build_scene(terr.xml)
# MJX doesn't implement cylinder collisions: swap the Go2's cylinder collision geoms for capsules
# (the same adaptation DeepMind's official MJX Go2 model makes). Patch a copy of go2.xml and point
# the generated scene at it.
go2 = path.parent / "go2.xml"
go2_mjx = path.parent / "go2_mjx_patched.xml"
go2_mjx.write_text(go2.read_text().replace('type="cylinder"', 'type="capsule"'))
scene_mjx = path.parent / (path.stem + "_mjx.xml")
scene_mjx.write_text(path.read_text().replace('include file="go2.xml"',
                                              'include file="go2_mjx_patched.xml"'))
model = mujoco.MjModel.from_xml_path(str(scene_mjx))
model.opt.iterations = 8          # MJX-friendly solver settings
model.opt.ls_iterations = 8
mx = mjx.put_model(model)
print(f"model in MJX: nq={model.nq} nv={model.nv} ncon-capable")

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
dx0 = mjx.put_data(model, data)


def make_batch(n):
    return jax.vmap(lambda _: dx0)(jnp.arange(n))


@jax.jit
def step_batch(dx):
    return jax.vmap(lambda d: mjx.step(mx, d))(dx)


for n_envs in [int(a) for a in (sys.argv[1:] or ["512", "2048", "4096"])]:
    dx = make_batch(n_envs)
    t0 = time.time()
    dx = step_batch(dx)
    jax.block_until_ready(dx.qpos)
    compile_s = time.time() - t0

    iters = 50
    t0 = time.time()
    for _ in range(iters):
        dx = step_batch(dx)
    jax.block_until_ready(dx.qpos)
    dt = (time.time() - t0) / iters
    sps = n_envs / dt
    print(f"n_envs={n_envs:5d}: {dt*1000:7.1f} ms/step-batch -> {sps:,.0f} env-steps/s "
          f"(compile {compile_s:.0f}s)")
print("CPU pipeline reference: ~1,700 env-steps/s (14 workers)")
