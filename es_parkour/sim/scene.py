"""Compose a MuJoCo scene (Go2 robot + terrain) and load it into an MjModel.

The Go2 model (``go2.xml``) uses ``<compiler meshdir="assets">`` and its meshes live in
``<go2 dir>/assets``. To keep those relative paths (and the ``<include>``) resolvable, we write the
generated scene next to ``go2.xml`` and load it with ``from_xml_path``. Terrain geometry is injected
as a raw XML fragment produced by :mod:`es_parkour.envs.terrain`.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco

# macOS offscreen rendering: GLFW context works with a display present. Allow override.
os.environ.setdefault("MUJOCO_GL", "glfw")

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
GO2_DIR = REPO_ROOT / "third_party" / "unitree_mujoco" / "unitree_robots" / "go2"
GO2_XML = GO2_DIR / "go2.xml"
# Per-process scene file: parallel workers each own one, so concurrent writes never clobber a file
# another process is compiling (the include + meshdir="assets" still resolve from the go2 dir).
_GENERATED = GO2_DIR / f"_es_generated_scene_{os.getpid()}.xml"

# Default simulation timestep for the Go2 model is 0.002 s (no <option timestep> in go2.xml).
SIM_DT = 0.002

_SCENE_TEMPLATE = """<mujoco model="es_parkour_go2">
  <include file="go2.xml"/>
  <option timestep="{sim_dt}" cone="elliptic" impratio="100"/>
  <statistic center="0 0 0.1" extent="0.8"/>
  <visual>
    <headlight diffuse="{hl} {hl} {hl}" ambient="{amb} {amb} {amb}" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.0" dir="0 0 -1" directional="true" diffuse="{light} {light} {light}"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
{terrain}
  </worldbody>
</mujoco>
"""


def build_scene(terrain_xml: str = "", *, sim_dt: float = SIM_DT,
                brightness: float = 1.0) -> Path:
    """Write a scene file that includes the Go2 model plus ``terrain_xml`` and return its path.

    ``brightness`` (0.2..2.0) scales the lighting so the same scene can be rendered under
    normal / over- / under-exposed conditions (paper's lighting robustness study, Table V).
    """
    b = float(max(0.05, brightness))
    xml = _SCENE_TEMPLATE.format(
        sim_dt=sim_dt,
        terrain=terrain_xml,
        light=min(1.0, 0.9 * b),
        hl=min(1.0, 0.6 * b),
        amb=min(1.0, 0.3 * b),
    )
    _GENERATED.write_text(xml)
    return _GENERATED


def load_model(terrain_xml: str = "", **kwargs) -> mujoco.MjModel:
    """Build the scene and return a compiled :class:`mujoco.MjModel`."""
    path = build_scene(terrain_xml, **kwargs)
    return mujoco.MjModel.from_xml_path(str(path))
