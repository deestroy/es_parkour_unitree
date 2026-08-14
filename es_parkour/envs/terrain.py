"""Parkour terrain generators for the Go2 MuJoCo scene.

Four terrains matching the paper's Fig. 5 (Gap, Step, Hurdle, Parkour), each laid out as a straight
corridor along +x that the robot must traverse from a start platform to a goal. Every generator takes
a ``difficulty in [0, 1]`` curriculum knob that scales the hard parameter (gap width, step height, ...).

Each generator returns a :class:`Terrain` carrying the MJCF geom fragment (injected into the scene by
:func:`es_parkour.sim.scene.build_scene`), the robot spawn pose, the goal x, and per-terrain metadata.

Geometry uses MuJoCo's half-extent ``size`` convention; helper :func:`_box` takes full sizes and halves
them (mirroring unitree_mujoco's ``terrain_tool`` ``AddBox``). No cv2/noise dependency — box geoms only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

TERRAIN_KINDS = ("gap", "step", "hurdle", "parkour")

# Corridor half-width in y (platforms wide enough the robot won't walk off the side).
HALF_W = 0.75
# Nominal trunk height above the support surface at spawn (Go2 home stance ~0.27 + margin).
SPAWN_CLEARANCE = 0.32

_OBSTACLE_RGBA = "0.55 0.35 0.25 1"
_PLATFORM_RGBA = "0.35 0.45 0.55 1"


@dataclass
class Terrain:
    kind: str
    difficulty: float
    xml: str
    spawn_pos: tuple                      # (x, y, z) base spawn
    goal_x: float                         # x the robot must reach to "succeed"
    support_z: float = 0.0                # height of the support surface at the start
    has_floor: bool = True                # gap terrain floats platforms over a pit
    obstacles: list = field(default_factory=list)  # [(x_center, top_z)] for bookkeeping


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def _box(x, y, z, sx, sy, sz, rgba=_OBSTACLE_RGBA, name=None) -> str:
    """One axis-aligned box geom. (x,y,z) is the CENTER; (sx,sy,sz) are FULL sizes."""
    nm = f' name="{name}"' if name else ""
    return (f'    <geom{nm} type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
            f'size="{sx/2:.4f} {sy/2:.4f} {sz/2:.4f}" rgba="{rgba}"/>')


# --------------------------------------------------------------------------------------------------
def _gap(d: float, rng: np.random.Generator) -> Terrain:
    """Elevated platforms separated by gaps over a pit; robot must cross each gap.

    Platforms are raised well above the robot's standing height so that falling into a gap is a real
    drop (detected by the env as a failure), not a shallow bump.
    """
    # difficulty 0 = zero-width gaps (continuous raised walkway) so the policy learns crossing
    # progressively; platform top (0.4) sits above the robot's trunk height so a real fall is detected.
    plat_h = 0.40                       # platform thickness (top at plat_h, pit floor at 0)
    plat_len = 1.0                      # room to stabilise between gaps
    gap = _lerp(0.0, 0.70, d)           # Fig. 5 "Gap" difficulty: 0 (continuous) -> 0.7 m real leaps
                                        # e.g. d=0.3 -> 21 cm, d=0.5 -> 35 cm, d=1.0 -> 70 cm
    n = 5
    top = plat_h
    geoms, obstacles = [], []
    x = 0.0
    # start platform (a bit longer so the robot can settle before the first gap)
    geoms.append(_box(x, 0, top / 2, 1.3, 2 * HALF_W, plat_h, _PLATFORM_RGBA))
    x += 1.3 / 2
    for i in range(n):
        x += gap + plat_len / 2
        geoms.append(_box(x, 0, top / 2, plat_len, 2 * HALF_W, plat_h, _PLATFORM_RGBA))
        obstacles.append((x, top))
        x += plat_len / 2
    goal_x = x + 0.4
    spawn = (0.0, 0.0, top + SPAWN_CLEARANCE)
    return Terrain("gap", d, "\n".join(geoms), spawn, goal_x, support_z=top,
                   has_floor=True, obstacles=obstacles)


def _step(d: float, rng: np.random.Generator) -> Terrain:
    """Ascending stairs (mirrors terrain_tool.AddStairs)."""
    step_h = _lerp(0.03, 0.20, d)       # Fig. 5 "Step" difficulty (up to 20 cm risers)
    run = 0.32
    n = 6
    geoms, obstacles = [], []
    z = 0.0
    x = 1.0                             # flat run-up on the floor before the stairs
    for i in range(n):
        x += run
        z += step_h
        # each stair is a solid box from the ground up to height z, centered at x
        geoms.append(_box(x, 0, z / 2, run, 2 * HALF_W, z, _PLATFORM_RGBA))
        obstacles.append((x, z))
    goal_x = x + 0.6
    spawn = (0.0, 0.0, SPAWN_CLEARANCE)
    return Terrain("step", d, "\n".join(geoms), spawn, goal_x, support_z=0.0,
                   has_floor=True, obstacles=obstacles)


def _hurdle(d: float, rng: np.random.Generator) -> Terrain:
    """Thin raised bars the robot must step/jump over."""
    hur_h = _lerp(0.08, 0.30, d)        # Fig. 5 "Hurdle" difficulty (up to 30 cm bars)
    thick = 0.06
    spacing = 1.1
    n = 4
    geoms, obstacles = [], []
    x = 1.2
    for i in range(n):
        geoms.append(_box(x, 0, hur_h / 2, thick, 2 * HALF_W, hur_h))
        obstacles.append((x, hur_h))
        x += spacing
    goal_x = x + 0.2
    spawn = (0.0, 0.0, SPAWN_CLEARANCE)
    return Terrain("hurdle", d, "\n".join(geoms), spawn, goal_x, support_z=0.0,
                   has_floor=True, obstacles=obstacles)


def _parkour(d: float, rng: np.random.Generator) -> Terrain:
    """Mixed course: a step up, a hurdle, then a small gap-like block sequence."""
    geoms, obstacles = [], []
    # 1) a single step up
    step_h = _lerp(0.03, 0.14, d)
    x = 1.0
    geoms.append(_box(x, 0, step_h / 2, 0.6, 2 * HALF_W, step_h, _PLATFORM_RGBA))
    obstacles.append((x, step_h))
    # 2) a hurdle
    hur_h = _lerp(0.08, 0.24, d)
    x += 1.1
    geoms.append(_box(x, 0, hur_h / 2, 0.06, 2 * HALF_W, hur_h))
    obstacles.append((x, hur_h))
    # 3) two blocks with a small gap between (step-over)
    blk_h = _lerp(0.06, 0.20, d)
    gap = _lerp(0.0, 0.28, d)
    x += 1.0
    for _ in range(2):
        geoms.append(_box(x, 0, blk_h / 2, 0.5, 2 * HALF_W, blk_h, _PLATFORM_RGBA))
        obstacles.append((x, blk_h))
        x += 0.5 + gap
    goal_x = x + 0.4
    spawn = (0.0, 0.0, SPAWN_CLEARANCE)
    return Terrain("parkour", d, "\n".join(geoms), spawn, goal_x, support_z=0.0,
                   has_floor=True, obstacles=obstacles)


def _flat(d: float, rng: np.random.Generator) -> Terrain:
    """Flat corridor — a walking warm-up (not one of the paper's four terrains)."""
    return Terrain("flat", d, "", (0.0, 0.0, SPAWN_CLEARANCE), goal_x=5.0,
                   support_z=0.0, has_floor=True, obstacles=[])


_GENERATORS = {"flat": _flat, "gap": _gap, "step": _step, "hurdle": _hurdle, "parkour": _parkour}


def make_terrain(kind: str, difficulty: float = 0.0,
                 rng: Optional[np.random.Generator] = None) -> Terrain:
    """Build a :class:`Terrain` of ``kind`` at the given curriculum ``difficulty`` in [0, 1]."""
    if kind not in _GENERATORS:
        raise ValueError(f"unknown terrain '{kind}'; choose from {TERRAIN_KINDS}")
    if rng is None:
        rng = np.random.default_rng()
    return _GENERATORS[kind](float(difficulty), rng)
