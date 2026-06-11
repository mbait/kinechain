"""Build a four-bar linkage fixture and export it as STEP.

Topology: ground — crank — coupler — rocker — ground, a closed kinematic loop of
four revolute joints with parallel Z axes at

    A = (0, 0)    ground–crank        B = (60, 0)   ground–rocker
    C = (10, 40)  crank–coupler       D = (50, 40)  rocker–coupler

Geometry (all units mm): each link is a 5 mm thick bar (rectangle between two
round bosses) in one of three Z layers, with 2 mm air gaps between layers so no
plane-to-plane contact exists anywhere — every joint is evidenced purely by a
pin/hole coaxial cylinder pair:

    ground:        z in [0, 5],  integral pins (r=2) at A and B rising to z=12
    crank, rocker: z in [7, 12], holes (r=2.05) at A / B, integral pins at C / D
                   rising to z=19
    coupler:       z in [14, 19], holes at C and D

Pin engagement is 5 mm against a 4 mm pin diameter, so the bolt-aspect guard does
not reject, and the absence of shoulder contact leaves only revolute. The expected
output is four revolute joints whose graph contains one cycle: three tree joints
plus one loop-closure joint.
"""

import math
from pathlib import Path

import cadquery as cq

THICKNESS = 5.0
LAYER_GAP = 2.0
PIN_RADIUS = 2.0
HOLE_RADIUS = 2.05  # slight clearance
BOSS_RADIUS = 6.0
BAR_WIDTH = 12.0
GROUND_WIDTH = 16.0  # ground is the widest bar so volume-based root selection picks it

A = (0.0, 0.0)
B = (60.0, 0.0)
C = (10.0, 40.0)
D = (50.0, 40.0)

Z_GROUND = 0.0
Z_MID = THICKNESS + LAYER_GAP          # 7  — crank and rocker layer
Z_TOP = 2 * (THICKNESS + LAYER_GAP)    # 14 — coupler layer


def _bar(p: tuple, q: tuple, width: float, z0: float) -> cq.Workplane:
    """A link body: rectangle from p to q with round bosses at both ends."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx))
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    body = (
        cq.Workplane("XY", origin=(mx, my, z0))
        .box(length, width, THICKNESS, centered=(True, True, False))
        .rotate((mx, my, 0), (mx, my, 1), angle)
    )
    for x, y in (p, q):
        body = body.union(
            cq.Workplane("XY", origin=(x, y, z0)).circle(BOSS_RADIUS).extrude(THICKNESS)
        )
    return body


def _pin(at: tuple, z0: float) -> cq.Workplane:
    """An integral joint pin spanning from the top of layer z0 through the next layer."""
    length = THICKNESS + LAYER_GAP + THICKNESS  # reach the far face of the next layer
    return (
        cq.Workplane("XY", origin=(at[0], at[1], z0 + THICKNESS))
        .circle(PIN_RADIUS)
        .extrude(length)
    )


def _hole(at: tuple, z0: float) -> cq.Workplane:
    return (
        cq.Workplane("XY", origin=(at[0], at[1], z0)).circle(HOLE_RADIUS).extrude(THICKNESS)
    )


def build_assembly() -> cq.Assembly:
    ground = _bar(A, B, GROUND_WIDTH, Z_GROUND).union(_pin(A, Z_GROUND)).union(_pin(B, Z_GROUND))
    crank = _bar(A, C, BAR_WIDTH, Z_MID).cut(_hole(A, Z_MID)).union(_pin(C, Z_MID))
    rocker = _bar(B, D, BAR_WIDTH, Z_MID).cut(_hole(B, Z_MID)).union(_pin(D, Z_MID))
    coupler = _bar(C, D, BAR_WIDTH, Z_TOP).cut(_hole(C, Z_TOP)).cut(_hole(D, Z_TOP))

    asm = cq.Assembly(name="fourbar")
    asm.add(ground, name="ground", color=cq.Color(0.7, 0.7, 0.7))
    asm.add(crank, name="crank", color=cq.Color(0.9, 0.4, 0.3))
    asm.add(coupler, name="coupler", color=cq.Color(0.3, 0.6, 0.9))
    asm.add(rocker, name="rocker", color=cq.Color(0.4, 0.8, 0.4))
    return asm


def export(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    build_assembly().export(str(path), exportType="STEP")
    return path


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "fourbar.step"
    export(out)
    print(f"wrote {out}")
