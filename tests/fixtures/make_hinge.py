"""Build a minimal two-part hinge fixture and export it as STEP.

Geometry (all units mm):
    bracket: 40 x 20 x 20 block, centered on origin, with a through-hole
             of radius 4 along the X axis at the geometric center.
    pin:     a cylinder of radius 3.95 (slight clearance), length 50,
             axis along X, coaxial with the bracket hole.

The pin protrudes past both faces of the bracket, so there is no planar
shoulder contact — the joint can only be inferred as revolute.
"""

from pathlib import Path

import cadquery as cq

HOLE_RADIUS = 4.0
PIN_RADIUS = 3.95
BRACKET_X, BRACKET_Y, BRACKET_Z = 40.0, 20.0, 20.0
PIN_LENGTH = 50.0


def build_assembly() -> cq.Assembly:
    block = cq.Workplane("XY").box(BRACKET_X, BRACKET_Y, BRACKET_Z)
    hole = (
        cq.Workplane("YZ")
        .workplane(offset=-BRACKET_X / 2)
        .circle(HOLE_RADIUS)
        .extrude(BRACKET_X)
    )
    bracket = block.cut(hole)

    pin = (
        cq.Workplane("YZ")
        .workplane(offset=-PIN_LENGTH / 2)
        .circle(PIN_RADIUS)
        .extrude(PIN_LENGTH)
    )

    asm = cq.Assembly(name="hinge")
    asm.add(bracket, name="bracket", color=cq.Color(0.7, 0.7, 0.7))
    asm.add(pin, name="pin", color=cq.Color(0.3, 0.6, 0.9))
    return asm


def export(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    build_assembly().export(str(path), exportType="STEP")
    return path


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "hinge.step"
    export(out)
    print(f"wrote {out}")
