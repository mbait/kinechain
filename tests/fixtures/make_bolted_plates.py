"""Build a two-plate bolted fixture and export it as STEP.

Geometry (all units mm):
    base: 60 x 40 x 5 plate (z in [0, 5]) with four integral studs of radius 3.0
          and length 5, on a 40 x 24 bolt pattern, extruding up through the top plate.
    top:  60 x 40 x 5 plate (z in [5, 10]) with four clearance holes of radius 3.05
          on the same pattern.

The stud/hole pairs are coaxial cylinders — superficially revolute-like — but their
axial overlap (5) is shorter than their diameter (~6.1), and the plates share a large
face-to-face planar contact whose normal is parallel to every stud axis. Both signals
must steer the classifier to FIXED, never revolute.
"""

from pathlib import Path

import cadquery as cq

PLATE_X, PLATE_Y, PLATE_T = 60.0, 40.0, 5.0
STUD_RADIUS = 3.0
HOLE_RADIUS = 3.05  # slight clearance
PATTERN = [(x, y) for x in (-20.0, 20.0) for y in (-12.0, 12.0)]


def build_assembly() -> cq.Assembly:
    base = cq.Workplane("XY").box(PLATE_X, PLATE_Y, PLATE_T, centered=(True, True, False))
    for x, y in PATTERN:
        base = base.union(
            cq.Workplane("XY", origin=(x, y, PLATE_T)).circle(STUD_RADIUS).extrude(PLATE_T)
        )

    top = cq.Workplane("XY", origin=(0, 0, PLATE_T)).box(
        PLATE_X, PLATE_Y, PLATE_T, centered=(True, True, False)
    )
    for x, y in PATTERN:
        top = top.cut(
            cq.Workplane("XY", origin=(x, y, PLATE_T)).circle(HOLE_RADIUS).extrude(PLATE_T)
        )

    asm = cq.Assembly(name="bolted")
    asm.add(base, name="base", color=cq.Color(0.7, 0.7, 0.7))
    asm.add(top, name="top", color=cq.Color(0.9, 0.5, 0.3))
    return asm


def export(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    build_assembly().export(str(path), exportType="STEP")
    return path


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "bolted_plates.step"
    export(out)
    print(f"wrote {out}")
