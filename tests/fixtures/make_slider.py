"""Build a V-rail slider fixture and export it as STEP.

Geometry (all units mm):
    rail:     80 x 30 x 10 base block (z in [0, 10]) with a triangular ridge running
              the full length along X: base y in [-5, 5] at z = 10, apex at z = 15.
    carriage: 40 x 30 x 15 block (z in [10, 25]) with a matching triangular groove
              cut along its full length, seated on the rail with the groove mated
              to the ridge.

The contact consists of three face-to-face plane orientations: the two 45° ridge
flanks and the flat seat at z = 10. Their normals are mutually non-parallel but all
perpendicular to X, leaving exactly one free translation — slide along X. There are
no cylindrical surfaces, so only the prismatic rule can fire; the carriage is shorter
than the rail, so no X-normal face contact exists to block the slide.
"""

from pathlib import Path

import cadquery as cq

RAIL_X, RAIL_Y, RAIL_Z = 80.0, 30.0, 10.0
RIDGE_HALF_WIDTH, RIDGE_HEIGHT = 5.0, 5.0
CAR_X, CAR_Y, CAR_Z = 40.0, 30.0, 15.0


def _ridge_prism(length: float, x_start: float) -> cq.Workplane:
    """Triangular prism along X, sitting on the z = RAIL_Z plane."""
    return (
        cq.Workplane("YZ")
        .workplane(offset=x_start)
        .polyline(
            [
                (-RIDGE_HALF_WIDTH, RAIL_Z),
                (RIDGE_HALF_WIDTH, RAIL_Z),
                (0.0, RAIL_Z + RIDGE_HEIGHT),
            ]
        )
        .close()
        .extrude(length)
    )


def build_assembly() -> cq.Assembly:
    rail = (
        cq.Workplane("XY")
        .box(RAIL_X, RAIL_Y, RAIL_Z, centered=(True, True, False))
        .union(_ridge_prism(RAIL_X, -RAIL_X / 2))
    )
    carriage = (
        cq.Workplane("XY", origin=(0, 0, RAIL_Z))
        .box(CAR_X, CAR_Y, CAR_Z, centered=(True, True, False))
        .cut(_ridge_prism(CAR_X, -CAR_X / 2))
    )

    asm = cq.Assembly(name="slider")
    asm.add(rail, name="rail", color=cq.Color(0.7, 0.7, 0.7))
    asm.add(carriage, name="carriage", color=cq.Color(0.4, 0.8, 0.4))
    return asm


def export(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    build_assembly().export(str(path), exportType="STEP")
    return path


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "slider.step"
    export(out)
    print(f"wrote {out}")
