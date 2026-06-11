"""Scale-relative tolerance test: the same hinge at 1/100 scale.

At 0.01x, the hinge bore's axial overlap is 0.4 mm — below the absolute default
min_axial_overlap of 1 mm, so the default tolerances find no contact at all.
Tolerances.from_diagonal derives thresholds from the assembly size and recovers
the revolute joint.
"""

from __future__ import annotations

from pathlib import Path

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Pnt, gp_Trsf

from kinechain.contact import Tolerances, assembly_diagonal, find_contacts
from kinechain.io_step import load_step
from kinechain.joints import JointType, classify_all
from kinechain.surfaces import extract_surfaces

SCALE = 0.01


def _scaled_hinge(hinge_step: Path):
    model = load_step(hinge_step)
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0, 0, 0), SCALE)
    shapes = [BRepBuilderAPI_Transform(p.shape, trsf, True).Shape() for p in model.parts]
    # Mesh deflection must scale with the geometry, like everything else.
    surfaces = [extract_surfaces(s, mesh_deflection=0.5 * SCALE) for s in shapes]
    return shapes, surfaces


def test_default_tolerances_miss_tiny_hinge(hinge_step: Path) -> None:
    shapes, surfaces = _scaled_hinge(hinge_step)
    contacts = find_contacts(shapes, surfaces)  # absolute defaults
    assert classify_all(contacts) == []


def test_scale_relative_tolerances_recover_tiny_hinge(hinge_step: Path) -> None:
    shapes, surfaces = _scaled_hinge(hinge_step)
    tol = Tolerances.from_diagonal(assembly_diagonal(shapes))
    contacts = find_contacts(shapes, surfaces, tol)
    joints = classify_all(contacts, tol)
    assert len(joints) == 1
    assert joints[0].type is JointType.REVOLUTE
