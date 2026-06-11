"""Unit tests for the joint-classification rules on synthetic contact features.

These bypass CAD entirely: contact features are constructed directly, which makes
rule boundaries (fixed vs. prismatic vs. rejected) cheap to pin down exactly.
"""

from __future__ import annotations

import numpy as np

from kinechain.contact import CoaxialCylinderPair, CoincidentPlanePair, PartContact
from kinechain.joints import JointType, classify
from kinechain.surfaces import CylinderFace, PlaneFace


def _plane(normal) -> PlaneFace:
    n = np.array(normal, dtype=float)
    return PlaneFace(
        point=np.zeros(3),
        normal=n / np.linalg.norm(n),
        area=1000.0,
        triangles=np.zeros((0, 3, 3)),
    )


def _plane_pair(normal, overlap_area: float = 1000.0) -> CoincidentPlanePair:
    return CoincidentPlanePair(
        a=_plane(normal),
        b=_plane(-np.array(normal, dtype=float)),
        antiparallel=True,
        overlap_area=overlap_area,
    )


def _cyl_pair(axis, radius: float = 3.0, overlap: float = 20.0) -> CoaxialCylinderPair:
    d = np.array(axis, dtype=float)
    d /= np.linalg.norm(d)

    def face() -> CylinderFace:
        return CylinderFace(
            axis_point=np.zeros(3), axis_dir=d, radius=radius, axial_extent=overlap, area=100.0
        )

    return CoaxialCylinderPair(a=face(), b=face(), axial_overlap=overlap)


def _contact(cylinders=(), planes=()) -> PartContact:
    return PartContact(i=0, j=1, coaxial_cylinders=list(cylinders), coincident_planes=list(planes))


def test_two_nonparallel_planes_yield_prismatic() -> None:
    j = classify(_contact(planes=[_plane_pair([0, 0, 1]), _plane_pair([0, 1, 0])]))
    assert j is not None and j.type is JointType.PRISMATIC
    assert abs(float(np.dot(j.axis_dir, [1.0, 0.0, 0.0]))) > 0.999


def test_cylinder_perpendicular_to_slide_blocks_prismatic() -> None:
    # Slide direction would be X, but a long coaxial cylinder along Y pins the parts
    # to its axis line. Revolute is also locked (a contact normal parallels the
    # cylinder axis), so the pair must end up unjoined.
    contact = _contact(
        cylinders=[_cyl_pair([0, 1, 0])],
        planes=[_plane_pair([0, 0, 1]), _plane_pair([0, 1, 0])],
    )
    assert classify(contact) is None


def test_cylinder_along_slide_keeps_prismatic() -> None:
    # A cylindrical guideway parallel to the slide is consistent with sliding.
    contact = _contact(
        cylinders=[_cyl_pair([1, 0, 0])],
        planes=[_plane_pair([0, 0, 1]), _plane_pair([0, 1, 0])],
    )
    j = classify(contact)
    assert j is not None and j.type is JointType.PRISMATIC


def test_normals_spanning_3d_yield_fixed() -> None:
    contact = _contact(
        planes=[_plane_pair([1, 0, 0]), _plane_pair([0, 1, 0]), _plane_pair([0, 0, 1])]
    )
    j = classify(contact)
    assert j is not None and j.type is JointType.FIXED


def test_plane_plus_bolt_yields_fixed() -> None:
    # Bolt-like: overlap (4) below diameter (6).
    contact = _contact(
        cylinders=[_cyl_pair([0, 0, 1], radius=3.0, overlap=4.0)],
        planes=[_plane_pair([0, 0, 1])],
    )
    j = classify(contact)
    assert j is not None and j.type is JointType.FIXED


def test_single_plane_alone_is_unjoined() -> None:
    # One contact plane = a 3-DOF planar pair; not a joint kinechain emits.
    assert classify(_contact(planes=[_plane_pair([0, 0, 1])])) is None


def test_small_planes_are_ignored() -> None:
    # Sliver contacts below min_joint_plane_area must not license prismatic.
    contact = _contact(
        planes=[_plane_pair([0, 0, 1], overlap_area=5.0), _plane_pair([0, 1, 0], overlap_area=5.0)]
    )
    assert classify(contact) is None
