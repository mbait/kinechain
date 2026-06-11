"""Unit tests for contact-stage geometry: projected overlap area and tolerances."""

from __future__ import annotations

import numpy as np
import pytest

from kinechain.contact import Tolerances, _projected_overlap_area
from kinechain.surfaces import PlaneFace


def _square_face(side: float, origin: tuple[float, float] = (0.0, 0.0)) -> PlaneFace:
    """An axis-aligned square face at z=0 with normal +Z, triangulated as two triangles."""
    x0, y0 = origin
    p1 = (x0, y0, 0.0)
    p2 = (x0 + side, y0, 0.0)
    p3 = (x0 + side, y0 + side, 0.0)
    p4 = (x0, y0 + side, 0.0)
    return PlaneFace(
        point=np.array(p1),
        normal=np.array([0.0, 0.0, 1.0]),
        area=side * side,
        triangles=np.array([[p1, p2, p3], [p1, p3, p4]]),
    )


def test_identical_squares_overlap_fully() -> None:
    a, b = _square_face(10.0), _square_face(10.0)
    assert _projected_overlap_area(a, b) == pytest.approx(100.0)


def test_half_offset_squares() -> None:
    a, b = _square_face(10.0), _square_face(10.0, origin=(5.0, 0.0))
    assert _projected_overlap_area(a, b) == pytest.approx(50.0)


def test_quarter_offset_squares() -> None:
    a, b = _square_face(10.0), _square_face(10.0, origin=(5.0, 5.0))
    assert _projected_overlap_area(a, b) == pytest.approx(25.0)


def test_disjoint_coplanar_squares_overlap_zero() -> None:
    # The case min(area_a, area_b) got wrong: coplanar but laterally separate.
    a, b = _square_face(10.0), _square_face(10.0, origin=(20.0, 0.0))
    assert _projected_overlap_area(a, b) == pytest.approx(0.0)


def test_contained_square() -> None:
    a, b = _square_face(10.0), _square_face(4.0, origin=(3.0, 3.0))
    assert _projected_overlap_area(a, b) == pytest.approx(16.0)


def test_tolerances_from_diagonal_scale_linearly() -> None:
    small, large = Tolerances.from_diagonal(10.0), Tolerances.from_diagonal(1000.0)
    assert large.pos == pytest.approx(100 * small.pos)
    assert large.radius == pytest.approx(100 * small.radius)
    assert large.min_axial_overlap == pytest.approx(100 * small.min_axial_overlap)
    # Areas scale quadratically.
    assert large.min_plane_overlap_area == pytest.approx(100**2 * small.min_plane_overlap_area)
    assert large.min_joint_plane_area == pytest.approx(100**2 * small.min_joint_plane_area)
    # Angular tolerance is dimensionless and fixed.
    assert large.angle_cos == small.angle_cos
