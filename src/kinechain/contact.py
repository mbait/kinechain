"""Contact detection between parts.

Two-phase pipeline:
1. Broad phase — AABB overlap (with slack) eliminates non-touching part pairs.
2. Narrow phase — for each candidate pair, test analytic-surface coincidence.

The narrow phase emits typed "contact features" that the joint classifier consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopoDS import TopoDS_Shape

from .surfaces import CylinderFace, PartSurfaces, PlaneFace


@dataclass(frozen=True)
class Tolerances:
    pos: float = 1.0          # mm — perpendicular axis distance, plane offset, etc.
    angle_cos: float = 0.999  # cos(2.5°) ≈ 0.999; threshold for parallel direction
    radius: float = 0.5       # mm — cylinder radius mismatch tolerance
    min_axial_overlap: float = 1.0      # mm — minimum overlap to count cyl-cyl as contact
    min_plane_overlap_area: float = 1.0 # mm² — minimum overlap to count plane-plane as contact


@dataclass(frozen=True)
class CoaxialCylinderPair:
    a: CylinderFace
    b: CylinderFace
    axial_overlap: float


@dataclass(frozen=True)
class CoincidentPlanePair:
    a: PlaneFace
    b: PlaneFace
    antiparallel: bool  # True when the outward normals oppose — i.e. real face-to-face contact
    overlap_area: float  # contact-area estimate: min of the two face areas (see find_contacts)


@dataclass(frozen=True)
class PartContact:
    """All contact features between an ordered pair of parts (i, j)."""
    i: int
    j: int
    coaxial_cylinders: list[CoaxialCylinderPair]
    coincident_planes: list[CoincidentPlanePair]


def _aabb(shape: TopoDS_Shape, slack: float) -> tuple[np.ndarray, np.ndarray]:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, True)
    box.SetGap(slack)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])


def _aabb_overlap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))


def _axial_overlap(a: CylinderFace, b: CylinderFace) -> float:
    """Length of overlap between the two cylinders measured along their (shared) axis."""
    axis = a.axis_dir
    a_proj = np.dot(a.axis_point, axis)
    b_proj = np.dot(b.axis_point, axis)
    lo = max(a_proj - a.axial_extent / 2, b_proj - b.axial_extent / 2)
    hi = min(a_proj + a.axial_extent / 2, b_proj + b.axial_extent / 2)
    return max(0.0, hi - lo)


def _coaxial(a: CylinderFace, b: CylinderFace, tol: Tolerances) -> bool:
    """True if two cylinders share an axis line (parallel + zero offset within tol)."""
    if abs(float(np.dot(a.axis_dir, b.axis_dir))) < tol.angle_cos:
        return False
    if abs(a.radius - b.radius) > tol.radius:
        return False
    # Perpendicular distance between two parallel axes: |(p_b - p_a) - ((p_b - p_a)·d) d|
    delta = b.axis_point - a.axis_point
    perp = delta - np.dot(delta, a.axis_dir) * a.axis_dir
    return bool(np.linalg.norm(perp) <= tol.pos)


def _plane_coincident(a: PlaneFace, b: PlaneFace, tol: Tolerances) -> bool:
    """Two planes are 'coincident' when their normals are parallel (signed or not)
    and the perpendicular distance between them is within tol.pos."""
    if abs(float(np.dot(a.normal, b.normal))) < tol.angle_cos:
        return False
    return abs(float(np.dot(b.point - a.point, a.normal))) <= tol.pos


def find_contacts(
    shapes: list[TopoDS_Shape],
    surfaces: list[PartSurfaces],
    tol: Tolerances = Tolerances(),
) -> list[PartContact]:
    boxes = [_aabb(s, slack=tol.pos) for s in shapes]
    out: list[PartContact] = []
    n = len(shapes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _aabb_overlap(boxes[i], boxes[j]):
                continue
            si, sj = surfaces[i], surfaces[j]
            cyl_pairs: list[CoaxialCylinderPair] = []
            for ca in si.cylinders:
                for cb in sj.cylinders:
                    if not _coaxial(ca, cb, tol):
                        continue
                    overlap = _axial_overlap(ca, cb)
                    if overlap < tol.min_axial_overlap:
                        continue
                    cyl_pairs.append(CoaxialCylinderPair(a=ca, b=cb, axial_overlap=overlap))
            pln_pairs: list[CoincidentPlanePair] = []
            for pa in si.planes:
                for pb in sj.planes:
                    if not _plane_coincident(pa, pb, tol):
                        continue
                    # min(area) over-estimates the shared area for laterally offset
                    # faces (the broad phase is per-part, not per-face); a projected
                    # polygon intersection is the planned refinement.
                    overlap_area = min(pa.area, pb.area)
                    if overlap_area < tol.min_plane_overlap_area:
                        continue
                    pln_pairs.append(
                        CoincidentPlanePair(
                            a=pa,
                            b=pb,
                            antiparallel=float(np.dot(pa.normal, pb.normal)) < 0.0,
                            overlap_area=overlap_area,
                        )
                    )
            if cyl_pairs or pln_pairs:
                out.append(PartContact(i=i, j=j, coaxial_cylinders=cyl_pairs, coincident_planes=pln_pairs))
    return out
