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
    min_joint_plane_area: float = 50.0  # mm² — "large" face-to-face contact for joint rules

    @classmethod
    def from_diagonal(cls, diag: float) -> "Tolerances":
        """Scale-relative tolerances derived from the assembly AABB diagonal (mm).

        The defaults above assume desktop-scale mechanisms; this constructor makes the
        same pipeline work on watch- or excavator-scale assemblies. Angular tolerance
        is dimensionless and stays fixed.
        """
        return cls(
            pos=2e-3 * diag,
            radius=2e-3 * diag,
            min_axial_overlap=5e-3 * diag,
            min_plane_overlap_area=(5e-3 * diag) ** 2,
            min_joint_plane_area=(5e-2 * diag) ** 2,
        )


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


def assembly_diagonal(shapes: list[TopoDS_Shape]) -> float:
    """AABB diagonal of the whole assembly (mm) — the scale input for Tolerances."""
    boxes = [_aabb(s, slack=0.0) for s in shapes]
    lo = np.min(np.array([b[0] for b in boxes]), axis=0)
    hi = np.max(np.array([b[1] for b in boxes]), axis=0)
    return float(np.linalg.norm(hi - lo))


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


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal (u, v) basis spanning the plane with the given normal."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _poly_area(pts: list[np.ndarray]) -> float:
    """Shoelace area of a 2D polygon (absolute value)."""
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for i in range(len(pts)):
        total += _cross2(pts[i], pts[(i + 1) % len(pts)])
    return abs(total) / 2.0


def _clip_to_triangle(subject: list[np.ndarray], clip_tri: list[np.ndarray]) -> list[np.ndarray]:
    """Sutherland–Hodgman: clip a polygon against a CCW triangle (both convex)."""
    output = subject
    for i in range(3):
        if not output:
            return []
        a, b = clip_tri[i], clip_tri[(i + 1) % 3]
        edge = b - a
        pts, output = output, []
        prev = pts[-1]
        prev_in = _cross2(edge, prev - a) >= 0.0
        for cur in pts:
            cur_in = _cross2(edge, cur - a) >= 0.0
            if cur_in != prev_in:
                d1 = _cross2(edge, prev - a)
                d2 = _cross2(edge, cur - a)
                output.append(prev + (d1 / (d1 - d2)) * (cur - prev))
            if cur_in:
                output.append(cur)
            prev, prev_in = cur, cur_in
    return output


def _projected_overlap_area(a: PlaneFace, b: PlaneFace) -> float:
    """Exact shared contact area of two (near-)coincident planar faces.

    Both faces' triangulations are projected into a's plane and intersected
    triangle-by-triangle (convex clipping). Exact up to the tessellation, and
    correctly returns 0 for coplanar faces that do not overlap laterally.
    """
    if a.triangles.size == 0 or b.triangles.size == 0:
        return 0.0
    u, v = _plane_basis(a.normal)

    def to2d(tris: np.ndarray) -> np.ndarray:
        d = tris - a.point
        return np.stack([d @ u, d @ v], axis=-1)  # (n, 3, 2)

    total = 0.0
    for ta in to2d(a.triangles):
        # Orient the clip triangle CCW; skip degenerate slivers.
        signed = _cross2(ta[1] - ta[0], ta[2] - ta[0])
        if abs(signed) < 1e-12:
            continue
        clip = list(ta) if signed > 0 else [ta[0], ta[2], ta[1]]
        for tb in to2d(b.triangles):
            total += _poly_area(_clip_to_triangle(list(tb), clip))
    return total


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
                    overlap_area = _projected_overlap_area(pa, pb)
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
