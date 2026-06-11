"""Joint classification from contact features.

Rules are applied most-constraining first: FIXED → PRISMATIC → REVOLUTE.
A rule that doesn't match returns None rather than guessing, so unmatched part pairs
stay unjoined and are surfaced in diagnostics instead of being silently welded.

FIXED — a large face-to-face planar contact (antiparallel normals) plus either
    (a) at least one bolt-like coaxial cylinder pair (axial overlap shorter than the
        diameter — a stud or bolt stub, not a bearing bore), or
    (b) contact-plane normals that span 3D, i.e. no direction perpendicular to all of
        them exists — every translation is blocked (pocket-style over-constraint).

PRISMATIC — two or more large face-to-face planar contacts with non-parallel normals
    that all share a common perpendicular direction s (the cross product of the first
    two distinct normals). Such contacts block both rotations and every translation
    except along s ⇒ slide along s.

REVOLUTE — a coaxial-cylinder pair with axial overlap of at least one diameter (a
    "long enough" shared bore — not a bolt-stub) and no face-to-face planar contact
    whose normal is parallel to the cylinder axis (a shoulder would lock rotation
    against friction-fit interpretation).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .contact import CoaxialCylinderPair, CoincidentPlanePair, PartContact


class JointType(str, Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"


@dataclass(frozen=True)
class Joint:
    type: JointType
    parent: int        # part index in the AssemblyModel.parts list
    child: int
    axis_point: np.ndarray  # any point on the joint axis (world coords)
    axis_dir: np.ndarray    # unit vector


# Heuristic: a "bolt-like" cylinder is one whose axial overlap is shorter than its diameter.
# Such pairs lean toward FIXED (with a shoulder plane) rather than revolute. Bare numbers used
# liberally for the MVP — a later refactor can centralize them in a Tolerances dataclass.
_BOLT_ASPECT_THRESHOLD = 1.0    # overlap / diameter; below this we treat the bore as short
_MIN_PLANE_CONTACT_AREA = 50.0  # mm² — minimum face-to-face contact to count as "large"
_PARALLEL_COS = 0.999           # cos(2.5°) — same as Tolerances.angle_cos
_PERP_SIN = 0.044               # sin(2.5°) — |dot| below this counts as perpendicular


def _is_bolt_like(cyl: CoaxialCylinderPair) -> bool:
    radius = max(cyl.a.radius, cyl.b.radius)  # the larger radius is the bore
    return cyl.axial_overlap < _BOLT_ASPECT_THRESHOLD * (2 * radius)


def _contact_planes(contact: PartContact) -> list[CoincidentPlanePair]:
    """Face-to-face plane pairs large enough to be load-bearing contact."""
    return [
        pp
        for pp in contact.coincident_planes
        if pp.antiparallel and pp.overlap_area >= _MIN_PLANE_CONTACT_AREA
    ]


def _distinct_normals(planes: list[CoincidentPlanePair]) -> list[np.ndarray]:
    """Contact-plane normals deduplicated up to sign — parallel planes constrain alike."""
    dirs: list[np.ndarray] = []
    for pp in planes:
        n = pp.a.normal
        if not any(abs(float(np.dot(n, d))) > _PARALLEL_COS for d in dirs):
            dirs.append(n)
    return dirs


def _slide_direction(normals: list[np.ndarray]) -> np.ndarray | None:
    """Unit direction perpendicular to every contact normal — the single translation
    the plane contacts leave free — or None when the normals span 3D (over-constrained).
    Requires at least two distinct normals (one plane leaves a 3-DOF planar pair, not
    a slide)."""
    if len(normals) < 2:
        return None
    s = np.cross(normals[0], normals[1])
    s = s / np.linalg.norm(s)
    for n in normals[2:]:
        if abs(float(np.dot(n, s))) > _PERP_SIN:
            return None
    return s


def _plane_locks_axis(contact: PartContact, axis_dir: np.ndarray) -> bool:
    """True if a face-to-face plane pair has its normal parallel to the cylinder axis —
    a shoulder that argues against a free-running fit. Same-direction (flush) coincident
    faces are not contacts and don't count."""
    for pp in contact.coincident_planes:
        if not pp.antiparallel:
            continue
        if abs(float(np.dot(pp.a.normal, axis_dir))) > _PARALLEL_COS:
            return True
    return False


def _match_fixed(contact: PartContact) -> Joint | None:
    planes = _contact_planes(contact)
    if not planes:
        return None
    has_bolt = any(_is_bolt_like(cyl) for cyl in contact.coaxial_cylinders)
    # Two or more distinct contact normals leave at most one translation free; when
    # even that one is gone (normals span 3D), the pair is welded by geometry alone.
    normals = _distinct_normals(planes)
    over_constrained = len(normals) >= 2 and _slide_direction(normals) is None
    if not (has_bolt or over_constrained):
        return None
    # Anchor the (motionless) joint frame on the dominant contact plane.
    anchor = max(planes, key=lambda pp: pp.overlap_area)
    return Joint(
        type=JointType.FIXED,
        parent=contact.i,
        child=contact.j,
        axis_point=anchor.a.point.copy(),
        axis_dir=anchor.a.normal.copy(),
    )


def _match_prismatic(contact: PartContact) -> Joint | None:
    planes = _contact_planes(contact)
    normals = _distinct_normals(planes)
    s = _slide_direction(normals)
    if s is None:
        return None
    anchor = max(planes, key=lambda pp: pp.overlap_area)
    return Joint(
        type=JointType.PRISMATIC,
        parent=contact.i,
        child=contact.j,
        axis_point=anchor.a.point.copy(),
        axis_dir=s,
    )


def _match_revolute(contact: PartContact) -> Joint | None:
    for cyl in contact.coaxial_cylinders:
        if _is_bolt_like(cyl):
            continue
        if _plane_locks_axis(contact, cyl.a.axis_dir):
            continue
        return Joint(
            type=JointType.REVOLUTE,
            parent=contact.i,
            child=contact.j,
            axis_point=cyl.a.axis_point.copy(),
            axis_dir=cyl.a.axis_dir.copy(),
        )
    return None


def classify(contact: PartContact) -> Joint | None:
    """Apply joint-pattern rules to a single part pair's contact features.

    Most-constraining rule first. Returns None if no rule matches — caller should
    treat the pair as "unjoined" and surface it in diagnostics.
    """
    return _match_fixed(contact) or _match_prismatic(contact) or _match_revolute(contact)


def classify_all(contacts: list[PartContact]) -> list[Joint]:
    out: list[Joint] = []
    for c in contacts:
        j = classify(c)
        if j is not None:
            out.append(j)
    return out
