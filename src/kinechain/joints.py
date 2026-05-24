"""Joint classification from contact features.

MVP scope: revolute only (hinge slice). Fixed and prismatic to follow in later iterations
after the end-to-end pipeline lands.

Rule for this slice:
    A coaxial-cylinder pair with axial_overlap above the cylinder's radius (i.e. a
    "long enough" shared bore — not a bolt-stub) and no large planar contact whose
    normal is parallel to the cylinder axis ⇒ revolute joint about that axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .contact import PartContact


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
_BOLT_ASPECT_THRESHOLD = 1.0  # overlap / diameter; below this we treat the bore as short


def _is_long_enough_for_revolute(overlap: float, radius: float) -> bool:
    return overlap >= _BOLT_ASPECT_THRESHOLD * (2 * radius)


def _plane_locks_axis(contact: PartContact, axis_dir: np.ndarray) -> bool:
    """True if any coincident plane pair has its normal parallel to the cylinder axis —
    which would mean a shoulder is preventing rotation/sliding. None for the MVP hinge."""
    for pp in contact.coincident_planes:
        if abs(float(np.dot(pp.a.normal, axis_dir))) > 0.999:
            return True
    return False


def classify(contact: PartContact) -> Joint | None:
    """Apply joint-pattern rules to a single part pair's contact features.

    Returns None if no rule matches — caller should treat the pair as "unjoined" and
    surface it in diagnostics.
    """
    for cyl in contact.coaxial_cylinders:
        # Use the larger radius as the canonical one (the bore).
        radius = max(cyl.a.radius, cyl.b.radius)
        if not _is_long_enough_for_revolute(cyl.axial_overlap, radius):
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


def classify_all(contacts: list[PartContact]) -> list[Joint]:
    out: list[Joint] = []
    for c in contacts:
        j = classify(c)
        if j is not None:
            out.append(j)
    return out
