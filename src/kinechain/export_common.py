"""Helpers shared by the MJCF and SDF emitters: naming, units, inertia, mesh export.

Conventions:
    * Lengths are in millimetres on the STEP side; both simulators use metres.
      Emitters scale all lengths by MM_TO_M at the output boundary.
    * Inertial properties come from exact B-Rep volume integration
      (`BRepGProp.VolumeProperties`) at a default density: mass, centre of mass,
      and the full inertia tensor about the COM (parallel-axis transfer from the
      origin-referenced moment matrix OCCT returns).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.StlAPI import StlAPI_Writer

from .io_step import Part

MM_TO_M = 1e-3
DEFAULT_DENSITY_KG_PER_M3 = 1000.0  # water


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def vec(v: np.ndarray, scale: float = 1.0) -> str:
    return " ".join(f"{x * scale:.6g}" for x in v)


def inertial_properties(shape) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass (kg), centre of mass (m), and inertia tensor about the COM (kg·m², 3×3).

    OCCT integrates the B-Rep exactly with unit density; MatrixOfInertia is already
    referenced to the centre of mass, so no parallel-axis transfer is needed. We
    apply the default density and the mm→m unit change (lengths enter the inertia
    twice, so the volume factor 1e-9 picks up an extra 1e-6).
    """
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume_mm3 = props.Mass()
    c = np.array([props.CentreOfMass().X(), props.CentreOfMass().Y(), props.CentreOfMass().Z()])
    mat = props.MatrixOfInertia()
    i_com_mm5 = np.array([[mat.Value(r, col) for col in (1, 2, 3)] for r in (1, 2, 3)])

    mass = volume_mm3 * 1e-9 * DEFAULT_DENSITY_KG_PER_M3
    inertia = i_com_mm5 * DEFAULT_DENSITY_KG_PER_M3 * 1e-15
    return mass, c * MM_TO_M, inertia


def export_stl(shape, path: Path, deflection_mm: float = 0.5) -> None:
    BRepMesh_IncrementalMesh(shape, deflection_mm, False, 0.5, True)
    writer = StlAPI_Writer()
    writer.ASCIIMode = False  # MuJoCo's STL decoder accepts binary STL only
    writer.Write(shape, str(path))


def export_part_meshes(parts: list[Part], mesh_dir: Path) -> list[str]:
    """Export each part as `<key>.stl` into mesh_dir; return the per-part keys."""
    mesh_dir.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for part in parts:
        key = safe_filename(part.name)
        keys.append(key)
        export_stl(part.shape, mesh_dir / f"{key}.stl")
    return keys
