"""Extract analytic surfaces (Plane, Cylinder) from a part's B-Rep faces.

Only analytic types relevant to the MVP joint rules are emitted. Cones, spheres,
toroids, and free-form surfaces are skipped — they can be added later as new
joint patterns are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_Orientation, TopAbs_ShapeEnum
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape


@dataclass(frozen=True)
class PlaneFace:
    """A planar face: oriented by its outward normal, located by any point on it.

    The normal accounts for the face's topological orientation, so it always points
    away from the part's material. Two parts in face-to-face contact therefore have
    antiparallel normals.
    """
    point: np.ndarray   # shape (3,)
    normal: np.ndarray  # unit vector, shape (3,)
    area: float


@dataclass(frozen=True)
class CylinderFace:
    """A cylindrical face: axis line + radius, with axial extent measured along the axis."""
    axis_point: np.ndarray  # any point on the axis, shape (3,)
    axis_dir: np.ndarray    # unit vector along the axis, shape (3,)
    radius: float
    axial_extent: float     # length of the face's projection onto the axis (mm)
    area: float


@dataclass(frozen=True)
class PartSurfaces:
    planes: list[PlaneFace]
    cylinders: list[CylinderFace]


def _face_area(face: TopoDS_Face) -> float:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


def _to_np(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


def _axial_extent(face: TopoDS_Face, axis_point: np.ndarray, axis_dir: np.ndarray) -> float:
    """Project every triangulated vertex of the face onto the axis and return the spread.

    Uses the BRep tool's discretization rather than tessellating fresh: for the MVP it's
    enough to get a robust scalar that distinguishes a short bolt-stub from a long shaft.
    """
    loc = type(face).TShape  # unused, kept to remind that BRep_Tool.Triangulation takes a TopLoc_Location ref
    from OCP.TopLoc import TopLoc_Location
    location = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, location)
    if tri is None:
        # Fall back to UV bounds — coarse, but the face must at least have a parametric range.
        adaptor = BRepAdaptor_Surface(face)
        u1, u2, v1, v2 = adaptor.FirstUParameter(), adaptor.LastUParameter(), adaptor.FirstVParameter(), adaptor.LastVParameter()
        # For a cylinder, V is along the axis.
        return abs(v2 - v1)
    trsf = location.Transformation()
    projections: list[float] = []
    for i in range(1, tri.NbNodes() + 1):
        node = tri.Node(i).Transformed(trsf)
        v = _to_np(node) - axis_point
        projections.append(float(np.dot(v, axis_dir)))
    if not projections:
        return 0.0
    return max(projections) - min(projections)


def extract_surfaces(shape: TopoDS_Shape, *, mesh_deflection: float = 0.5) -> PartSurfaces:
    """Walk faces of a shape and return the planar / cylindrical ones with parameters.

    The shape is tessellated lazily (via `BRepMesh_IncrementalMesh`) to give us vertex
    samples for axial-extent estimation. Triangulation is cached on the shape, so a
    later mesh export reuses it.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    BRepMesh_IncrementalMesh(shape, mesh_deflection, False, 0.5, True)

    planes: list[PlaneFace] = []
    cylinders: list[CylinderFace] = []

    exp = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        adaptor = BRepAdaptor_Surface(face)
        stype = adaptor.GetType()
        if stype == GeomAbs_SurfaceType.GeomAbs_Plane:
            pln = adaptor.Plane()
            loc = pln.Location()
            ax = pln.Axis().Direction()
            normal = _to_np(ax)
            # The geometric surface normal ignores topology; a REVERSED face has its
            # material on the surface-normal side, so flip to get the outward normal.
            if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED:
                normal = -normal
            planes.append(
                PlaneFace(
                    point=_to_np(loc),
                    normal=normal,
                    area=_face_area(face),
                )
            )
        elif stype == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cyl = adaptor.Cylinder()
            loc = cyl.Location()
            ax = cyl.Axis().Direction()
            axis_point = _to_np(loc)
            axis_dir = _to_np(ax)
            cylinders.append(
                CylinderFace(
                    axis_point=axis_point,
                    axis_dir=axis_dir,
                    radius=cyl.Radius(),
                    axial_extent=_axial_extent(face, axis_point, axis_dir),
                    area=_face_area(face),
                )
            )
        exp.Next()
    return PartSurfaces(planes=planes, cylinders=cylinders)
