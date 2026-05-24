"""Emit a MuJoCo MJCF XML model from a KinematicTree.

Each part becomes a `<body>` containing a mesh geom (exported as STL) and an inertia
attribute (computed from volume × density). Joints in the tree become `<joint>`
elements parented to the child body. Loop-closure joints become `<equality>` constraints.

Conventions:
    * Lengths are in millimetres on the STEP side; MJCF uses metres by default. We
      emit `<compiler ... meshdir="meshes" angle="radian"/>` and scale all lengths by
      `1e-3` so the model lives in metres without needing per-geom rescale.
    * The root body is welded to worldbody (no free joint). Add a free joint manually
      later if you want the chain to float.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.StlAPI import StlAPI_Writer

from .graph import KinematicTree
from .io_step import Part
from .joints import Joint, JointType


_MM_TO_M = 1e-3
_DEFAULT_DENSITY_KG_PER_M3 = 1000.0  # water


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _vec(v: np.ndarray, scale: float = 1.0) -> str:
    return " ".join(f"{x * scale:.6g}" for x in v)


def _mass(shape) -> float:
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume_mm3 = props.Mass()
    return volume_mm3 * 1e-9 * _DEFAULT_DENSITY_KG_PER_M3


def _export_stl(shape, path: Path, deflection_mm: float = 0.5) -> None:
    BRepMesh_IncrementalMesh(shape, deflection_mm, False, 0.5, True)
    writer = StlAPI_Writer()
    writer.ASCIIMode = False  # MuJoCo's STL decoder accepts binary STL only
    writer.Write(shape, str(path))


def write_mjcf(tree: KinematicTree, out_path: Path, *, model_name: str = "kinechain") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_path.parent / "meshes"
    mesh_dir.mkdir(exist_ok=True)

    # Export each part as STL into meshes/
    part_keys: list[str] = []
    for part in tree.parts:
        key = _safe_filename(part.name)
        part_keys.append(key)
        _export_stl(part.shape, mesh_dir / f"{key}.stl")

    # Build the joints-by-child map for tree traversal
    joints_by_child: dict[int, Joint] = {j.child: j for j in tree.joints}
    children_of: dict[int, list[int]] = {}
    for j in tree.joints:
        children_of.setdefault(j.parent, []).append(j.child)

    mujoco = ET.Element("mujoco", model=model_name)
    ET.SubElement(mujoco, "compiler", meshdir="meshes", angle="radian")

    asset = ET.SubElement(mujoco, "asset")
    for key in part_keys:
        ET.SubElement(asset, "mesh", name=key, file=f"{key}.stl", scale=f"{_MM_TO_M} {_MM_TO_M} {_MM_TO_M}")

    worldbody = ET.SubElement(mujoco, "worldbody")
    _emit_body(worldbody, tree.root, tree.parts, part_keys, joints_by_child, children_of)

    if tree.loop_joints:
        equality = ET.SubElement(mujoco, "equality")
        for lj in tree.loop_joints:
            ET.SubElement(
                equality,
                "connect",
                body1=part_keys[lj.parent],
                body2=part_keys[lj.child],
                anchor=_vec(lj.axis_point, scale=_MM_TO_M),
            )

    tree_xml = ET.ElementTree(mujoco)
    ET.indent(tree_xml, space="  ")
    tree_xml.write(out_path, xml_declaration=True, encoding="utf-8")
    return out_path


def _emit_body(
    parent_xml: ET.Element,
    part_idx: int,
    parts: list[Part],
    part_keys: list[str],
    joints_by_child: dict[int, Joint],
    children_of: dict[int, list[int]],
) -> None:
    key = part_keys[part_idx]
    body = ET.SubElement(parent_xml, "body", name=key, pos="0 0 0")

    # Inertial: simple mass from volume × default density. MuJoCo's compiler will
    # auto-derive an inertia tensor from the geom if we omit diaginertia/fullinertia.
    mass = _mass(parts[part_idx].shape)
    ET.SubElement(body, "inertial", pos="0 0 0", mass=f"{mass:.6g}", diaginertia=f"{mass * 1e-4:.6g} {mass * 1e-4:.6g} {mass * 1e-4:.6g}")

    if part_idx in joints_by_child:
        joint = joints_by_child[part_idx]
        kind = {
            JointType.REVOLUTE: "hinge",
            JointType.PRISMATIC: "slide",
        }.get(joint.type)
        if kind is not None:
            ET.SubElement(
                body,
                "joint",
                name=f"{key}_joint",
                type=kind,
                pos=_vec(joint.axis_point, scale=_MM_TO_M),
                axis=_vec(joint.axis_dir),
            )
        # JointType.FIXED: no <joint> element — body is rigidly attached to its parent.

    ET.SubElement(body, "geom", type="mesh", mesh=key)

    for child_idx in children_of.get(part_idx, []):
        _emit_body(body, child_idx, parts, part_keys, joints_by_child, children_of)
