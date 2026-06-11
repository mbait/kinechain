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

from .export_common import MM_TO_M, export_part_meshes, inertial_properties, vec
from .graph import KinematicTree
from .io_step import Part
from .joints import Joint, JointType


def write_mjcf(tree: KinematicTree, out_path: Path, *, model_name: str = "kinechain") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_keys = export_part_meshes(tree.parts, out_path.parent / "meshes")

    # Build the joints-by-child map for tree traversal
    joints_by_child: dict[int, Joint] = {j.child: j for j in tree.joints}
    children_of: dict[int, list[int]] = {}
    for j in tree.joints:
        children_of.setdefault(j.parent, []).append(j.child)

    mujoco = ET.Element("mujoco", model=model_name)
    ET.SubElement(mujoco, "compiler", meshdir="meshes", angle="radian")

    asset = ET.SubElement(mujoco, "asset")
    for key in part_keys:
        ET.SubElement(asset, "mesh", name=key, file=f"{key}.stl", scale=f"{MM_TO_M} {MM_TO_M} {MM_TO_M}")

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
                anchor=vec(lj.axis_point, scale=MM_TO_M),
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

    # Exact inertial from B-Rep volume integration: COM as the inertial frame origin,
    # full tensor about the COM (MJCF fullinertia order: ixx iyy izz ixy ixz iyz).
    mass, com, inertia = inertial_properties(parts[part_idx].shape)
    ET.SubElement(
        body,
        "inertial",
        pos=vec(com),
        mass=f"{mass:.6g}",
        fullinertia=" ".join(
            f"{x:.6g}"
            for x in (
                inertia[0, 0], inertia[1, 1], inertia[2, 2],
                inertia[0, 1], inertia[0, 2], inertia[1, 2],
            )
        ),
    )

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
                pos=vec(joint.axis_point, scale=MM_TO_M),
                axis=vec(joint.axis_dir),
            )
        # JointType.FIXED: no <joint> element — body is rigidly attached to its parent.

    ET.SubElement(body, "geom", type="mesh", mesh=key)

    for child_idx in children_of.get(part_idx, []):
        _emit_body(body, child_idx, parts, part_keys, joints_by_child, children_of)
