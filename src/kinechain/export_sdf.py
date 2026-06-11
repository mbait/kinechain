"""Emit an SDF (Gazebo) model from a KinematicTree.

Shares the KinematicTree IR and mesh export with the MJCF emitter. Two properties
of SDF make emission simpler than MJCF:

    * Links are flat siblings inside <model>; joints reference parent and child
      links by name, so no body-tree recursion is needed. Fixed joints become
      explicit <joint type="fixed"> elements instead of body nesting.
    * Kinematic loops are first-class: loop-closure joints are emitted as ordinary
      <joint> elements rather than being demoted to equality constraints.

Conventions mirror the MJCF emitter:
    * Lengths scaled mm → m at the output boundary; all link poses are the world
      origin (meshes are already in world coordinates), so joint poses — which SDF
      resolves relative to the child link frame — can carry world coordinates
      directly.
    * The root link is welded to the world via an explicit fixed joint (SDF allows
      "world" as a joint parent), matching the grounded-root MJCF output.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .export_common import MM_TO_M, export_part_meshes, inertial_properties, vec
from .graph import KinematicTree
from .joints import Joint, JointType

_SDF_JOINT_TYPE = {
    JointType.REVOLUTE: "revolute",
    JointType.PRISMATIC: "prismatic",
    JointType.FIXED: "fixed",
}


def write_sdf(tree: KinematicTree, out_path: Path, *, model_name: str = "kinechain") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_keys = export_part_meshes(tree.parts, out_path.parent / "meshes")

    sdf = ET.Element("sdf", version="1.8")
    model = ET.SubElement(sdf, "model", name=model_name)

    for part, key in zip(tree.parts, part_keys):
        _emit_link(model, part.shape, key)

    # Ground the root, mirroring the MJCF emitter's world-welded root body.
    anchor = ET.SubElement(model, "joint", name="world_anchor", type="fixed")
    ET.SubElement(anchor, "parent").text = "world"
    ET.SubElement(anchor, "child").text = part_keys[tree.root]

    # Tree and loop joints alike are plain SDF joints.
    for joint in [*tree.joints, *tree.loop_joints]:
        _emit_joint(model, joint, part_keys)

    tree_xml = ET.ElementTree(sdf)
    ET.indent(tree_xml, space="  ")
    tree_xml.write(out_path, xml_declaration=True, encoding="utf-8")
    return out_path


def _emit_link(model: ET.Element, shape, key: str) -> None:
    link = ET.SubElement(model, "link", name=key)
    ET.SubElement(link, "pose").text = "0 0 0 0 0 0"

    # Exact inertial from B-Rep volume integration. The inertial <pose> places the
    # frame at the COM; the tensor is about the COM in that frame's (link-aligned) axes.
    mass, com, tensor = inertial_properties(shape)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "pose").text = f"{vec(com)} 0 0 0"
    ET.SubElement(inertial, "mass").text = f"{mass:.6g}"
    inertia = ET.SubElement(inertial, "inertia")
    for tag, value in (
        ("ixx", tensor[0, 0]), ("ixy", tensor[0, 1]), ("ixz", tensor[0, 2]),
        ("iyy", tensor[1, 1]), ("iyz", tensor[1, 2]), ("izz", tensor[2, 2]),
    ):
        ET.SubElement(inertia, tag).text = f"{value:.6g}"

    for kind in ("visual", "collision"):
        el = ET.SubElement(link, kind, name=f"{key}_{kind}")
        geometry = ET.SubElement(el, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        ET.SubElement(mesh, "uri").text = f"meshes/{key}.stl"
        ET.SubElement(mesh, "scale").text = f"{MM_TO_M} {MM_TO_M} {MM_TO_M}"


def _emit_joint(model: ET.Element, joint: Joint, part_keys: list[str]) -> None:
    # Parent+child naming: with loop joints a child name alone can repeat (a loop
    # child already has a tree joint), but at most one joint exists per part pair.
    el = ET.SubElement(
        model,
        "joint",
        name=f"{part_keys[joint.parent]}__{part_keys[joint.child]}",
        type=_SDF_JOINT_TYPE[joint.type],
    )
    ET.SubElement(el, "parent").text = part_keys[joint.parent]
    ET.SubElement(el, "child").text = part_keys[joint.child]
    # Relative to the child link frame == world frame (all links sit at the origin).
    ET.SubElement(el, "pose").text = f"{vec(joint.axis_point, scale=MM_TO_M)} 0 0 0"
    if joint.type is not JointType.FIXED:
        axis = ET.SubElement(el, "axis")
        ET.SubElement(axis, "xyz").text = vec(joint.axis_dir)
