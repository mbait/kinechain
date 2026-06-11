"""Structural tests for the SDF emitter across all fixtures.

No Gazebo/sdformat runtime is assumed, so assertions parse the emitted XML and
check SDF semantics: flat links, joints referencing parent/child by name, an
explicit world anchor, fixed joints as real elements (unlike MJCF's body nesting),
and loop joints emitted as ordinary joints (unlike MJCF's equality constraints).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from kinechain.contact import find_contacts
from kinechain.export_sdf import write_sdf
from kinechain.graph import build_tree
from kinechain.io_step import load_step
from kinechain.joints import classify_all
from kinechain.surfaces import extract_surfaces


def _emit(step_path: Path, out: Path) -> ET.Element:
    model = load_step(step_path)
    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    contacts = find_contacts(shapes, surfaces)
    joints = classify_all(contacts)
    tree = build_tree(model, joints)
    path = write_sdf(tree, out)
    root = ET.parse(path).getroot()
    assert root.tag == "sdf"
    return root.find("model")


def _axis(joint: ET.Element) -> np.ndarray:
    return np.array([float(x) for x in joint.find("axis").findtext("xyz").split()])


def test_hinge_sdf(hinge_step: Path, tmp_path: Path) -> None:
    model = _emit(hinge_step, tmp_path / "hinge.sdf")

    assert {ln.get("name") for ln in model.findall("link")} == {"hinge_bracket", "hinge_pin"}
    for link in model.findall("link"):
        assert link.find("visual/geometry/mesh/uri") is not None
        assert link.find("collision/geometry/mesh/uri") is not None
        assert float(link.findtext("inertial/mass")) > 0

    joints = model.findall("joint")
    world = [j for j in joints if j.findtext("parent") == "world"]
    assert len(world) == 1
    assert world[0].get("type") == "fixed"
    assert world[0].findtext("child") == "hinge_bracket"

    rev = [j for j in joints if j.get("type") == "revolute"]
    assert len(rev) == 1
    assert rev[0].findtext("parent") == "hinge_bracket"
    assert rev[0].findtext("child") == "hinge_pin"
    assert abs(float(np.dot(_axis(rev[0]), np.array([1.0, 0.0, 0.0])))) > 0.999

    assert (tmp_path / "meshes" / "hinge_bracket.stl").exists()
    assert (tmp_path / "meshes" / "hinge_pin.stl").exists()


def test_bolted_sdf_emits_explicit_fixed_joint(bolted_step: Path, tmp_path: Path) -> None:
    model = _emit(bolted_step, tmp_path / "bolted.sdf")

    # Unlike MJCF (body nesting), SDF represents the fixed joint explicitly.
    fixed = [
        j
        for j in model.findall("joint")
        if j.get("type") == "fixed" and j.findtext("parent") != "world"
    ]
    assert len(fixed) == 1
    assert fixed[0].findtext("parent") == "bolted_base"
    assert fixed[0].findtext("child") == "bolted_top"
    assert fixed[0].find("axis") is None  # a fixed joint has no axis


def test_slider_sdf(slider_step: Path, tmp_path: Path) -> None:
    model = _emit(slider_step, tmp_path / "slider.sdf")

    pris = [j for j in model.findall("joint") if j.get("type") == "prismatic"]
    assert len(pris) == 1
    assert abs(float(np.dot(_axis(pris[0]), np.array([1.0, 0.0, 0.0])))) > 0.999


def test_fourbar_sdf_emits_loop_joint_directly(fourbar_step: Path, tmp_path: Path) -> None:
    model = _emit(fourbar_step, tmp_path / "fourbar.sdf")

    # All four loop edges are ordinary revolute joints — including the
    # loop-closure edge that MJCF demotes to an equality constraint.
    rev = [j for j in model.findall("joint") if j.get("type") == "revolute"]
    assert len(rev) == 4
    names = {j.get("name") for j in rev}
    assert len(names) == 4  # parent__child naming keeps loop joints unique

    world = [j for j in model.findall("joint") if j.findtext("parent") == "world"]
    assert len(world) == 1
    assert world[0].findtext("child") == "fourbar_ground"


def test_cli_format_sdf(hinge_step: Path, tmp_path: Path) -> None:
    from kinechain.cli import main

    out = tmp_path / "hinge.sdf"
    assert main([str(hinge_step), "--out", str(out), "--format", "sdf"]) == 0
    assert out.exists()
    assert ET.parse(out).getroot().tag == "sdf"
