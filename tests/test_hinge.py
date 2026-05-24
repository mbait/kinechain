"""End-to-end smoke test: STEP → joints → MJCF on the hinge fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from kinechain.contact import find_contacts
from kinechain.export_mjcf import write_mjcf
from kinechain.graph import build_tree
from kinechain.io_step import load_step
from kinechain.joints import JointType, classify_all
from kinechain.surfaces import extract_surfaces


def test_pipeline_yields_one_revolute(hinge_step: Path, tmp_path: Path) -> None:
    model = load_step(hinge_step)
    assert {p.name for p in model.parts} == {"hinge/bracket", "hinge/pin"}

    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    contacts = find_contacts(shapes, surfaces)
    joints = classify_all(contacts)

    assert len(joints) == 1
    j = joints[0]
    assert j.type is JointType.REVOLUTE

    # Axis should be parallel (or antiparallel) to world X
    assert abs(float(np.dot(j.axis_dir, np.array([1.0, 0.0, 0.0])))) > 0.999

    tree = build_tree(model, joints)
    assert len(tree.joints) == 1
    assert tree.loop_joints == []
    # The larger volume part (bracket) should be the root
    assert model.parts[tree.root].name == "hinge/bracket"

    out = write_mjcf(tree, tmp_path / "hinge.xml")
    assert out.exists()
    content = out.read_text()
    assert 'type="hinge"' in content
    assert (tmp_path / "meshes" / "hinge_bracket.stl").exists()
    assert (tmp_path / "meshes" / "hinge_pin.stl").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="mujoco not installed (install kinechain[sim] to enable)",
)
def test_mjcf_loads_in_mujoco(hinge_step: Path, tmp_path: Path) -> None:
    import mujoco

    model = load_step(hinge_step)
    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    contacts = find_contacts(shapes, surfaces)
    joints = classify_all(contacts)
    tree = build_tree(model, joints)
    out = write_mjcf(tree, tmp_path / "hinge.xml")

    mj_model = mujoco.MjModel.from_xml_path(str(out))
    assert mj_model.njnt == 1
    # joint type 3 = hinge in MuJoCo (mjJNT_HINGE)
    assert mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_HINGE
