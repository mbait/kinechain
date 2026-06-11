"""End-to-end test: a four-bar linkage must yield a spanning tree plus one loop joint.

This is the cycle-handling test: four revolute joints form a closed loop, which no
simulator tree can represent directly. Expect three tree joints and one loop-closure
joint, emitted as an MJCF <equality><connect> constraint.
"""

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


def _run_pipeline(step_path: Path):
    model = load_step(step_path)
    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    contacts = find_contacts(shapes, surfaces)
    joints = classify_all(contacts)
    return model, joints


def test_pipeline_yields_four_revolutes_with_one_loop(fourbar_step: Path, tmp_path: Path) -> None:
    model, joints = _run_pipeline(fourbar_step)
    assert {p.name for p in model.parts} == {
        "fourbar/ground",
        "fourbar/crank",
        "fourbar/coupler",
        "fourbar/rocker",
    }

    assert len(joints) == 4
    assert all(j.type is JointType.REVOLUTE for j in joints)
    z = np.array([0.0, 0.0, 1.0])
    assert all(abs(float(np.dot(j.axis_dir, z))) > 0.999 for j in joints)

    # The four joints close a loop: ground-crank-coupler-rocker-ground. Each part
    # participates in exactly two joints.
    degree: dict[int, int] = {}
    for j in joints:
        degree[j.parent] = degree.get(j.parent, 0) + 1
        degree[j.child] = degree.get(j.child, 0) + 1
    assert sorted(degree.values()) == [2, 2, 2, 2]

    tree = build_tree(model, joints)
    assert len(tree.joints) == 3
    assert len(tree.loop_joints) == 1
    # The widest/longest link (ground) should be the root.
    assert model.parts[tree.root].name == "fourbar/ground"

    out = write_mjcf(tree, tmp_path / "fourbar.xml")
    content = out.read_text()
    assert content.count('type="hinge"') == 3
    assert "<equality>" in content
    assert "<connect" in content


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="mujoco not installed (install kinechain[sim] to enable)",
)
def test_mjcf_loads_in_mujoco(fourbar_step: Path, tmp_path: Path) -> None:
    import mujoco

    model, joints = _run_pipeline(fourbar_step)
    tree = build_tree(model, joints)
    out = write_mjcf(tree, tmp_path / "fourbar.xml")

    mj_model = mujoco.MjModel.from_xml_path(str(out))
    assert mj_model.njnt == 3
    assert all(t == mujoco.mjtJoint.mjJNT_HINGE for t in mj_model.jnt_type)
    assert mj_model.neq == 1
    assert mj_model.eq_type[0] == mujoco.mjtEq.mjEQ_CONNECT
