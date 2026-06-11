"""End-to-end test: a V-rail slide must classify as PRISMATIC along the rail axis.

The fixture has no cylinders, so this isolates the plane-only path: two 45° flank
contacts plus the flat seat — three contact normals, all perpendicular to X — must
yield a slide along X, and must NOT be over-constrained into FIXED.
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


def test_pipeline_yields_one_prismatic(slider_step: Path, tmp_path: Path) -> None:
    model = load_step(slider_step)
    assert {p.name for p in model.parts} == {"slider/rail", "slider/carriage"}

    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    contacts = find_contacts(shapes, surfaces)

    assert len(contacts) == 1
    c = contacts[0]
    assert c.coaxial_cylinders == []
    # The two angled flanks must register as face-to-face contact with non-parallel
    # normals — that is what licenses prismatic over a planar pair.
    face_to_face = [pp for pp in c.coincident_planes if pp.antiparallel]
    assert len(face_to_face) >= 2
    assert any(
        abs(float(np.dot(p.a.normal, q.a.normal))) < 0.999
        for k, p in enumerate(face_to_face)
        for q in face_to_face[k + 1 :]
    )

    joints = classify_all(contacts)
    assert len(joints) == 1
    j = joints[0]
    assert j.type is JointType.PRISMATIC

    # Slide direction must be the rail axis (world X), up to sign.
    assert abs(float(np.dot(j.axis_dir, np.array([1.0, 0.0, 0.0])))) > 0.999

    tree = build_tree(model, joints)
    assert len(tree.joints) == 1
    assert tree.loop_joints == []
    # The larger-volume part (rail) should be the root.
    assert model.parts[tree.root].name == "slider/rail"

    out = write_mjcf(tree, tmp_path / "slider.xml")
    content = out.read_text()
    assert 'type="slide"' in content


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="mujoco not installed (install kinechain[sim] to enable)",
)
def test_mjcf_loads_in_mujoco(slider_step: Path, tmp_path: Path) -> None:
    import mujoco

    model = load_step(slider_step)
    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    contacts = find_contacts(shapes, surfaces)
    joints = classify_all(contacts)
    tree = build_tree(model, joints)
    out = write_mjcf(tree, tmp_path / "slider.xml")

    mj_model = mujoco.MjModel.from_xml_path(str(out))
    assert mj_model.njnt == 1
    assert mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE
