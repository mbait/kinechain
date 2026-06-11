"""End-to-end test: bolted plates must classify as FIXED, never revolute.

This is the false-positive guard for the revolute rule: the fixture contains four
coaxial cylinder pairs that superficially look like hinges, but the bolt-aspect
heuristic and the shoulder-plane lock must both steer classification to FIXED.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from kinechain.contact import find_contacts
from kinechain.export_mjcf import write_mjcf
from kinechain.graph import build_tree
from kinechain.io_step import load_step
from kinechain.joints import JointType, classify_all
from kinechain.surfaces import extract_surfaces


def test_pipeline_yields_one_fixed(bolted_step: Path, tmp_path: Path) -> None:
    model = load_step(bolted_step)
    assert {p.name for p in model.parts} == {"bolted/base", "bolted/top"}

    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    contacts = find_contacts(shapes, surfaces)

    assert len(contacts) == 1
    c = contacts[0]
    # All four stud/hole pairs are detected as coaxial cylinders...
    assert len(c.coaxial_cylinders) == 4
    # ...and the plates share at least one face-to-face (antiparallel) planar contact.
    assert any(pp.antiparallel for pp in c.coincident_planes)

    joints = classify_all(contacts)
    assert len(joints) == 1
    assert joints[0].type is JointType.FIXED

    tree = build_tree(model, joints)
    assert len(tree.joints) == 1
    assert tree.loop_joints == []
    # The larger-volume part (base plate + studs) should be the root.
    assert model.parts[tree.root].name == "bolted/base"

    out = write_mjcf(tree, tmp_path / "bolted.xml")
    content = out.read_text()
    # Fixed joint: the child body nests rigidly in the parent — no <joint> element.
    assert "<joint" not in content
    assert content.index('name="bolted_base"') < content.index('name="bolted_top"')


@pytest.mark.skipif(
    importlib.util.find_spec("mujoco") is None,
    reason="mujoco not installed (install kinechain[sim] to enable)",
)
def test_mjcf_loads_in_mujoco(bolted_step: Path, tmp_path: Path) -> None:
    import mujoco

    model = load_step(bolted_step)
    shapes = [p.shape for p in model.parts]
    surfaces = [extract_surfaces(s) for s in shapes]
    contacts = find_contacts(shapes, surfaces)
    joints = classify_all(contacts)
    tree = build_tree(model, joints)
    out = write_mjcf(tree, tmp_path / "bolted.xml")

    mj_model = mujoco.MjModel.from_xml_path(str(out))
    assert mj_model.njnt == 0  # fixed joint emits no MuJoCo joint
    assert mj_model.nbody == 3  # world + base + top
